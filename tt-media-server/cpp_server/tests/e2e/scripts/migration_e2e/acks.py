# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""Produce one migration request and count matching SUCCESSFUL acks.

Designed for the broadcast pattern set up by migration_worker_rank_launch.sh:
each prefill worker is in its own Kafka consumer group, so a single request
fans out to every prefill (one ack per worker). The caller decides how many
acks to expect (== NUM_PREFILL).

Building blocks (`build_ack_consumer`, `wait_for_assignment`, `produce_one`,
`drain_acks`) are split out so future tests can compose them differently
without copy-pasting — e.g. produce N requests, assert per-id partitioning,
or check FAILED-status paths.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from confluent_kafka import Consumer, KafkaError, Producer

from migration_e2e import ACK_TOPIC, REQUEST_TOPIC
from migration_e2e.config import Config


def build_ack_consumer(brokers: str) -> Consumer:
    """Unique-group consumer that only sees ACKs produced after subscribe()."""
    # Unique group + auto.offset.reset=latest: only ACKs published AFTER
    # subscribe() are observable, so we won't pick up stale acks from a
    # previous test run that happened to reuse our migration_id.
    group = f"migration-ack-counter-{os.getpid()}-{int(time.time() * 1000)}"
    return Consumer(
        {
            "bootstrap.servers": brokers,
            "group.id": group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        }
    )


def wait_for_assignment(consumer: Consumer, timeout_sec: float = 5.0) -> None:
    """Block until the consumer has at least one partition assigned.

    Producing before assignment can lose the first ack on slow brokers; we
    poll lightly to let the rebalance settle and warn (but don't fail) if it
    doesn't.
    """
    deadline = time.monotonic() + timeout_sec
    while not consumer.assignment() and time.monotonic() < deadline:
        consumer.poll(timeout=0.2)
    if not consumer.assignment():
        print(
            f"WARN: consumer assignment empty after {timeout_sec:.0f}s; "
            "producing anyway",
            file=sys.stderr,
        )


def produce_one(brokers: str, payload: dict[str, Any]) -> None:
    """Send exactly one migration request to REQUEST_TOPIC. Raises on failure."""
    producer = Producer(
        {
            "bootstrap.servers": brokers,
            "linger.ms": 0,
            "enable.idempotence": False,
        }
    )
    err_box: list[Any] = []

    def on_delivery(err: Any, _msg: Any) -> None:
        if err is not None:
            err_box.append(err)

    producer.produce(
        REQUEST_TOPIC,
        json.dumps(payload).encode(),
        on_delivery=on_delivery,
    )
    producer.flush(timeout=10)
    if err_box:
        raise RuntimeError(f"producer delivery failed: {err_box[0]}")


def drain_acks(
    consumer: Consumer,
    migration_id: int,
    expected: int,
    timeout_sec: float,
) -> tuple[list[str], int]:
    """Poll ACK_TOPIC until `expected` matching acks arrive or timeout fires.

    Returns (statuses_matching_migration_id, unrelated_ack_count). The caller
    decides what counts as PASS (we just collect evidence).
    """
    deadline = time.monotonic() + timeout_sec
    statuses: list[str] = []
    unrelated = 0
    while time.monotonic() < deadline and len(statuses) < expected:
        msg = consumer.poll(timeout=0.5)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                print(f"!!! consumer poll error: {msg.error()}", file=sys.stderr)
            continue
        try:
            payload = json.loads(msg.value().decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, AttributeError):
            print(f"!!! unparseable ack: {msg.value()!r}", file=sys.stderr)
            continue
        if payload.get("migration_id") == migration_id:
            statuses.append(str(payload.get("status")))
            print(
                f"<- ack {len(statuses)}/{expected} "
                f"status={payload.get('status')} "
                f"(partition={msg.partition()} offset={msg.offset()})"
            )
        else:
            unrelated += 1
    return statuses, unrelated


def produce_migration_request(cfg: Config) -> tuple[int, Consumer]:
    """Subscribe to ACK_TOPIC, then send exactly one migration request.

    The ACK consumer is subscribed and waits for partition assignment BEFORE
    the produce, otherwise early acks can be lost on slow brokers (see
    `wait_for_assignment`). Returns (migration_id, live consumer) — the caller
    owns the consumer and must close it (e.g. via ExitStack).
    """
    # nanoseconds-since-epoch keeps the id unique across rapid re-runs. The
    # ack drainer matches on migration_id alone, so collisions would let
    # stale acks falsely satisfy a fresh request.
    migration_id = time.time_ns()
    payload = {
        "migration_id": migration_id,
        "src_slot": 0,
        "dst_slot": 1,
        "layer_id": 0,
        "position_start": 0,
        "position_end": 16,
    }

    consumer = build_ack_consumer(cfg.kafka_brokers)
    consumer.subscribe([ACK_TOPIC])
    wait_for_assignment(consumer)

    print(
        f"Producing 1 request migration_id={migration_id}, "
        f"expecting {cfg.num_prefill} acks..."
    )
    print(f"-> {REQUEST_TOPIC}: {json.dumps(payload)}")
    produce_one(cfg.kafka_brokers, payload)
    return migration_id, consumer


def count_acks(cfg: Config, migration_id: int, consumer: Consumer) -> bool:
    """Drain ACK_TOPIC and PASS iff cfg.num_prefill SUCCESSFUL acks arrive."""
    print(
        f"waiting up to {cfg.ack_timeout_sec:.0f}s for {cfg.num_prefill} "
        f"acks on {ACK_TOPIC} with migration_id={migration_id}..."
    )
    statuses, unrelated = drain_acks(
        consumer, migration_id, cfg.num_prefill, cfg.ack_timeout_sec
    )

    print(f"matched={len(statuses)} expected={cfg.num_prefill}")
    if unrelated:
        print(f"(saw {unrelated} unrelated acks during the window)")

    if len(statuses) != cfg.num_prefill:
        return False
    bad = [s for s in statuses if s != "SUCCESSFUL"]
    if bad:
        print(f"non-SUCCESSFUL statuses: {bad}", file=sys.stderr)
        return False
    return True
