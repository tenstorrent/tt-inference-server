#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""
Send one migration request, then count distinct ACKs from the cpp_server
prefill workers within a timeout.

Designed for the MooncakeKafkaMigration MPI test
(tests/e2e/scripts/run_kafka_migration_workers_mpi.sh). Each prefill worker is
launched in its own Kafka consumer group, so a single request fans out to all
prefills (one ack per worker); this helper asserts the expected count came
back for the test's migration_id.

The script subscribes to the ack topic from *latest* BEFORE producing the
request, so ACKs emitted while the test was still warming up don't get
counted. Acks are matched by migration_id (the only field tying them to the
request); we therefore use a unique migration_id per invocation.

Exit codes:
  0  -- exactly --expected-acks ACKs with status=SUCCESSFUL received
  1  -- timed out / wrong count / wrong status
  2  -- usage / argument / Kafka error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer


REQUEST_TOPIC = "kv-migration-requests"
ACK_TOPIC = "kv-migration-acks"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--brokers",
        default=os.environ.get("KAFKA_BROKERS", "kafka:9092"),
        help="bootstrap.servers (default: env KAFKA_BROKERS or kafka:9092)",
    )
    p.add_argument(
        "--migration-id",
        type=int,
        required=True,
        help="migration_id to send and match in ACKs",
    )
    p.add_argument(
        "--expected-acks",
        type=int,
        required=True,
        help="exact number of ACKs to wait for",
    )
    p.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
        help="wall-clock budget for ACKs to arrive (default: 30s)",
    )
    p.add_argument(
        "--src-slot", type=int, default=0,
    )
    p.add_argument(
        "--dst-slot", type=int, default=1,
    )
    p.add_argument(
        "--layer-id", type=int, default=0,
    )
    p.add_argument(
        "--position-start", type=int, default=0,
    )
    p.add_argument(
        "--position-end", type=int, default=16,
    )
    return p.parse_args()


def _build_consumer(brokers: str) -> Consumer:
    # Unique group + auto.offset.reset=latest means we only see ACKs published
    # AFTER subscribe(); a manual assign() against the high watermark would
    # be tighter but adds a metadata round-trip we don't need here.
    group = f"migration-ack-counter-{os.getpid()}-{int(time.time() * 1000)}"
    return Consumer(
        {
            "bootstrap.servers": brokers,
            "group.id": group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        }
    )


def _produce_request(brokers: str, payload: dict[str, Any]) -> bool:
    producer = Producer(
        {
            "bootstrap.servers": brokers,
            "linger.ms": 0,
            "enable.idempotence": False,
        }
    )
    delivered = False
    failed_err: Any = None

    def on_delivery(err: Any, msg: Any) -> None:
        nonlocal delivered, failed_err
        if err is None:
            delivered = True
        else:
            failed_err = err

    producer.produce(
        REQUEST_TOPIC,
        json.dumps(payload).encode(),
        on_delivery=on_delivery,
    )
    producer.flush(timeout=10)
    if not delivered:
        print(f"!!! producer delivery failed: {failed_err}", file=sys.stderr)
        return False
    return True


def _count_acks(
    consumer: Consumer, migration_id: int, expected: int, timeout_sec: float
) -> tuple[int, list[str], list[str]]:
    """Drain ACK topic until `expected` matching acks arrive or timeout fires.

    Returns (matched_count, statuses, unrelated_acks_seen).
    """
    deadline = time.monotonic() + timeout_sec
    matched = 0
    statuses: list[str] = []
    unrelated: list[str] = []
    while time.monotonic() < deadline and matched < expected:
        msg = consumer.poll(timeout=0.5)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                print(f"!!! consumer poll error: {msg.error()}", file=sys.stderr)
            continue
        try:
            raw = msg.value().decode("utf-8", errors="replace")
            payload = json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            print(f"!!! unparseable ack: {msg.value()!r}", file=sys.stderr)
            continue
        if payload.get("migration_id") == migration_id:
            matched += 1
            statuses.append(str(payload.get("status")))
            print(
                f"<- ack {matched}/{expected} "
                f"status={payload.get('status')} "
                f"(partition={msg.partition()} offset={msg.offset()})"
            )
        else:
            unrelated.append(raw)
    return matched, statuses, unrelated


def main() -> int:
    args = _parse_args()
    if args.expected_acks <= 0:
        print("ERROR: --expected-acks must be > 0", file=sys.stderr)
        return 2

    try:
        consumer = _build_consumer(args.brokers)
        consumer.subscribe([ACK_TOPIC])
    except KafkaException as exc:
        print(f"ERROR: cannot subscribe to {ACK_TOPIC}: {exc}", file=sys.stderr)
        return 2

    # Block briefly until the assignment is in place; otherwise produce() can
    # race the subscription and we'd miss the first ACK on slow CI.
    deadline = time.monotonic() + 5.0
    while not consumer.assignment() and time.monotonic() < deadline:
        consumer.poll(timeout=0.2)
    if not consumer.assignment():
        print(
            "WARN: consumer assignment empty after 5s; producing anyway",
            file=sys.stderr,
        )

    payload = {
        "migration_id": args.migration_id,
        "src_slot": args.src_slot,
        "dst_slot": args.dst_slot,
        "layer_id": args.layer_id,
        "position_start": args.position_start,
        "position_end": args.position_end,
    }
    print(f"-> {REQUEST_TOPIC}: {json.dumps(payload)}")
    if not _produce_request(args.brokers, payload):
        consumer.close()
        return 2

    print(
        f"waiting up to {args.timeout_sec:.0f}s for {args.expected_acks} acks "
        f"on {ACK_TOPIC} with migration_id={args.migration_id}..."
    )
    matched, statuses, unrelated = _count_acks(
        consumer, args.migration_id, args.expected_acks, args.timeout_sec
    )
    consumer.close()

    print(f"matched={matched} expected={args.expected_acks}")
    if unrelated:
        print(f"(saw {len(unrelated)} unrelated acks during the window)")

    if matched != args.expected_acks:
        print("RESULT: FAIL", file=sys.stderr)
        return 1

    failed_statuses = [s for s in statuses if s != "SUCCESSFUL"]
    if failed_statuses:
        print(
            f"RESULT: FAIL — non-SUCCESSFUL statuses: {failed_statuses}",
            file=sys.stderr,
        )
        return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
