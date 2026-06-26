#!/usr/bin/env python3
"""Kafka CLI for the cpp_server KvMigrationWorker dev loop.

Uses confluent-kafka (librdkafka bindings) -- the same client library the C++
code uses -- so behavior parity with the production messaging layer is
maintained.

Subcommands:
    setup           Create the app topics (kv-migration-requests, kv-migration-acks)
    produce         Publish one or many migration request messages
    tail            Stream messages from acks (default) or requests
    reset-offsets   Move a consumer group's committed offsets to earliest/latest
    status          Show broker/topic snapshot

Examples:
    python migration_cli.py setup
    python migration_cli.py produce --src-slot 0 --dst-slot 1 --pos-end 32
    python migration_cli.py produce --count 100 --rate 50 --random
    python migration_cli.py tail acks
    python migration_cli.py tail requests --from-beginning --max 10
    python migration_cli.py reset-offsets --to earliest
    python migration_cli.py status

Configuration:
    --brokers <host:port>     Override the bootstrap address
    KAFKA_BROKERS             Same, via env (default: kafka:9092)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any

from confluent_kafka import (
    ConsumerGroupTopicPartitions,
    Consumer,
    KafkaError,
    KafkaException,
    Producer,
    TopicPartition,
)
from confluent_kafka.admin import AdminClient, NewTopic


DEFAULT_BROKERS = os.environ.get("KAFKA_BROKERS", "kafka:9092")
REQUEST_TOPIC = "kv-migration-requests"
ACK_TOPIC = "kv-migration-acks"
DEFAULT_GROUP = "migration-workers"

APP_TOPICS = (REQUEST_TOPIC, ACK_TOPIC)


# ---------- helpers ----------------------------------------------------------


def _wait_for_broker(admin: AdminClient, timeout_s: float = 30.0) -> None:
    """Block until the broker responds to a metadata call."""
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            md = admin.list_topics(timeout=2.0)
            if md.brokers:
                return
        except KafkaException as exc:
            last_err = exc
        time.sleep(1.0)
    raise SystemExit(f"broker not ready after {timeout_s:.0f}s (last error: {last_err})")


def _build_request(args: argparse.Namespace, index: int) -> dict[str, Any]:
    if args.random:
        return {
            "migration_id": random.getrandbits(63),
            "src_slot": random.randint(0, 31),
            "dst_slot": random.randint(0, 31),
            "layer_id": random.randint(0, 95),
            "position_start": 0,
            "position_end": random.choice([16, 32, 64, 128, 256]),
        }
    base_id = (
        args.migration_id
        if args.migration_id is not None
        else time.time_ns()
    )
    return {
        "migration_id": base_id + index,
        "src_slot": args.src_slot,
        "dst_slot": args.dst_slot,
        "layer_id": args.layer_id,
        "position_start": args.position_start,
        "position_end": args.position_end,
    }


# ---------- subcommands ------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> None:
    admin = AdminClient({"bootstrap.servers": args.brokers})
    _wait_for_broker(admin)
    new_topics = [
        NewTopic(
            t,
            num_partitions=args.partitions,
            replication_factor=args.replication_factor,
        )
        for t in APP_TOPICS
    ]
    futures = admin.create_topics(new_topics, request_timeout=10)
    rc = 0
    for name, fut in futures.items():
        try:
            fut.result()
            print(f"created topic: {name}")
        except KafkaException as exc:
            err = exc.args[0]
            if err.code() == KafkaError.TOPIC_ALREADY_EXISTS:
                print(f"topic already exists: {name}")
            else:
                print(f"failed to create {name}: {err}", file=sys.stderr)
                rc = 1
    sys.exit(rc)


def cmd_produce(args: argparse.Namespace) -> None:
    producer = Producer(
        {
            "bootstrap.servers": args.brokers,
            "linger.ms": 0,
            "enable.idempotence": False,
        }
    )
    delivered = 0
    failed = 0

    def _on_delivery(err: Any, msg: Any) -> None:
        nonlocal delivered, failed
        if err is None:
            delivered += 1
        else:
            failed += 1
            print(f"delivery failed: {err}", file=sys.stderr)

    interval = (1.0 / args.rate) if args.rate > 0 else 0.0
    for i in range(args.count):
        payload = _build_request(args, i)
        producer.produce(
            args.topic,
            json.dumps(payload).encode(),
            on_delivery=_on_delivery,
        )
        producer.poll(0)
        if args.verbose or args.count <= 5:
            print(f"-> {args.topic}: {json.dumps(payload)}")
        if interval and i + 1 < args.count:
            time.sleep(interval)

    producer.flush(timeout=10)
    print(f"produced={delivered} failed={failed} topic={args.topic}")
    if failed:
        sys.exit(1)


def cmd_tail(args: argparse.Namespace) -> None:
    if args.topic:
        topic = args.topic
    elif args.which == "requests":
        topic = REQUEST_TOPIC
    else:
        topic = ACK_TOPIC

    # Use a unique group id so tailing never disturbs the C++ consumer's
    # offsets (group.id=migration-workers) and so re-runs see all messages.
    group = f"migration-cli-tail-{os.getpid()}-{int(time.time())}"

    consumer = Consumer(
        {
            "bootstrap.servers": args.brokers,
            "group.id": group,
            "auto.offset.reset": "earliest" if args.from_beginning else "latest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    seen = 0
    start = "beginning" if args.from_beginning else "latest"
    print(f"tailing {topic} from {start} (Ctrl-C to stop)")
    try:
        while True:
            msg = consumer.poll(0.5)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"!!! {msg.error()}", file=sys.stderr)
                continue
            value = msg.value().decode("utf-8", errors="replace")
            print(
                f"[{topic} p{msg.partition()} o{msg.offset()}] {value}"
            )
            seen += 1
            if args.max and seen >= args.max:
                break
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        print(f"\nstopped after {seen} message(s)")


def cmd_reset_offsets(args: argparse.Namespace) -> None:
    # AdminClient.alter_consumer_group_offsets is the proper external reset
    # path -- it does not require being an active member of the group, unlike
    # Consumer.commit() which fails with UNKNOWN_MEMBER_ID for non-members.
    admin = AdminClient({"bootstrap.servers": args.brokers})

    # Look up watermarks via a throwaway consumer (admin has no API for this).
    probe = Consumer(
        {
            "bootstrap.servers": args.brokers,
            "group.id": f"migration-cli-reset-probe-{os.getpid()}",
            "enable.auto.commit": False,
        }
    )
    md = probe.list_topics(topic=args.topic, timeout=5)
    if args.topic not in md.topics or md.topics[args.topic].error:
        probe.close()
        raise SystemExit(f"topic not found: {args.topic}")

    target_offsets: list[TopicPartition] = []
    for partition_id in md.topics[args.topic].partitions:
        tp = TopicPartition(args.topic, partition_id)
        low, high = probe.get_watermark_offsets(tp, timeout=5)
        offset = low if args.to == "earliest" else high
        target_offsets.append(TopicPartition(args.topic, partition_id, offset))
    probe.close()

    futures = admin.alter_consumer_group_offsets(
        [ConsumerGroupTopicPartitions(args.group, target_offsets)]
    )
    for group_id, fut in futures.items():
        fut.result()
        print(f"reset group='{group_id}' topic='{args.topic}' to {args.to}:")
    for tp in target_offsets:
        print(f"  partition {tp.partition} -> offset {tp.offset}")


def cmd_status(args: argparse.Namespace) -> None:
    admin = AdminClient({"bootstrap.servers": args.brokers})
    md = admin.list_topics(timeout=5)
    brokers = sorted(md.brokers.values(), key=lambda b: b.id)
    print(f"brokers ({len(brokers)}):")
    for b in brokers:
        print(f"  id={b.id} host={b.host}:{b.port}")

    print("topics:")
    for tid in sorted(md.topics):
        if tid.startswith("__"):
            continue
        topic = md.topics[tid]
        consumer = Consumer(
            {
                "bootstrap.servers": args.brokers,
                "group.id": f"migration-cli-status-{os.getpid()}",
                "enable.auto.commit": False,
            }
        )
        try:
            offsets: list[str] = []
            for partition_id in topic.partitions:
                low, high = consumer.get_watermark_offsets(
                    TopicPartition(tid, partition_id), timeout=2
                )
                offsets.append(f"p{partition_id}:[{low},{high})")
            print(f"  {tid}  partitions={len(topic.partitions)}  {' '.join(offsets)}")
        finally:
            consumer.close()


# ---------- argparse wiring --------------------------------------------------


def _add_produce_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--topic", default=REQUEST_TOPIC, help="target topic")
    p.add_argument("--count", type=int, default=1, help="number of messages")
    p.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="messages per second (0 = as fast as possible)",
    )
    p.add_argument(
        "--random",
        action="store_true",
        help="generate random field values per message",
    )
    p.add_argument(
        "--migration-id",
        type=int,
        default=None,
        help="explicit migration_id (default: nanoseconds-since-epoch + index)",
    )
    p.add_argument("--src-slot", type=int, default=0)
    p.add_argument("--dst-slot", type=int, default=1)
    p.add_argument("--layer-id", type=int, default=0)
    p.add_argument("--position-start", type=int, default=0)
    p.add_argument("--position-end", type=int, default=16)
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kafka CLI for the cpp_server migration flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--brokers",
        default=DEFAULT_BROKERS,
        help=f"bootstrap.servers (default: {DEFAULT_BROKERS}; env KAFKA_BROKERS)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup", help="Create the app topics")
    p_setup.add_argument("--partitions", type=int, default=1)
    p_setup.add_argument("--replication-factor", type=int, default=1)
    p_setup.set_defaults(func=cmd_setup)

    p_prod = sub.add_parser("produce", help="Produce migration request messages")
    _add_produce_args(p_prod)
    p_prod.set_defaults(func=cmd_produce)

    p_tail = sub.add_parser("tail", help="Tail a topic")
    p_tail.add_argument(
        "which",
        choices=["acks", "requests"],
        nargs="?",
        default="acks",
        help="which topic to tail (default: acks)",
    )
    p_tail.add_argument("--topic", help="override topic name")
    p_tail.add_argument("--from-beginning", action="store_true")
    p_tail.add_argument(
        "--max",
        type=int,
        default=0,
        help="stop after N messages (0 = unbounded)",
    )
    p_tail.set_defaults(func=cmd_tail)

    p_reset = sub.add_parser(
        "reset-offsets",
        help="Reset a consumer group's committed offsets",
    )
    p_reset.add_argument("--group", default=DEFAULT_GROUP)
    p_reset.add_argument("--topic", default=REQUEST_TOPIC)
    p_reset.add_argument(
        "--to",
        choices=["earliest", "latest"],
        default="earliest",
    )
    p_reset.set_defaults(func=cmd_reset_offsets)

    p_status = sub.add_parser("status", help="Show broker/topic snapshot")
    p_status.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
