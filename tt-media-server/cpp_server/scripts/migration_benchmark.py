#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Benchmark KV migration throughput across multiple layers.

Sends N migration requests (one per layer) and measures the wall-clock time
between the first request being produced and the final SUCCESSFUL ACK being
received on the kv-migration-acks topic.

Uses confluent-kafka (same client library as migration_cli.py / the C++ code)
so timing behavior matches the production messaging layer.

Example:
    python migration_benchmark.py \\
        --brokers $(hostname):9092 \\
        --src-slot 0 --dst-slot 1 \\
        --layer-begin 26 --layer-end 30 \\
        --positions 60000

Which is equivalent to running four `migration_cli.py produce` commands for
layers 26..29 (layer-end 30 exclusive), each migrating 60k positions.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

from confluent_kafka import Consumer, KafkaError, Producer, TopicPartition


DEFAULT_BROKERS = os.environ.get("KAFKA_BROKERS", "10.32.89.65:9092")
REQUEST_TOPIC = "kv-migration-requests"
ACK_TOPIC = "kv-migration-acks"


def _build_request(
    migration_id: int,
    src_slot: int,
    dst_slot: int,
    layer: int,
    src_pos_begin: int,
    src_pos_end: int,
    dst_pos_begin: int,
    dst_pos_end: int,
) -> dict[str, Any]:
    return {
        "migration_id": migration_id,
        "src_slot": src_slot,
        "dst_slot": dst_slot,
        "layer_begin": layer,
        "layer_end": layer + 1,
        "src_position_begin": src_pos_begin,
        "src_position_end": src_pos_end,
        "dst_position_begin": dst_pos_begin,
        "dst_position_end": dst_pos_end,
    }


def _run_single_iteration(
    args: argparse.Namespace,
    layers: list[int],
    src_pos_end: int,
    dst_pos_end: int,
    base_id: int,
) -> dict[str, Any]:
    """Run one migration round-trip and return timing data.

    Returns a dict with keys:
        total_wall : first-produce -> last-SUCCESSFUL-ACK wall time (s)
        produce_s  : time to produce+flush all requests (s)
        ack_s      : time from produce-done to last SUCCESSFUL ACK (s)
        latencies  : per-request produce->ACK latencies (s), one per layer
        failed     : list of migration_ids that FAILED
        timed_out  : bool -- True if not all ACKs arrived within --timeout
    """
    pending: dict[int, dict[str, Any]] = {}
    for i, layer in enumerate(layers):
        migration_id = base_id + i
        pending[migration_id] = {
            "layer": layer,
            "produce_ts": None,
            "ack_ts": None,
        }

    # Subscribe to acks BEFORE producing, with a fresh group starting from
    # latest so we only see ACKs for this benchmark run.
    group = f"migration-benchmark-{os.getpid()}-{time.time_ns()}"
    consumer = Consumer(
        {
            "bootstrap.servers": args.brokers,
            "group.id": group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        }
    )

    # Manually assign each ACK partition seeked to the current high watermark
    # so we don't miss the first ACK due to subscribe/rebalance latency.
    md = consumer.list_topics(topic=ACK_TOPIC, timeout=5)
    if ACK_TOPIC not in md.topics or md.topics[ACK_TOPIC].error:
        consumer.close()
        raise SystemExit(f"topic not found: {ACK_TOPIC}")
    partitions = list(md.topics[ACK_TOPIC].partitions)
    assignment: list[TopicPartition] = []
    for pid in partitions:
        _low, high = consumer.get_watermark_offsets(
            TopicPartition(ACK_TOPIC, pid), timeout=5
        )
        assignment.append(TopicPartition(ACK_TOPIC, pid, high))
    consumer.assign(assignment)

    producer = Producer(
        {
            "bootstrap.servers": args.brokers,
            "linger.ms": 0,
            "enable.idempotence": False,
        }
    )

    def _on_delivery(err: Any, msg: Any) -> None:
        if err is not None:
            print(f"delivery failed: {err}", file=sys.stderr)

    overall_start = time.monotonic()
    for migration_id, info in pending.items():
        payload = _build_request(
            migration_id=migration_id,
            src_slot=args.src_slot,
            dst_slot=args.dst_slot,
            layer=info["layer"],
            src_pos_begin=args.src_pos_begin,
            src_pos_end=src_pos_end,
            dst_pos_begin=args.dst_pos_begin,
            dst_pos_end=dst_pos_end,
        )
        info["produce_ts"] = time.monotonic()
        producer.produce(
            REQUEST_TOPIC,
            json.dumps(payload).encode(),
            on_delivery=_on_delivery,
        )
        producer.poll(0)
        if args.verbose:
            print(f"-> {REQUEST_TOPIC}: {json.dumps(payload)}")

    producer.flush(timeout=10)
    produced_ts = time.monotonic()

    outstanding = set(pending.keys())
    deadline = time.monotonic() + args.timeout
    failed_ids: list[int] = []
    while outstanding and time.monotonic() < deadline:
        msg = consumer.poll(0.5)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                print(f"!!! {msg.error()}", file=sys.stderr)
            continue
        try:
            data = json.loads(msg.value().decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        mid = data.get("migration_id")
        status = data.get("status")
        if mid not in pending:
            continue
        if status == "SUCCESSFUL":
            pending[mid]["ack_ts"] = time.monotonic()
            outstanding.discard(mid)
            if args.verbose:
                print(f"<- ACK SUCCESSFUL migration_id={mid} layer={pending[mid]['layer']}")
        elif status == "FAILED":
            pending[mid]["ack_ts"] = time.monotonic()
            outstanding.discard(mid)
            failed_ids.append(mid)
            print(
                f"<- ACK FAILED migration_id={mid} layer={pending[mid]['layer']}",
                file=sys.stderr,
            )
        # IN_PROGRESS: ignore

    consumer.close()

    if outstanding:
        return {
            "total_wall": float("nan"),
            "produce_s": produced_ts - overall_start,
            "ack_s": float("nan"),
            "latencies": [],
            "failed": failed_ids,
            "timed_out": True,
            "outstanding": sorted(outstanding),
        }

    ack_times = [info["ack_ts"] for info in pending.values()]
    latencies = [info["ack_ts"] - info["produce_ts"] for info in pending.values()]
    total_wall = max(ack_times) - overall_start
    return {
        "total_wall": total_wall,
        "produce_s": produced_ts - overall_start,
        "ack_s": max(ack_times) - produced_ts,
        "latencies": latencies,
        "failed": failed_ids,
        "timed_out": False,
        "outstanding": [],
    }


def run_benchmark(args: argparse.Namespace) -> int:
    layers = list(range(args.layer_begin, args.layer_end))
    if not layers:
        print("layer range is empty", file=sys.stderr)
        return 2

    src_pos_end = args.src_pos_end if args.src_pos_end is not None else args.positions
    dst_pos_end = args.dst_pos_end if args.dst_pos_end is not None else args.positions

    if args.iterations < 1:
        print("--iterations must be >= 1", file=sys.stderr)
        return 2

    print(
        f"running {args.iterations} iteration(s) of {len(layers)} requests each: "
        f"slots {args.src_slot}->{args.dst_slot} "
        f"layers [{args.layer_begin},{args.layer_end}) "
        f"src_pos [{args.src_pos_begin},{src_pos_end}) "
        f"dst_pos [{args.dst_pos_begin},{dst_pos_end})"
    )
    if args.warmup > 0:
        print(f"warmup iterations (excluded from stats): {args.warmup}")

    base_id_seed = args.migration_id if args.migration_id is not None else time.time_ns()
    # Reserve a distinct migration_id range per iteration.
    id_stride = max(len(layers), 1)

    total_walls: list[float] = []
    all_latencies: list[float] = []
    produce_times: list[float] = []
    ack_times_list: list[float] = []
    iteration_reports: list[dict[str, Any]] = []
    any_failed = False

    total_runs = args.warmup + args.iterations
    for i in range(total_runs):
        is_warmup = i < args.warmup
        base_id = base_id_seed + i * id_stride
        label = f"warmup {i + 1}/{args.warmup}" if is_warmup else \
                f"iter {i - args.warmup + 1}/{args.iterations}"
        result = _run_single_iteration(
            args, layers, src_pos_end, dst_pos_end, base_id
        )
        if result["timed_out"]:
            print(
                f"[{label}] timed out after {args.timeout}s waiting for ACK(s): "
                f"{result['outstanding']}",
                file=sys.stderr,
            )
            return 1
        if result["failed"]:
            any_failed = True

        print(
            f"[{label}] total_wall={result['total_wall']:.4f}s  "
            f"produce={result['produce_s']:.4f}s  ack={result['ack_s']:.4f}s"
        )

        if is_warmup:
            continue
        total_walls.append(result["total_wall"])
        all_latencies.extend(result["latencies"])
        produce_times.append(result["produce_s"])
        ack_times_list.append(result["ack_s"])
        iteration_reports.append(result)

        if args.settle > 0 and i + 1 < total_runs:
            time.sleep(args.settle)

    # Aggregate report.
    mean_wall = statistics.fmean(total_walls)
    median_wall = statistics.median(total_walls)
    print()
    print("=== migration benchmark results ===")
    print(f"layers migrated : {len(layers)} (range [{args.layer_begin},{args.layer_end}))")
    print(f"positions/layer : {src_pos_end - args.src_pos_begin}")
    print(f"iterations      : {args.iterations} (warmup: {args.warmup})")
    print("total wall time per iteration (first produce -> last SUCCESSFUL ACK):")
    print(f"  min    = {min(total_walls):.4f}s")
    print(f"  max    = {max(total_walls):.4f}s")
    print(f"  mean   = {mean_wall:.4f}s")
    print(f"  median = {median_wall:.4f}s")
    if len(total_walls) > 1:
        stdev_wall = statistics.stdev(total_walls)
        print(f"  stdev  = {stdev_wall:.4f}s"
              f"  ({(stdev_wall / mean_wall * 100.0):.1f}% of mean)")
    print(f"produce phase mean : {statistics.fmean(produce_times):.4f}s")
    print(f"ack phase mean     : {statistics.fmean(ack_times_list):.4f}s")
    print(f"per-request latency across all iterations (n={len(all_latencies)}):")
    print(f"  min  = {min(all_latencies):.4f}s")
    print(f"  max  = {max(all_latencies):.4f}s")
    print(f"  mean = {statistics.fmean(all_latencies):.4f}s")
    if len(all_latencies) > 1:
        print(f"  stdev= {statistics.stdev(all_latencies):.4f}s")

    # Extrapolate to a full model (default 61 layers) assuming migration time
    # scales linearly with the number of layers at the same slot pair and
    # positions/layer. Use both mean and median total wall time to smooth out
    # per-iteration variance.
    target_layers = args.extrapolate_layers
    if target_layers > 0 and len(layers) > 0:
        scale = target_layers / len(layers)
        linear_from_mean = mean_wall * scale
        linear_from_median = median_wall * scale
        print()
        print(f"extrapolation to {target_layers} layers (assumes same slot pair, "
              f"same positions/layer, and")
        print("  that migration time scales linearly with the number of layers):")
        print(f"  from mean total wall   : {linear_from_mean:.4f}s"
              f"  (= {mean_wall:.4f}s * {target_layers}/{len(layers)})")
        print(f"  from median total wall : {linear_from_median:.4f}s"
              f"  (= {median_wall:.4f}s * {target_layers}/{len(layers)})")
        if len(total_walls) > 1:
            lo = min(total_walls) * scale
            hi = max(total_walls) * scale
            print(f"  observed range         : [{lo:.4f}s, {hi:.4f}s]")

    return 1 if any_failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--brokers",
        default=DEFAULT_BROKERS,
        help=f"bootstrap.servers (default: {DEFAULT_BROKERS}; env KAFKA_BROKERS)",
    )
    parser.add_argument("--src-slot", type=int, default=0)
    parser.add_argument("--dst-slot", type=int, default=1)
    parser.add_argument(
        "--layer-begin",
        type=int,
        default=26,
        help="first layer to migrate (inclusive)",
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=30,
        help="last layer to migrate (exclusive)",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=60000,
        help="number of positions per layer (used for both src and dst when "
        "--src-pos-end/--dst-pos-end are not given)",
    )
    parser.add_argument("--src-pos-begin", type=int, default=0)
    parser.add_argument(
        "--src-pos-end",
        type=int,
        default=None,
        help="src token pos end, exclusive (default: --positions)",
    )
    parser.add_argument("--dst-pos-begin", type=int, default=0)
    parser.add_argument(
        "--dst-pos-end",
        type=int,
        default=None,
        help="dst token pos end, exclusive (default: --positions)",
    )
    parser.add_argument(
        "--migration-id",
        type=int,
        default=None,
        help="explicit base migration_id (default: nanoseconds-since-epoch)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="seconds to wait for all ACKs before giving up",
    )
    parser.add_argument(
        "--extrapolate-layers",
        type=int,
        default=61,
        help="extrapolate the measured time to a hypothetical model of this many "
        "layers (default: 61; set to 0 to disable)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="number of measured iterations to run and average over. 10 is a "
        "good default: it flattens the ~30%% single-run variance we observed "
        "while keeping total runtime bounded (~10 * total_wall).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="number of warmup iterations run before measurement (excluded "
        "from stats) to prime caches / JIT-like effects on the worker side",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.5,
        help="seconds to sleep between iterations so the worker can quiesce "
        "before the next batch",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(run_benchmark(args))


if __name__ == "__main__":
    main()
