#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
test_cold_path.py — TRUE Cold Path (Remote Fetch)

Tests fetching data from a REMOTE client:

  Client A (publisher)              Client B (consumer)
       │                                  │
       │ put("key", data)                 │
       │ ───────────────►                 │
       │                    Mooncake      │
       │                    Master        │
       │                                  │
       │                                  │ get("key")
       │                                  │ ◄─── data comes from A's memory
       │                                  │      (via Mooncake transfer)

This is the TRUE cold path:
- Data is NOT on Client B's DRAM
- Data is NOT on Client B's local SSD
- Data must be fetched from Client A over the NETWORK

In our single-host test, both clients run on localhost but are separate
processes with separate DRAM allocations — simulating the remote fetch.
"""

import sys
import time
import multiprocessing
from client import MooncakeClient, create_test_data


def publisher_process(ready_event, done_event, results_queue):
    """
    Publisher: puts data and waits for consumer to finish.
    """
    client = MooncakeClient("publisher", dram_size_mb=32)

    if not client.connect():
        results_queue.put(
            {"role": "publisher", "success": False, "error": "connect failed"}
        )
        ready_event.set()
        return

    try:
        # Put test data
        keys = []
        data_map = {}
        for i in range(10):
            key = f"cold_remote_{i:04d}"
            data = create_test_data(size_kb=64)
            result = client.put(key, data)
            if result.success:
                keys.append(key)
                data_map[key] = len(data)
            else:
                print(f"[publisher] put failed: {result.error}")

        print(f"[publisher] Put {len(keys)} items, signaling ready...")

        results_queue.put(
            {
                "role": "publisher",
                "success": True,
                "keys": keys,
                "data_sizes": data_map,
            }
        )

        # Signal consumer we're ready
        ready_event.set()

        # Wait for consumer to finish (keep connection alive)
        done_event.wait(timeout=30.0)
        print("[publisher] Consumer done, exiting")

    finally:
        client.close()


def consumer_process(ready_event, done_event, results_queue):
    """
    Consumer: waits for publisher, then fetches data.

    Key point: Consumer's DRAM is EMPTY — data must come from publisher
    (or from DFS if publisher's data was persisted).
    """
    # Wait for publisher to be ready
    print("[consumer] Waiting for publisher...")
    ready_event.wait(timeout=30.0)

    # Small delay to ensure data is propagated
    time.sleep(0.5)

    client = MooncakeClient("consumer", dram_size_mb=32)

    if not client.connect():
        results_queue.put(
            {"role": "consumer", "success": False, "error": "connect failed"}
        )
        done_event.set()
        return

    try:
        # Use fixed keys (matching publisher)
        keys = [f"cold_remote_{i:04d}" for i in range(10)]
        print(f"[consumer] Attempting to read {len(keys)} items from remote...")

        # Fetch data — this should come from publisher's memory (cold path)
        success_count = 0
        latencies = []

        for key in keys:
            retrieved, result = client.get(key)

            if result.success:
                success_count += 1
                latencies.append(result.latency_ms)
                print(f"  [consumer] get '{key}': {result.latency_ms:.2f}ms")
            else:
                print(f"  [consumer] get '{key}' FAILED: {result.error}")

        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        results_queue.put(
            {
                "role": "consumer",
                "success": success_count == len(keys),
                "read_count": success_count,
                "total_keys": len(keys),
                "avg_latency_ms": avg_latency,
                "latencies": latencies,
            }
        )

        print(
            f"[consumer] Read {success_count}/{len(keys)} items, avg latency: {avg_latency:.2f}ms"
        )

    finally:
        client.close()
        done_event.set()


def test_cold_path():
    print("=" * 60)
    print("COLD PATH TEST (Remote Fetch)")
    print("=" * 60)
    print()
    print("This tests the TRUE cold path:")
    print("  - Client A (publisher) puts data")
    print("  - Client B (consumer) fetches it")
    print("  - Data comes from A's memory over the network")
    print("  - B's DRAM is empty — no local hit possible")
    print()

    # Synchronization primitives
    ready_event = multiprocessing.Event()
    done_event = multiprocessing.Event()
    results_queue = multiprocessing.Queue()

    # Start publisher
    publisher = multiprocessing.Process(
        target=publisher_process, args=(ready_event, done_event, results_queue)
    )
    publisher.start()

    # Start consumer
    consumer = multiprocessing.Process(
        target=consumer_process, args=(ready_event, done_event, results_queue)
    )
    consumer.start()

    # Wait for both to finish
    publisher.join(timeout=60.0)
    consumer.join(timeout=60.0)

    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)

    pub_ok = False
    con_ok = False

    for r in results:
        role = r.get("role", "unknown")
        success = r.get("success", False)
        print(f"  {role}: success={success}")

        if role == "publisher":
            pub_ok = success
        elif role == "consumer":
            con_ok = success
            if success:
                print(f"    read: {r.get('read_count')}/{r.get('total_keys')}")
                print(f"    avg latency: {r.get('avg_latency_ms', 0):.2f}ms")

    print()
    if pub_ok and con_ok:
        print("=" * 60)
        print("[PASS] Cold path test completed")
        print("  Data was fetched from remote client successfully")
        print("=" * 60)
        return True
    else:
        print("=" * 60)
        print("[FAIL] Cold path test failed")
        if not pub_ok:
            print("  Publisher failed")
        if not con_ok:
            print("  Consumer failed to fetch remote data")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = test_cold_path()
    sys.exit(0 if success else 1)
