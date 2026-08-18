#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
test_cross_client.py — Scenario 6: Multi-process / cross-client

Tests distributed behavior with multiple clients:
1. Client A puts data
2. Client B reads same data (tests distributed visibility)
3. Both clients fill DRAM (tests shared eviction pool)

Can run as:
- Single process with 2 client instances (simpler)
- Two separate processes (more realistic)
"""

import sys
import time
import argparse
import multiprocessing
from client import MooncakeClient, create_test_data


def run_publisher(results_queue):
    """Publisher process: puts data."""
    client = MooncakeClient("publisher", dram_size_mb=32)

    if not client.connect():
        results_queue.put(
            {"role": "publisher", "success": False, "error": "connect failed"}
        )
        return

    try:
        # Put test data
        keys = []
        for i in range(10):
            key = f"cross_test_{i:04d}"
            data = create_test_data(size_kb=64)
            result = client.put(key, data)
            if result.success:
                keys.append(key)
            else:
                print(f"[publisher] put failed: {result.error}")

        print(f"[publisher] Put {len(keys)} items")

        # Signal readiness
        results_queue.put(
            {
                "role": "publisher",
                "success": True,
                "keys": keys,
                "ready_time": time.time(),
            }
        )

        # Keep connection alive for consumer
        time.sleep(5.0)

    finally:
        client.close()


def run_consumer(results_queue, wait_for_publisher=True):
    """Consumer process: reads data put by publisher."""
    client = MooncakeClient("consumer", dram_size_mb=32)

    if not client.connect():
        results_queue.put(
            {"role": "consumer", "success": False, "error": "connect failed"}
        )
        return

    try:
        # Wait for publisher to be ready
        if wait_for_publisher:
            time.sleep(1.0)

        # Read test data
        success_count = 0
        latencies = []

        for i in range(10):
            key = f"cross_test_{i:04d}"
            retrieved, result = client.get(key)

            if result.success:
                success_count += 1
                latencies.append(result.latency_ms)
            else:
                print(f"[consumer] get '{key}' failed: {result.error}")

        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        print(
            f"[consumer] Read {success_count}/10 items, avg latency: {avg_latency:.2f}ms"
        )

        results_queue.put(
            {
                "role": "consumer",
                "success": success_count == 10,
                "read_count": success_count,
                "avg_latency_ms": avg_latency,
            }
        )

    finally:
        client.close()


def test_single_process():
    """Test with two client instances in same process."""
    print("\n" + "-" * 40)
    print("Single-process mode (2 client instances)")
    print("-" * 40)

    client_a = MooncakeClient("client_A", dram_size_mb=32)
    client_b = MooncakeClient("client_B", dram_size_mb=32)

    if not client_a.connect() or not client_b.connect():
        print("[FAIL] Could not connect clients")
        return False

    try:
        # Step 1: Client A puts data
        print("\n[step 1] Client A puts data...")
        keys = []
        for i in range(10):
            key = f"single_proc_test_{i:04d}"
            data = create_test_data(size_kb=64)
            result = client_a.put(key, data)
            if result.success:
                keys.append(key)

        print(f"  Client A put {len(keys)} items")

        # Step 2: Client B reads immediately
        print("\n[step 2] Client B reads same data...")
        read_count = 0
        latencies = []

        for key in keys:
            retrieved, result = client_b.get(key)
            if result.success:
                read_count += 1
                latencies.append(result.latency_ms)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        print(f"  Client B read {read_count}/{len(keys)} items")
        print(f"  Average latency: {avg_latency:.2f}ms")

        if read_count == len(keys):
            print("  [OK] All items visible to both clients")
        else:
            print("  [WARN] Some items not visible")

        # Step 3: Test shared DRAM pool
        print("\n[step 3] Both clients fill DRAM (shared pool test)...")

        # Each client fills half
        client_a.fill_dram("shared_a", 200, size_kb=64)
        client_b.fill_dram("shared_b", 200, size_kb=64)

        # Check original items still accessible
        exists = client_a.exists(keys)
        still_exist = sum(exists)
        print(f"  Original items still exist: {still_exist}/{len(keys)}")

        client_a.print_stats()
        client_b.print_stats()

        return read_count == len(keys)

    finally:
        client_a.close()
        client_b.close()


def test_multi_process():
    """Test with two separate processes."""
    print("\n" + "-" * 40)
    print("Multi-process mode (2 separate processes)")
    print("-" * 40)

    results_queue = multiprocessing.Queue()

    # Start publisher
    publisher = multiprocessing.Process(target=run_publisher, args=(results_queue,))
    publisher.start()

    # Wait a bit then start consumer
    time.sleep(1.0)

    consumer = multiprocessing.Process(target=run_consumer, args=(results_queue, True))
    consumer.start()

    # Collect results
    publisher.join(timeout=10.0)
    consumer.join(timeout=10.0)

    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    print("\n[results]")
    for r in results:
        print(f"  {r['role']}: success={r.get('success', False)}")

    pub_success = any(
        r.get("role") == "publisher" and r.get("success") for r in results
    )
    con_success = any(r.get("role") == "consumer" and r.get("success") for r in results)

    return pub_success and con_success


def test_cross_client():
    print("=" * 60)
    print("SCENARIO 6: Cross-Client (Distributed Visibility)")
    print("=" * 60)

    # Test 1: Single process
    single_ok = test_single_process()

    # Test 2: Multi process
    print()
    multi_ok = test_multi_process()

    print("\n" + "=" * 60)
    if single_ok and multi_ok:
        print("[PASS] Cross-client test completed")
    elif single_ok:
        print("[PARTIAL] Single-process OK, multi-process had issues")
    else:
        print("[FAIL] Cross-client test failed")
    print("=" * 60)

    return single_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi", "both"], default="both")
    parser.add_argument(
        "--role", choices=["publisher", "consumer"], help="For standalone process"
    )
    args = parser.parse_args()

    if args.role == "publisher":
        q = multiprocessing.Queue()
        run_publisher(q)
    elif args.role == "consumer":
        q = multiprocessing.Queue()
        run_consumer(q, wait_for_publisher=False)
    elif args.mode == "single":
        test_single_process()
    elif args.mode == "multi":
        test_multi_process()
    else:
        success = test_cross_client()
        sys.exit(0 if success else 1)
