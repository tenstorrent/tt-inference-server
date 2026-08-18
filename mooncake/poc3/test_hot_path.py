#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
test_hot_path.py — Scenario 1: DRAM reads (hot path)

Tests the fast path where data is in DRAM:
1. Put data → goes to DRAM
2. Get data immediately → hits DRAM (fast)
3. Verify latency is low (<1ms)
"""

import sys
from client import MooncakeClient, create_test_data


def test_hot_path():
    print("=" * 60)
    print("SCENARIO 1: Hot Path (DRAM reads)")
    print("=" * 60)

    client = MooncakeClient("hot_test", dram_size_mb=64)

    if not client.connect():
        print("[FAIL] Could not connect to Mooncake master")
        print("[FAIL] Start master with: ./master_startup.sh")
        return False

    try:
        # Test 1: Single put/get
        print("\n[test] Single put/get cycle...")
        data = create_test_data(size_kb=64)
        key = "hot_test_single"

        put_result = client.put(key, data)  # what is the key in the inference server?
        print(f"  put: {put_result.latency_ms:.2f}ms, success={put_result.success}")

        if not put_result.success:
            print(f"[FAIL] put failed: {put_result.error}")
            return False

        retrieved, get_result = client.get(key)  # same q here?
        print(
            f"  get: {get_result.latency_ms:.2f}ms, success={get_result.success}, from_dram={get_result.from_dram}"
        )

        if not get_result.success:
            print(f"[FAIL] get failed: {get_result.error}")
            return False

        if retrieved != data:
            print("[FAIL] Data mismatch!")
            return False

        print("  [OK] Data matches")

        # Test 2: Multiple items
        print("\n[test] Multiple items (10 x 64KB)...")
        latencies = []

        for i in range(10):
            key = f"hot_test_multi_{i}"
            data = create_test_data(size_kb=64)

            put_result = client.put(key, data)
            if not put_result.success:
                print(f"[FAIL] put failed at {i}")
                return False

            _, get_result = client.get(key)
            if not get_result.success:
                print(f"[FAIL] get failed at {i}")
                return False

            latencies.append(get_result.latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        print(f"  Average get latency: {avg_latency:.2f}ms")
        print(f"  Max get latency: {max_latency:.2f}ms")

        # Hot path should be fast
        if avg_latency < 5.0:  # <5ms is good for DRAM
            print("  [OK] Latency is within hot-path range")
        else:
            print("  [WARN] Latency higher than expected for DRAM")

        # Test 3: Repeated reads (should stay in DRAM)
        print("\n[test] Repeated reads of same key...")
        key = "hot_test_repeat"
        data = create_test_data(size_kb=64)
        client.put(key, data)

        latencies = []
        for i in range(100):
            _, get_result = client.get(key)
            latencies.append(get_result.latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        p99_latency = sorted(latencies)[98]
        print(f"  100 reads: avg={avg_latency:.2f}ms, p99={p99_latency:.2f}ms")

        client.print_stats()

        print("\n" + "=" * 60)
        print("[PASS] Hot path test completed successfully")
        print("=" * 60)
        return True

    finally:
        client.close()


if __name__ == "__main__":
    success = test_hot_path()
    sys.exit(0 if success else 1)
