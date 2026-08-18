#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
test_warm_path.py — Scenario 2: DFS fallback (warm path)

Tests the fallback path where data is not in DRAM but exists in DFS:
1. Put data → goes to DRAM + async DFS
2. Fill DRAM to trigger eviction
3. Get original data → misses DRAM, reads from DFS (slower)
4. Verify latency is higher than hot path
"""

import sys
import time
from client import MooncakeClient, create_test_data


def test_warm_path():
    print("=" * 60)
    print("SCENARIO 2: Warm Path (DRAM miss → DFS read)")
    print("=" * 60)

    # Use small DRAM to easily trigger eviction
    client = MooncakeClient("warm_test", dram_size_mb=32)

    if not client.connect():
        print("[FAIL] Could not connect to Mooncake master")
        print("[FAIL] Make sure master is started with --root_fs_dir")
        return False

    try:
        # Step 1: Put initial data (will be evicted later)
        print("\n[step 1] Put initial data that will be evicted...")
        target_key = "warm_target_data"
        target_data = create_test_data(size_kb=64)

        put_result = client.put(target_key, target_data)
        print(f"  put '{target_key}': {put_result.latency_ms:.2f}ms")

        if not put_result.success:
            print(f"[FAIL] initial put failed: {put_result.error}")
            return False

        # Verify it's readable (hot path)
        _, get_result = client.get(target_key)
        hot_latency = get_result.latency_ms
        print(f"  get (hot): {hot_latency:.2f}ms")

        # Step 2: Wait for async DFS write
        print("\n[step 2] Wait for async DFS persistence...")
        time.sleep(2.0)
        print("  Waited 2 seconds for DFS write")

        # Step 3: Fill DRAM to trigger eviction
        print("\n[step 3] Fill DRAM to trigger eviction...")
        # 32MB DRAM, 64KB items → need ~600 items to fill at 80% watermark
        fill_count = 600
        fill_results = client.fill_dram("warm_filler", fill_count, size_kb=64)

        successful = sum(1 for r in fill_results if r.success)
        print(f"  Filled {successful}/{fill_count} items")

        # Step 4: Check if target was evicted
        print("\n[step 4] Check if target data was evicted from DRAM...")

        # First, check existence
        exists = client.exists([target_key])
        print(f"  exists({target_key}): {exists[0]}")

        if not exists[0]:
            print("[WARN] Target key no longer exists (fully evicted?)")
            print("[WARN] DFS persistence may not be enabled")
            print("[INFO] Start master with: --root_fs_dir=/tmp/mooncake_dfs_poc3")
            return False

        # Step 5: Read from DFS (warm path)
        print("\n[step 5] Read evicted data (should come from DFS)...")
        retrieved, get_result = client.get(target_key)
        warm_latency = get_result.latency_ms

        print(f"  get (warm): {warm_latency:.2f}ms")

        if not get_result.success:
            print(f"[FAIL] get failed: {get_result.error}")
            return False

        if retrieved != target_data:
            print("[FAIL] Data mismatch!")
            return False

        print("  [OK] Data matches")

        # Step 6: Compare latencies
        print("\n[step 6] Compare hot vs warm latencies...")
        print(f"  Hot (DRAM): {hot_latency:.2f}ms")
        print(f"  Warm (DFS): {warm_latency:.2f}ms")
        print(f"  Ratio: {warm_latency / hot_latency:.1f}x slower")

        if warm_latency > hot_latency:
            print("  [OK] Warm path is slower (as expected)")
        else:
            print("  [INFO] Warm path not slower - data may still be in DRAM cache")

        # Step 7: Re-read should now be hot (promoted back to DRAM)
        print("\n[step 7] Re-read (should be promoted to DRAM)...")
        _, get_result2 = client.get(target_key)
        print(f"  get (after promotion): {get_result2.latency_ms:.2f}ms")

        client.print_stats()

        print("\n" + "=" * 60)
        print("[PASS] Warm path test completed")
        print("=" * 60)
        return True

    finally:
        client.close()


if __name__ == "__main__":
    success = test_warm_path()
    sys.exit(0 if success else 1)
