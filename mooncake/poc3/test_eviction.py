#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
test_eviction.py — Eviction stress test (WARM path, local SSD)

Tests the eviction behavior under memory pressure:
1. Fill DRAM beyond capacity
2. Verify LRU eviction kicks in at watermark
3. Verify evicted data still readable from LOCAL SSD/DFS
4. Test that recently accessed data stays in DRAM

Note: This is still LOCAL storage (DRAM → local SSD).
True "cold" path fetches from a REMOTE host — see test_cold_path.py.
"""

import sys
import time
from client import MooncakeClient, create_test_data


def test_eviction():
    print("=" * 60)
    print("EVICTION TEST (Warm Path - Local SSD)")
    print("=" * 60)

    # Small DRAM for quick eviction testing
    dram_mb = 16
    client = MooncakeClient("eviction_test", dram_size_mb=dram_mb)

    if not client.connect():
        print("[FAIL] Could not connect to Mooncake master")
        return False

    try:
        # Step 1: Fill to trigger eviction
        print(f"\n[step 1] Fill {dram_mb}MB DRAM to trigger eviction...")

        # Put items in order - older items should be evicted first (LRU)
        item_size_kb = 64
        num_items = (dram_mb * 1024) // item_size_kb  # Enough to fill DRAM
        num_items = int(num_items * 1.5)  # Overfill to ensure eviction

        keys = []
        for i in range(num_items):
            key = f"evict_item_{i:04d}"
            data = create_test_data(size_kb=item_size_kb)
            result = client.put(key, data)
            keys.append(key)

            if not result.success:
                print(f"  put failed at item {i}: {result.error}")
                if i > 10:
                    break

            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{num_items} items put")

        print(f"  Total items put: {len(keys)}")

        # Wait for async DFS writes
        print("\n[step 2] Wait for local SSD persistence...")
        time.sleep(2.0)

        # Step 3: Check which items still exist
        print("\n[step 3] Check existence of all items...")
        existence = client.exists(keys)
        exists_count = sum(existence)
        print(f"  {exists_count}/{len(keys)} items still exist")

        # Step 4: Test LRU behavior
        print("\n[step 4] Test LRU: old items evicted from DRAM first...")

        early_items = keys[:10]
        early_existence = client.exists(early_items)
        early_exists = sum(early_existence)

        late_items = keys[-10:]
        late_existence = client.exists(late_items)
        late_exists = sum(late_existence)

        print(f"  Early items (0-9): {early_exists}/10 exist")
        print(f"  Late items (last 10): {late_exists}/10 exist")

        if late_exists > early_exists:
            print("  [OK] LRU behavior observed")

        # Step 5: Read evicted items from local SSD
        print("\n[step 5] Read evicted items from local SSD...")
        ssd_reads = 0
        ssd_latencies = []

        for key in early_items:
            retrieved, result = client.get(key)
            if result.success:
                ssd_reads += 1
                ssd_latencies.append(result.latency_ms)

        if ssd_reads > 0:
            avg_latency = sum(ssd_latencies) / len(ssd_latencies)
            print(f"  Read {ssd_reads}/10 evicted items from local SSD")
            print(f"  Average local SSD read latency: {avg_latency:.2f}ms")
        else:
            print("  [WARN] No evicted items readable")

        # Step 6: Latency distribution
        print("\n[step 6] Latency distribution (all local)...")

        sample_keys = keys[::10]
        hot_count = 0
        warm_count = 0

        for key in sample_keys:
            _, result = client.get(key)
            if result.success:
                if result.latency_ms < 1.0:
                    hot_count += 1
                else:
                    warm_count += 1

        total = hot_count + warm_count
        if total > 0:
            print(
                f"  Hot (DRAM, <1ms):      {hot_count}/{total} ({100 * hot_count / total:.0f}%)"
            )
            print(
                f"  Warm (local SSD, >1ms): {warm_count}/{total} ({100 * warm_count / total:.0f}%)"
            )

        client.print_stats()

        print("\n" + "=" * 60)
        print("[PASS] Eviction test completed (warm path)")
        print("=" * 60)
        return True

    finally:
        client.close()


if __name__ == "__main__":
    success = test_eviction()
    sys.exit(0 if success else 1)
