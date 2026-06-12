#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
run_all.py — Run all poc3 scenarios

Requires Mooncake master running with DFS enabled:
    ./master_startup.sh
"""

import sys
import subprocess
import time

SCENARIOS = [
    ("test_hot_path.py", "Hot Path (DRAM reads)"),
    ("test_warm_path.py", "Warm Path (local SSD fallback)"),
    ("test_eviction.py", "Eviction Stress (LRU + local SSD)"),
    ("test_cold_path.py", "Cold Path (REMOTE fetch)"),
    ("test_realistic_kv.py", "Realistic KV Cache (real data format)"),
]


def run_scenario(script: str, name: str) -> bool:
    """Run a single scenario."""
    print(f"\n{'#' * 70}")
    print(f"# Running: {name}")
    print(f"# Script: {script}")
    print(f"{'#' * 70}\n")

    try:
        result = subprocess.run(
            [sys.executable, script],
            timeout=60,
            capture_output=False,  # Show output live
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name} timed out after 60s")
        return False
    except Exception as e:
        print(f"[ERROR] {name} failed: {e}")
        return False


def check_master():
    """Check if master is running and DFS is enabled."""
    print("Checking Mooncake master...")

    try:
        from client import MooncakeClient

        client = MooncakeClient("check", dram_size_mb=16)
        if not client.connect():
            return False, False

        print("[OK] Master is running")

        # Check DFS
        dfs_dir_exists = client.is_dfs_enabled()
        print(f"[INFO] DFS directory exists: {dfs_dir_exists}")

        if dfs_dir_exists:
            dfs_works = client.check_dfs_persistence()
            print(f"[INFO] DFS persistence works: {dfs_works}")
        else:
            dfs_works = False
            print("[WARN] DFS not enabled — warm/cold tests will fail")
            print("[WARN] Start master with: --root_fs_dir=/tmp/mooncake_dfs_poc3")

        client.close()
        return True, dfs_works

    except Exception as e:
        print(f"[ERROR] {e}")
        return False, False


def main():
    print("=" * 70)
    print("PoC3: Mooncake Multi-Tier Storage Test Suite")
    print("=" * 70)

    master_ok, dfs_ok = check_master()

    if not master_ok:
        print("\n[FAIL] Mooncake master not running!")
        print("\nStart master with:")
        print("    cd mooncake/poc3")
        print("    ./master_startup.sh")
        print("\nMake sure to enable DFS for warm/cold path tests.")
        sys.exit(1)

    if not dfs_ok:
        print("\n[WARN] DFS not enabled — skipping warm/eviction tests")
        print("[WARN] To enable: restart master with --root_fs_dir")
        # Filter out tests that need DFS (warm and eviction need local SSD)
        global SCENARIOS
        SCENARIOS = [
            (s, n)
            for s, n in SCENARIOS
            if "warm" not in s.lower() and "eviction" not in s.lower()
        ]
        print(f"[INFO] Running {len(SCENARIOS)} scenarios (local SSD tests skipped)")

    results = []

    for script, name in SCENARIOS:
        success = run_scenario(script, name)
        results.append((name, success))

        # Brief pause between scenarios
        time.sleep(1.0)

    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = 0
    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
        if success:
            passed += 1

    print()
    print(f"Total: {passed}/{len(results)} scenarios passed")
    print("=" * 70)

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
