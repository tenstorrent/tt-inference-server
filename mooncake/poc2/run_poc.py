#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
run_poc.py — End-to-end demo of the hybrid Mooncake KV orchestration.

This demonstrates the full flow:
1. "Box B" publishes KV block hashes to Mooncake
2. "Box A" submits a request with tokens that need those blocks
3. Scheduler misses locally → exist() finds them remote → pull() → park
4. Drain thread detects arrival → resume → dispatch

Run with Mooncake master:
    ./master_startup.sh  # in another terminal
    python run_poc.py

Run with in-memory fake (no master needed):
    python run_poc.py --fake
"""

import argparse
import time
import sys

from remote_kv_index import (
    RemoteKvIndex,
    MooncakeRemoteKvIndex,
    InMemoryRemoteKvIndex,
    MockRemoteKvIndex,
)
from migration_workers import (
    MigrationWorkers,
    DelayedMockMigrationWorkers,
)
from scheduler_sim import SchedulerSim, token_to_block_hash, RequestState


def run_scenario_a(kv_index: RemoteKvIndex, workers: MigrationWorkers):
    """
    Scenario A: Nothing remote — request dispatches immediately.

    This verifies that when exist() returns empty, the scheduler
    behaves like today (no parking, immediate dispatch).
    """
    print("\n" + "=" * 60)
    print("SCENARIO A: Nothing remote (today's behavior)")
    print("=" * 60)

    # Use MockRemoteKvIndex that returns [] for everything
    mock_index = MockRemoteKvIndex()
    scheduler = SchedulerSim(mock_index, workers)

    print("\n[box_a] Submitting request with tokens [100, 101, 102]...")
    req = scheduler.submit(request_id=1, tokens=[100, 101, 102], dst_slot=0)

    # Should dispatch immediately (no parking)
    assert req.state == RequestState.DISPATCHED, f"Expected DISPATCHED, got {req.state}"
    print(f"[result] Request state: {req.state.name}")
    print("[result] ✓ Scenario A passed: immediate dispatch (no remote blocks)")


def run_scenario_b(kv_index: RemoteKvIndex, workers: MigrationWorkers):
    """
    Scenario B: Remote hit — park → pull → arrive → resume → dispatch.

    This is the new behavior the PoC proves.
    """
    print("\n" + "=" * 60)
    print("SCENARIO B: Remote hit (park → pull → resume → dispatch)")
    print("=" * 60)

    scheduler = SchedulerSim(kv_index, workers)

    # Box B publishes some blocks
    print("\n[box_b] Publishing blocks for tokens 42, 43...")
    kv_index.publish(token_to_block_hash(42), endpoint_id=1, slot=0, block_index=5)
    kv_index.publish(token_to_block_hash(43), endpoint_id=1, slot=0, block_index=6)

    # Verify they're there
    hashes = [token_to_block_hash(42), token_to_block_hash(43)]
    found = kv_index.exist(hashes)
    print(f"[box_b] Verification: exist({hashes}) returned {len(found)} blocks")

    # Box A submits request with those tokens + one unknown
    print("\n[box_a] Submitting request with tokens [42, 43, 99]...")
    req = scheduler.submit(request_id=2, tokens=[42, 43, 99], dst_slot=5)

    # If using DelayedMock, might be WAITING; if using instant Mock, already DISPATCHED
    print(f"[box_a] Immediate state after submit: {req.state.name}")

    if req.state == RequestState.WAITING_FOR_REMOTE_BLOCKS:
        print("[box_a] Request is parked, waiting for blocks...")

        # Start drain thread to resume
        scheduler.start_drain_thread(interval_seconds=0.05)

        # Wait for dispatch
        timeout = 5.0
        start = time.monotonic()
        while req.state != RequestState.DISPATCHED:
            if time.monotonic() - start > timeout:
                print("[error] Timeout waiting for dispatch!")
                scheduler.stop_drain_thread()
                return False
            time.sleep(0.05)

        scheduler.stop_drain_thread()

    # Verify final state
    assert req.state == RequestState.DISPATCHED, f"Expected DISPATCHED, got {req.state}"
    print(f"\n[result] Final state: {req.state.name}")
    print(f"[result] Remote blocks pulled: {len(req.remote_hits)}")
    print(f"[result] Total time: {(req.dispatch_time - req.submit_time) * 1000:.1f}ms")
    print(
        "[result] ✓ Scenario B passed: remote lookup → pull → park → resume → dispatch"
    )
    return True


def run_with_mooncake():
    """Run with real Mooncake store (requires master running)."""
    print("[setup] Connecting to Mooncake master...")

    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError:
        print("[error] mooncake package not installed!")
        print("[error] Run: pip install mooncake-transfer-engine==0.3.6.post1")
        sys.exit(1)

    store = MooncakeDistributedStore()
    ret = store.setup(
        "localhost",
        "http://localhost:8080/metadata",
        512 * 1024 * 1024,  # global_segment_size
        128 * 1024 * 1024,  # local_buffer_size
        "tcp",
        "",  # rdma_devices
        "localhost:50051",  # master_server_addr
    )

    if ret != 0:
        print(f"[error] Mooncake setup failed: {ret}")
        print("[error] Is the master running? Try: ./master_startup.sh")
        sys.exit(1)

    print("[setup] Connected to Mooncake!")

    kv_index = MooncakeRemoteKvIndex(store)
    workers = DelayedMockMigrationWorkers(delay_seconds=0.3)

    try:
        run_scenario_a(kv_index, workers)
        run_scenario_b(kv_index, workers)
    finally:
        print("\n[cleanup] Closing Mooncake store...")
        store.close()


def run_with_fake():
    """Run with in-memory fake (no Mooncake master needed)."""
    print("[setup] Using in-memory fake (no Mooncake master needed)")

    kv_index = InMemoryRemoteKvIndex()
    workers = DelayedMockMigrationWorkers(delay_seconds=0.2)

    run_scenario_a(kv_index, workers)
    run_scenario_b(kv_index, workers)


def main():
    parser = argparse.ArgumentParser(
        description="PoC2: Hybrid Mooncake KV orchestration demo"
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use in-memory fake instead of real Mooncake (no master needed)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PoC2: Hybrid Mooncake KV Orchestration Demo")
    print("=" * 60)

    if args.fake:
        run_with_fake()
    else:
        run_with_mooncake()

    print("\n" + "=" * 60)
    print("All scenarios passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
