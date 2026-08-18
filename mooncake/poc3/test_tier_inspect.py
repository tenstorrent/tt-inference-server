#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
test_tier_inspect.py — Authoritative tier labeling via get_replica_desc.

Unlike the latency heuristic, MooncakeClient.classify_read_tier() inspects the
same replica list the read path selects from, so it reports the REAL tier,
named after Mooncake's own replica types in its priority order:

  local DRAM > remote DRAM > local_disk (offload RPC) > shared disk (DFS)

Demos:
  1. local DRAM -> shared disk : one client, key evicted from DRAM to disk.
  2. remote DRAM               : consumer reads a key still live in a
                                 publisher's DRAM.
  3. shared disk               : consumer reads a key that exists ONLY on disk
                                 (the publisher evicted its DRAM copy), so no
                                 memory replica exists anywhere.

Requires a master started with offload + the HTTP metadata server:
    ./master_startup.sh

Two gotchas this test is built around:
  1. classify_read_tier() calls get_replica_desc(), which grants a read-lease.
     A leased object will NOT be evicted, so we never touch the target between
     put and flood, and we wait out the lease TTL first.
  2. The file tier is only populated when the master runs with --enable_offload.

Single-host note: the "remote SSD via offload RPC" sub-tier needs each node to
have its OWN (non-shared) disk. With one shared root_fs_dir the disk replica is
a shared-DFS (DISK) replica, read directly — so offload_rpc_read_count stays 0.
"""

import sys
import time
import uuid
import multiprocessing
from client import MooncakeClient, create_test_data

# Master defaults: default_kv_lease_ttl=5s. Wait past it (plus async offload).
LEASE_AND_OFFLOAD_WAIT_S = 7.0
EVICTION_WAIT_S = 3.0
FLOOD_ITEMS = 400
FLOOD_ITEM_KB = 64
ITEM_KB = 64


def _unique_key(prefix: str) -> str:
    """Fresh key per run — the master persists keys on disk across runs, and a
    reused key returns its OLD (often disk-only) replicas instead of a new one."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _force_eviction_of_target(client: MooncakeClient):
    """Flood DRAM so the (unleased) target's DRAM replica gets evicted."""
    print("\n[step 3] Flood DRAM WITHOUT touching the target (so it can evict)...")
    client.fill_dram(_unique_key("tier_flood"), FLOOD_ITEMS, size_kb=FLOOD_ITEM_KB)
    print(f"[step 4] Wait {EVICTION_WAIT_S}s for background eviction...")
    time.sleep(EVICTION_WAIT_S)


def demo_local_dram_to_disk() -> bool:
    """Single process: watch a key move from local DRAM to shared disk."""
    print("=" * 60)
    print("DEMO 1: local DRAM -> shared disk (DFS)")
    print("=" * 60)

    client = MooncakeClient("tier_inspect", dram_size_mb=16)
    if not client.connect():
        print("[FAIL] connect failed — is master up? run ./master_startup.sh")
        return False

    target_key = _unique_key("tier_inspect_target")
    try:
        print("\n[step 1] Put target, confirm it starts in local DRAM...")
        client.put(target_key, create_test_data(size_kb=ITEM_KB))
        tier_initial = client.classify_read_tier(target_key)
        print(f"  tier = {tier_initial}")

        print(
            f"\n[step 2] Wait {LEASE_AND_OFFLOAD_WAIT_S}s "
            "(async offload to SSD + read-lease expiry)..."
        )
        time.sleep(LEASE_AND_OFFLOAD_WAIT_S)
        on_disk = len(client.list_dfs_keys(limit=10)) > 0
        print(f"  file-tier replica present on disk: {on_disk}")

        _force_eviction_of_target(client)

        tier_final = client.classify_read_tier(target_key)
        data, result = client.get(target_key, inspect_tier=False)
        data_ok = data == create_test_data(size_kb=ITEM_KB)
        print(f"\n[result] tier = {tier_final}")
        print(
            f"  get(): success={result.success}  data_ok={data_ok}  "
            f"{result.latency_ms:.2f}ms"
        )

        passed = (
            tier_initial == client.TIER_LOCAL_DRAM
            and tier_final == client.TIER_SHARED_DISK
        )
        print(
            "\n"
            + ("[PASS]" if passed else "[WARN]")
            + f" local DRAM -> disk transition {'observed' if passed else 'NOT observed'}"
        )
        return passed
    finally:
        client.close()


def _publisher(remote_key, ready_event, done_event):
    """Publisher: put a key that stays in THIS process's DRAM, then idle."""
    client = MooncakeClient("publisher", dram_size_mb=32)
    if not client.connect():
        ready_event.set()
        return
    try:
        client.put(remote_key, create_test_data(size_kb=ITEM_KB))
        print(f"[publisher] put '{remote_key}' (lives in publisher DRAM)")
        ready_event.set()
        done_event.wait(timeout=30.0)
    finally:
        client.close()


def _consumer(remote_key, ready_event, done_event, results_queue):
    """Consumer: empty DRAM, so the key must be a REMOTE DRAM replica."""
    ready_event.wait(timeout=30.0)
    time.sleep(0.5)
    client = MooncakeClient("consumer", dram_size_mb=32)
    if not client.connect():
        results_queue.put(None)
        done_event.set()
        return
    try:
        tier = client.classify_read_tier(remote_key)
        data, result = client.get(remote_key, inspect_tier=False)
        print(f"[consumer] tier = {tier}")
        print(f"[consumer] my endpoint = {client.local_endpoint}")
        print(
            f"[consumer] offload RPC reads = {client._store.get_offload_rpc_read_count()}"
        )
        results_queue.put(
            {
                "tier": tier,
                "success": result.success,
                "data_ok": data == create_test_data(size_kb=ITEM_KB),
            }
        )
    finally:
        client.close()
        done_event.set()


def demo_remote_dram() -> bool:
    """Two processes: consumer reads a key still live in the publisher's DRAM."""
    print("\n" + "=" * 60)
    print("DEMO 2: remote DRAM (data in another process's memory)")
    print("=" * 60)

    # 'spawn' = fresh interpreter per child. Required because demo 1 already
    # initialized Mooncake (gRPC threads) in this process, and forking a
    # threaded gRPC client deadlocks the children.
    ctx = multiprocessing.get_context("spawn")
    ready_event = ctx.Event()
    done_event = ctx.Event()
    results_queue = ctx.Queue()
    remote_key = _unique_key("remote_key")

    publisher = ctx.Process(
        target=_publisher, args=(remote_key, ready_event, done_event)
    )
    consumer = ctx.Process(
        target=_consumer, args=(remote_key, ready_event, done_event, results_queue)
    )
    publisher.start()
    consumer.start()
    publisher.join(timeout=60.0)
    consumer.join(timeout=60.0)

    result = results_queue.get() if not results_queue.empty() else None
    if not result:
        print("\n[WARN] consumer produced no result")
        return False

    passed = result["tier"] == MooncakeClient.TIER_REMOTE_DRAM and result["data_ok"]
    print(
        "\n"
        + ("[PASS]" if passed else "[WARN]")
        + f" consumer saw '{result['tier']}', data_ok={result['data_ok']}"
    )
    return passed


def _cold_publisher(cold_key, ready_event, done_event):
    """Publisher: put a key, then evict its OWN DRAM copy so only disk remains."""
    client = MooncakeClient("cold_publisher", dram_size_mb=16)
    if not client.connect():
        ready_event.set()
        return
    try:
        client.put(cold_key, create_test_data(size_kb=ITEM_KB))
        print(f"[cold_publisher] put '{cold_key}', waiting for offload + lease...")
        time.sleep(LEASE_AND_OFFLOAD_WAIT_S)
        print("[cold_publisher] flooding own DRAM to evict the key's memory copy...")
        client.fill_dram(_unique_key("cold_flood"), FLOOD_ITEMS, size_kb=FLOOD_ITEM_KB)
        time.sleep(EVICTION_WAIT_S)
        ready_event.set()
        done_event.wait(timeout=30.0)
    finally:
        client.close()


def _cold_consumer(cold_key, ready_event, done_event, results_queue):
    """Consumer: read a key that exists ONLY on disk (no memory replica left)."""
    ready_event.wait(timeout=40.0)
    client = MooncakeClient("cold_consumer", dram_size_mb=32)
    if not client.connect():
        results_queue.put(None)
        done_event.set()
        return
    try:
        tier = client.classify_read_tier(cold_key)
        data, result = client.get(cold_key, inspect_tier=False)
        print(f"[cold_consumer] tier = {tier}")
        print(
            f"[cold_consumer] offload RPC reads = {client._store.get_offload_rpc_read_count()}"
        )
        results_queue.put(
            {
                "tier": tier,
                "success": result.success,
                "data_ok": data == create_test_data(size_kb=ITEM_KB),
            }
        )
    finally:
        client.close()
        done_event.set()


def demo_shared_disk() -> bool:
    """Two processes: consumer reads a key that lives ONLY on disk."""
    print("\n" + "=" * 60)
    print("DEMO 3: shared disk (key evicted from all DRAM, read from disk)")
    print("=" * 60)

    ctx = multiprocessing.get_context("spawn")
    ready_event = ctx.Event()
    done_event = ctx.Event()
    results_queue = ctx.Queue()
    cold_key = _unique_key("cold_key")

    publisher = ctx.Process(
        target=_cold_publisher, args=(cold_key, ready_event, done_event)
    )
    consumer = ctx.Process(
        target=_cold_consumer, args=(cold_key, ready_event, done_event, results_queue)
    )
    publisher.start()
    consumer.start()
    publisher.join(timeout=60.0)
    consumer.join(timeout=60.0)

    result = results_queue.get() if not results_queue.empty() else None
    if not result:
        print("\n[WARN] consumer produced no result")
        return False

    passed = result["tier"] == MooncakeClient.TIER_SHARED_DISK and result["data_ok"]
    print(
        "\n"
        + ("[PASS]" if passed else "[WARN]")
        + f" consumer saw '{result['tier']}', data_ok={result['data_ok']}"
    )
    return passed


def main():
    local_to_disk_ok = demo_local_dram_to_disk()
    remote_dram_ok = demo_remote_dram()
    shared_disk_ok = demo_shared_disk()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'[PASS]' if local_to_disk_ok else '[WARN]'} local DRAM -> shared disk")
    print(f"  {'[PASS]' if remote_dram_ok else '[WARN]'} remote DRAM (cross-process)")
    print(f"  {'[PASS]' if shared_disk_ok else '[WARN]'} shared disk (cross-process)")
    print("=" * 60)
    sys.exit(0 if (local_to_disk_ok and remote_dram_ok and shared_disk_ok) else 1)


if __name__ == "__main__":
    main()
