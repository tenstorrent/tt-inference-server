#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""
Stress test: sends many memory requests rapidly through SHM to the C++
bridge helper and verifies all results come back with correct task IDs,
correct success/failure, and in FIFO order.
"""

import os
import subprocess
import sys
import time

BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "build")
HELPER_BIN = os.path.join(BUILD_DIR, "memory_manager_test")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "runners"))
from shared_memory import MemoryRequestSharedMemory, MemoryResultSharedMemory

ALLOCATE = 0
DEALLOCATE = 1
PAGED = 0
NUM_REQUESTS = 50


def wait_for_ready(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line.strip() == "READY":
            return
        if proc.poll() is not None:
            raise RuntimeError(
                f"Helper exited early with code {proc.returncode}"
            )
    raise TimeoutError("Helper did not print READY in time")


def collect_results(res, count: int, timeout: float = 10.0):
    results = []
    deadline = time.monotonic() + timeout
    while len(results) < count and time.monotonic() < deadline:
        r = res.try_read_result()
        if r is not None:
            results.append(r)
        else:
            time.sleep(0.0005)
    return results


def test_burst_allocations():
    """Send NUM_REQUESTS unique allocations rapidly, verify all succeed in order."""
    proc = subprocess.Popen(
        [HELPER_BIN, "--bridge", str(NUM_REQUESTS), "30000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_ready(proc)

        req = MemoryRequestSharedMemory("tt_mem_requests")
        req.open(create=False)
        res = MemoryResultSharedMemory("tt_mem_results")
        res.open(create=False)

        task_ids = [f"burst-{i:04d}" for i in range(NUM_REQUESTS)]
        for tid in task_ids:
            req.write_request(tid, input_seq_len=16, action=ALLOCATE, memory_layout=PAGED)

        results = collect_results(res, NUM_REQUESTS)

        assert len(results) == NUM_REQUESTS, (
            f"Expected {NUM_REQUESTS} results, got {len(results)}"
        )
        for i, r in enumerate(results):
            assert r.task_id == task_ids[i], (
                f"Result {i}: expected task_id={task_ids[i]!r}, got {r.task_id!r}"
            )
            assert r.success, f"Result {i} ({r.task_id}) should have succeeded"

        req.close()
        res.close()
        proc.wait(timeout=5)
        assert proc.returncode == 0, f"Helper exited with {proc.returncode}"
    finally:
        proc.kill()
        proc.wait()


def test_interleaved_allocate_deallocate():
    """Allocate N tasks, then deallocate them all. Verify correct results."""
    total = NUM_REQUESTS * 2
    proc = subprocess.Popen(
        [HELPER_BIN, "--bridge", str(total), "30000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_ready(proc)

        req = MemoryRequestSharedMemory("tt_mem_requests")
        req.open(create=False)
        res = MemoryResultSharedMemory("tt_mem_results")
        res.open(create=False)

        task_ids = [f"interleave-{i:04d}" for i in range(NUM_REQUESTS)]

        for tid in task_ids:
            req.write_request(tid, input_seq_len=8, action=ALLOCATE, memory_layout=PAGED)

        alloc_results = collect_results(res, NUM_REQUESTS)
        assert len(alloc_results) == NUM_REQUESTS, (
            f"Expected {NUM_REQUESTS} alloc results, got {len(alloc_results)}"
        )
        for i, r in enumerate(alloc_results):
            assert r.task_id == task_ids[i], (
                f"Alloc {i}: expected {task_ids[i]!r}, got {r.task_id!r}"
            )
            assert r.success

        for tid in task_ids:
            req.write_request(tid, input_seq_len=0, action=DEALLOCATE, memory_layout=PAGED)

        dealloc_results = collect_results(res, NUM_REQUESTS)
        assert len(dealloc_results) == NUM_REQUESTS, (
            f"Expected {NUM_REQUESTS} dealloc results, got {len(dealloc_results)}"
        )
        for i, r in enumerate(dealloc_results):
            assert r.task_id == task_ids[i], (
                f"Dealloc {i}: expected {task_ids[i]!r}, got {r.task_id!r}"
            )
            assert r.success

        req.close()
        res.close()
        proc.wait(timeout=5)
        assert proc.returncode == 0
    finally:
        proc.kill()
        proc.wait()


def test_rapid_mixed_success_failure():
    """Mix valid allocations and invalid duplicates, verify correct success/failure."""
    unique_count = NUM_REQUESTS // 2
    total = unique_count * 2
    proc = subprocess.Popen(
        [HELPER_BIN, "--bridge", str(total), "30000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_ready(proc)

        req = MemoryRequestSharedMemory("tt_mem_requests")
        req.open(create=False)
        res = MemoryResultSharedMemory("tt_mem_results")
        res.open(create=False)

        task_ids = [f"mixed-{i:04d}" for i in range(unique_count)]
        expected_success = []

        for tid in task_ids:
            req.write_request(tid, input_seq_len=4, action=ALLOCATE, memory_layout=PAGED)
            expected_success.append(True)

        for tid in task_ids:
            req.write_request(tid, input_seq_len=4, action=ALLOCATE, memory_layout=PAGED)
            expected_success.append(False)

        results = collect_results(res, total)
        assert len(results) == total, (
            f"Expected {total} results, got {len(results)}"
        )

        for i in range(unique_count):
            assert results[i].success, f"First alloc {task_ids[i]} should succeed"
        for i in range(unique_count):
            idx = unique_count + i
            assert not results[idx].success, (
                f"Duplicate alloc {task_ids[i]} should fail"
            )

        req.close()
        res.close()
        proc.wait(timeout=5)
        assert proc.returncode == 0
    finally:
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    tests = [
        ("burst_allocations", test_burst_allocations),
        ("interleaved_allocate_deallocate", test_interleaved_allocate_deallocate),
        ("rapid_mixed_success_failure", test_rapid_mixed_success_failure),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
