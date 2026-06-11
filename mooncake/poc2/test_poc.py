#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
test_poc.py — Unit tests for the hybrid Mooncake KV orchestration PoC.

All tests use in-memory mocks — no Mooncake master required.

Run:
    python test_poc.py
    python -m pytest test_poc.py -v
"""

import time
import unittest
from typing import List

from remote_kv_index import (
    RemoteBlock,
    MockRemoteKvIndex,
    InMemoryRemoteKvIndex,
)
from migration_workers import (
    MockMigrationWorkers,
    DelayedMockMigrationWorkers,
    PullStatus,
)
from scheduler_sim import (
    SchedulerSim,
    RequestState,
    token_to_block_hash,
)


class TestRemoteKvIndex(unittest.TestCase):
    """Tests for RemoteKvIndex implementations."""

    def test_mock_returns_empty(self):
        """MockRemoteKvIndex.exist() always returns empty."""
        index = MockRemoteKvIndex()
        result = index.exist(["hash1", "hash2", "hash3"])
        self.assertEqual(result, [])

    def test_in_memory_publish_and_exist(self):
        """InMemoryRemoteKvIndex stores and retrieves blocks."""
        index = InMemoryRemoteKvIndex()

        # Publish some blocks
        index.publish("hash_a", endpoint_id=1, slot=0, block_index=10)
        index.publish("hash_b", endpoint_id=2, slot=5, block_index=20)

        # Check existence
        result = index.exist(["hash_a", "hash_b", "hash_c"])
        self.assertEqual(len(result), 2)

        # Verify contents
        by_hash = {b.hash: b for b in result}
        self.assertIn("hash_a", by_hash)
        self.assertEqual(by_hash["hash_a"].src_endpoint_id, 1)
        self.assertEqual(by_hash["hash_a"].block_index, 10)

        self.assertIn("hash_b", by_hash)
        self.assertEqual(by_hash["hash_b"].src_endpoint_id, 2)
        self.assertEqual(by_hash["hash_b"].block_index, 20)

        self.assertNotIn("hash_c", by_hash)

    def test_in_memory_remove(self):
        """InMemoryRemoteKvIndex.remove() deletes entries."""
        index = InMemoryRemoteKvIndex()
        index.publish("hash_x", endpoint_id=1, slot=0, block_index=0)

        self.assertEqual(len(index.exist(["hash_x"])), 1)
        self.assertTrue(index.remove("hash_x"))
        self.assertEqual(len(index.exist(["hash_x"])), 0)
        self.assertFalse(index.remove("hash_x"))  # already gone


class TestMigrationWorkers(unittest.TestCase):
    """Tests for MigrationWorkers implementations."""

    def test_mock_instant_completion(self):
        """MockMigrationWorkers completes pulls instantly."""
        workers = MockMigrationWorkers()

        blocks = [
            RemoteBlock("h1", block_index=0, src_endpoint_id=1, src_slot=0),
            RemoteBlock("h2", block_index=1, src_endpoint_id=1, src_slot=0),
        ]
        handle = workers.pull(dst_slot=5, blocks=blocks)

        self.assertEqual(handle.status, PullStatus.COMPLETED)
        self.assertTrue(workers.is_complete(handle))

        arrived = workers.check_arrived_blocks()
        self.assertEqual(len(arrived), 1)
        self.assertEqual(arrived[0].id, handle.id)

    def test_delayed_mock_waits(self):
        """DelayedMockMigrationWorkers waits before completing."""
        workers = DelayedMockMigrationWorkers(delay_seconds=0.1)

        blocks = [RemoteBlock("h1", 0, 1, 0)]
        handle = workers.pull(dst_slot=0, blocks=blocks)

        # Initially pending
        self.assertEqual(handle.status, PullStatus.PENDING)
        self.assertFalse(workers.is_complete(handle))

        # Nothing arrived yet
        arrived = workers.check_arrived_blocks()
        self.assertEqual(len(arrived), 0)

        # Wait for completion
        time.sleep(0.15)
        arrived = workers.check_arrived_blocks()
        self.assertEqual(len(arrived), 1)
        self.assertEqual(arrived[0].status, PullStatus.COMPLETED)


class TestSchedulerSim(unittest.TestCase):
    """Tests for the scheduler orchestration loop."""

    def test_no_remote_blocks_dispatches_immediately(self):
        """When exist() returns [], dispatch immediately (today's behavior)."""
        index = MockRemoteKvIndex()  # always returns []
        workers = MockMigrationWorkers()
        scheduler = SchedulerSim(index, workers)

        req = scheduler.submit(request_id=1, tokens=[1, 2, 3], dst_slot=0)

        self.assertEqual(req.state, RequestState.DISPATCHED)
        self.assertEqual(len(req.remote_hits), 0)

    def test_all_local_dispatches_immediately(self):
        """When all blocks are local, dispatch immediately."""
        index = InMemoryRemoteKvIndex()
        workers = MockMigrationWorkers()

        # Mark some blocks as local
        local_blocks = {token_to_block_hash(1), token_to_block_hash(2)}
        scheduler = SchedulerSim(index, workers, local_blocks=local_blocks)

        req = scheduler.submit(request_id=1, tokens=[1, 2], dst_slot=0)

        self.assertEqual(req.state, RequestState.DISPATCHED)
        self.assertEqual(len(req.local_hits), 2)
        self.assertEqual(len(req.remote_hits), 0)

    def test_remote_hit_with_instant_mock(self):
        """Remote blocks found → instant mock completes → dispatch."""
        index = InMemoryRemoteKvIndex()
        workers = MockMigrationWorkers()
        scheduler = SchedulerSim(index, workers)

        # Publish remote blocks
        index.publish(token_to_block_hash(10), endpoint_id=2, slot=3, block_index=0)
        index.publish(token_to_block_hash(11), endpoint_id=2, slot=3, block_index=1)

        req = scheduler.submit(request_id=1, tokens=[10, 11, 99], dst_slot=0)

        # With instant mock, should dispatch in one drain() call
        scheduler.drain()

        self.assertEqual(req.state, RequestState.DISPATCHED)
        self.assertEqual(len(req.remote_hits), 2)

    def test_remote_hit_with_delayed_mock(self):
        """Remote blocks found → delayed mock → WAITING → drain → dispatch."""
        index = InMemoryRemoteKvIndex()
        workers = DelayedMockMigrationWorkers(delay_seconds=0.1)
        scheduler = SchedulerSim(index, workers)

        # Publish remote blocks
        index.publish(token_to_block_hash(20), endpoint_id=1, slot=0, block_index=0)

        req = scheduler.submit(request_id=1, tokens=[20], dst_slot=0)

        # Should be waiting
        self.assertEqual(req.state, RequestState.WAITING_FOR_REMOTE_BLOCKS)
        self.assertIsNotNone(req.pull_handle)

        # Drain won't find anything yet
        scheduler.drain()
        self.assertEqual(req.state, RequestState.WAITING_FOR_REMOTE_BLOCKS)

        # Wait for delay
        time.sleep(0.15)

        # Now drain should find the completed transfer
        scheduler.drain()
        self.assertEqual(req.state, RequestState.DISPATCHED)

    def test_timeout_fallback(self):
        """Request times out → dispatches anyway (fallback to local prefill)."""
        index = InMemoryRemoteKvIndex()
        workers = DelayedMockMigrationWorkers(delay_seconds=10.0)  # won't complete
        scheduler = SchedulerSim(index, workers, timeout_seconds=0.1)

        index.publish(token_to_block_hash(30), endpoint_id=1, slot=0, block_index=0)

        req = scheduler.submit(request_id=1, tokens=[30], dst_slot=0)
        self.assertEqual(req.state, RequestState.WAITING_FOR_REMOTE_BLOCKS)

        # Wait past timeout
        time.sleep(0.15)
        scheduler.drain()

        self.assertEqual(req.state, RequestState.DISPATCHED)

    def test_cancel_waiting_request(self):
        """Cancel a waiting request → state becomes ABORTED."""
        index = InMemoryRemoteKvIndex()
        workers = DelayedMockMigrationWorkers(delay_seconds=10.0)
        scheduler = SchedulerSim(index, workers)

        index.publish(token_to_block_hash(40), endpoint_id=1, slot=0, block_index=0)

        req = scheduler.submit(request_id=1, tokens=[40], dst_slot=0)
        self.assertEqual(req.state, RequestState.WAITING_FOR_REMOTE_BLOCKS)

        self.assertTrue(scheduler.cancel(request_id=1))
        self.assertEqual(req.state, RequestState.ABORTED)

    def test_drain_thread(self):
        """Background drain thread resumes waiting requests."""
        index = InMemoryRemoteKvIndex()
        workers = DelayedMockMigrationWorkers(delay_seconds=0.1)
        scheduler = SchedulerSim(index, workers)

        index.publish(token_to_block_hash(50), endpoint_id=1, slot=0, block_index=0)

        scheduler.start_drain_thread(interval_seconds=0.02)

        try:
            req = scheduler.submit(request_id=1, tokens=[50], dst_slot=0)

            # Wait for drain thread to pick it up
            deadline = time.monotonic() + 1.0
            while req.state != RequestState.DISPATCHED:
                if time.monotonic() > deadline:
                    self.fail("Timeout waiting for drain thread to dispatch")
                time.sleep(0.02)

            self.assertEqual(req.state, RequestState.DISPATCHED)
        finally:
            scheduler.stop_drain_thread()

    def test_on_dispatch_callback(self):
        """on_dispatch callback fires when request dispatches."""
        index = MockRemoteKvIndex()
        workers = MockMigrationWorkers()

        dispatched_ids: List[int] = []

        def on_dispatch(req):
            dispatched_ids.append(req.id)

        scheduler = SchedulerSim(index, workers, on_dispatch=on_dispatch)
        scheduler.submit(request_id=42, tokens=[1], dst_slot=0)

        self.assertEqual(dispatched_ids, [42])

    def test_partial_remote_hit(self):
        """Some blocks remote, some missing → pulls only the remote ones."""
        index = InMemoryRemoteKvIndex()
        workers = MockMigrationWorkers()
        scheduler = SchedulerSim(index, workers)

        # Only publish one of three
        index.publish(token_to_block_hash(60), endpoint_id=1, slot=0, block_index=0)

        req = scheduler.submit(request_id=1, tokens=[60, 61, 62], dst_slot=0)
        scheduler.drain()

        self.assertEqual(req.state, RequestState.DISPATCHED)
        self.assertEqual(len(req.remote_hits), 1)
        self.assertEqual(req.remote_hits[0].hash, token_to_block_hash(60))


class TestTokenToBlockHash(unittest.TestCase):
    """Tests for the token → hash mapping."""

    def test_deterministic(self):
        """Same token always produces same hash."""
        h1 = token_to_block_hash(123)
        h2 = token_to_block_hash(123)
        self.assertEqual(h1, h2)

    def test_different_tokens_different_hashes(self):
        """Different tokens produce different hashes."""
        h1 = token_to_block_hash(100)
        h2 = token_to_block_hash(200)
        self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
