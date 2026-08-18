# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
MigrationWorkers — "move the KV bytes" abstraction over migration worker(s).

This module provides the async interface for KV block transfers:
- pull(): start a non-blocking transfer (returns a handle)
- check_arrived_blocks(): poll for completed transfers
- is_complete(): check if a specific transfer is done

The PoC uses a mock that completes instantly. Real impl will wrap
MigrationLayerClient from tt-llm-engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import threading
import time


class PullStatus(Enum):
    PENDING = 1
    COMPLETED = 2
    FAILED = 3


@dataclass
class PullHandle:
    """Handle for tracking an async pull operation."""

    id: int
    blocks: List  # List[RemoteBlock]
    dst_slot: int
    status: PullStatus = PullStatus.PENDING
    error: Optional[str] = None


class MigrationWorkers(ABC):
    """Abstract interface for async KV block transfers."""

    @abstractmethod
    def pull(self, dst_slot: int, blocks: List) -> PullHandle:
        """
        Start an async transfer of blocks to dst_slot.

        This is non-blocking — returns immediately with a handle.
        Use check_arrived_blocks() or is_complete() to track completion.

        Args:
            dst_slot: Destination slot on this endpoint
            blocks: List of remote blocks to pull

        Returns:
            PullHandle for tracking the transfer
        """
        pass

    @abstractmethod
    def check_arrived_blocks(self) -> List[PullHandle]:
        """
        Poll for completed transfers.

        Called by the background drain thread. Returns handles that have
        completed since the last call (success or failure).

        Returns:
            List of completed PullHandle objects
        """
        pass

    @abstractmethod
    def is_complete(self, handle: PullHandle) -> bool:
        """
        Check if a specific transfer is complete.

        Args:
            handle: The handle returned by pull()

        Returns:
            True if transfer is done (success or failure)
        """
        pass


class MockMigrationWorkers(MigrationWorkers):
    """
    Mock that completes transfers instantly.

    Use this for testing the orchestration loop without real migration.
    """

    def __init__(self):
        self._next_id = 0
        self._pending: List[PullHandle] = []
        self._lock = threading.Lock()

    def pull(self, dst_slot: int, blocks: List) -> PullHandle:
        with self._lock:
            handle = PullHandle(
                id=self._next_id,
                blocks=blocks,
                dst_slot=dst_slot,
                status=PullStatus.COMPLETED,  # instant completion
            )
            self._next_id += 1
            self._pending.append(handle)
            return handle

    def check_arrived_blocks(self) -> List[PullHandle]:
        with self._lock:
            arrived = [h for h in self._pending if h.status != PullStatus.PENDING]
            self._pending = [h for h in self._pending if h.status == PullStatus.PENDING]
            return arrived

    def is_complete(self, handle: PullHandle) -> bool:
        return handle.status != PullStatus.PENDING


class DelayedMockMigrationWorkers(MigrationWorkers):
    """
    Mock with artificial delay — for visualizing park/resume in demos.

    Transfers complete after a configurable delay, so you can see the
    WAITING state before the drain thread resumes the request.
    """

    def __init__(self, delay_seconds: float = 0.5):
        self._delay = delay_seconds
        self._next_id = 0
        self._pending: List[tuple[PullHandle, float]] = []  # (handle, complete_time)
        self._lock = threading.Lock()

    def pull(self, dst_slot: int, blocks: List) -> PullHandle:
        with self._lock:
            handle = PullHandle(
                id=self._next_id,
                blocks=blocks,
                dst_slot=dst_slot,
                status=PullStatus.PENDING,
            )
            self._next_id += 1
            complete_time = time.monotonic() + self._delay
            self._pending.append((handle, complete_time))
            return handle

    def check_arrived_blocks(self) -> List[PullHandle]:
        now = time.monotonic()
        arrived = []
        still_pending = []

        with self._lock:
            for handle, complete_time in self._pending:
                if now >= complete_time:
                    handle.status = PullStatus.COMPLETED
                    arrived.append(handle)
                else:
                    still_pending.append((handle, complete_time))
            self._pending = still_pending

        return arrived

    def is_complete(self, handle: PullHandle) -> bool:
        return handle.status != PullStatus.PENDING


class CoalescingMigrationWorkers(MigrationWorkers):
    """
    Wrapper that coalesces contiguous blocks into fewer pull() calls.

    In the real impl, this would map to fewer migrate() calls to the
    migration worker by merging adjacent (endpoint, slot, block_index) ranges.

    For the PoC, this just demonstrates the coalescing logic.
    """

    def __init__(self, inner: MigrationWorkers):
        self._inner = inner

    def pull(self, dst_slot: int, blocks: List) -> PullHandle:
        coalesced = self._coalesce_blocks(blocks)
        return self._inner.pull(dst_slot, coalesced)

    def check_arrived_blocks(self) -> List[PullHandle]:
        return self._inner.check_arrived_blocks()

    def is_complete(self, handle: PullHandle) -> bool:
        return self._inner.is_complete(handle)

    def _coalesce_blocks(self, blocks: List) -> List:
        """
        Group contiguous blocks from the same (endpoint, slot).

        For now, just returns the input — real impl would merge
        adjacent block_index ranges into single migrate() calls.
        """
        if not blocks:
            return blocks

        # Sort by (endpoint, slot, block_index) for contiguity detection
        sorted_blocks = sorted(
            blocks, key=lambda b: (b.src_endpoint_id, b.src_slot, b.block_index)
        )

        # For PoC, return as-is. Real impl would merge ranges.
        return sorted_blocks
