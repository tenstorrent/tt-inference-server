# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
SchedulerSim — simulates the LLMPipeline::resolveSession orchestration loop.

This is the brain that decides:
1. Which blocks are needed (tokens → hashes)
2. Which are available locally (mock: none)
3. Which are available remotely (via RemoteKvIndex.exist())
4. Whether to pull and park, or dispatch immediately

State machine:
    RESOLVING → (no remote) → DISPATCHED
    RESOLVING → (has remote) → WAITING_FOR_REMOTE_BLOCKS → (arrived) → DISPATCHED
    RESOLVING → (has remote) → WAITING_FOR_REMOTE_BLOCKS → (timeout) → DISPATCHED (fallback)
    Any state → (cancel) → ABORTED
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Callable, Set
import threading
import time

from remote_kv_index import RemoteKvIndex, RemoteBlock
from migration_workers import MigrationWorkers, PullHandle, PullStatus


class RequestState(Enum):
    RESOLVING = 1
    WAITING_FOR_REMOTE_BLOCKS = 2
    DISPATCHED = 3
    ABORTED = 4


@dataclass
class Request:
    """A request being processed by the scheduler."""

    id: int
    tokens: List[int]
    dst_slot: int
    state: RequestState = RequestState.RESOLVING
    pull_handle: Optional[PullHandle] = None
    local_hits: List[str] = field(default_factory=list)
    remote_hits: List[RemoteBlock] = field(default_factory=list)
    submit_time: float = field(default_factory=time.monotonic)
    dispatch_time: Optional[float] = None


class SchedulerSim:
    """
    Simulates the scheduler's KV resolution logic.

    The flow:
    1. submit(request) → starts resolution
    2. _resolve() checks local, then remote
    3. If remote blocks found → pull() → state=WAITING
    4. drain() (background) → checks arrivals → resumes → dispatch
    """

    def __init__(
        self,
        kv_index: RemoteKvIndex,
        workers: MigrationWorkers,
        local_blocks: Optional[Set[str]] = None,
        timeout_seconds: float = 30.0,
        on_dispatch: Optional[Callable[[Request], None]] = None,
    ):
        """
        Args:
            kv_index: RemoteKvIndex for looking up remote blocks
            workers: MigrationWorkers for pulling blocks
            local_blocks: Set of block hashes available locally (mock)
            timeout_seconds: How long to wait for remote blocks before fallback
            on_dispatch: Callback when a request is dispatched
        """
        self.kv_index = kv_index
        self.workers = workers
        self.local_blocks = local_blocks or set()
        self.timeout_seconds = timeout_seconds
        self.on_dispatch = on_dispatch

        self._requests: Dict[int, Request] = {}
        self._dispatched: List[Request] = []
        self._lock = threading.Lock()

        # Background drain thread
        self._drain_running = False
        self._drain_thread: Optional[threading.Thread] = None

    def submit(self, request_id: int, tokens: List[int], dst_slot: int) -> Request:
        """
        Submit a new request for KV resolution.

        Args:
            request_id: Unique ID for this request
            tokens: Token IDs that need KV blocks
            dst_slot: Destination slot for any pulled blocks

        Returns:
            The Request object (can inspect state)
        """
        req = Request(id=request_id, tokens=tokens, dst_slot=dst_slot)

        with self._lock:
            self._requests[request_id] = req

        self._resolve(req)
        return req

    def cancel(self, request_id: int) -> bool:
        """
        Cancel a request (e.g., client disconnect).

        Returns:
            True if cancelled, False if not found or already dispatched
        """
        with self._lock:
            req = self._requests.get(request_id)
            if not req:
                return False
            if req.state == RequestState.DISPATCHED:
                return False
            req.state = RequestState.ABORTED
            return True

    def get_request(self, request_id: int) -> Optional[Request]:
        """Get a request by ID."""
        with self._lock:
            return self._requests.get(request_id)

    def get_dispatched(self) -> List[Request]:
        """Get list of dispatched requests."""
        with self._lock:
            return list(self._dispatched)

    def _resolve(self, req: Request):
        """
        Core resolution logic — called on submit.

        1. Convert tokens to block hashes
        2. Check local availability
        3. Check remote availability via exist()
        4. Pull if needed, or dispatch immediately
        """
        # Step 1: tokens → block hashes
        # Simplified: hash = "block_{token}"
        hashes = [f"block_{t}" for t in req.tokens]

        # Step 2: local lookup
        local_hits = [h for h in hashes if h in self.local_blocks]
        leftover = [h for h in hashes if h not in self.local_blocks]
        req.local_hits = local_hits

        print(
            f"[scheduler] Request {req.id}: {len(local_hits)} local hits, {len(leftover)} leftover"
        )

        if not leftover:
            # All blocks available locally
            self._dispatch(req, reason="all local")
            return

        # Step 3: remote lookup
        remote_blocks = self.kv_index.exist(leftover)
        req.remote_hits = remote_blocks

        print(
            f"[scheduler] Request {req.id}: exist() found {len(remote_blocks)} remote blocks"
        )

        if not remote_blocks:
            # Nothing remote either — dispatch (will prefill locally)
            self._dispatch(req, reason="no remote hits, prefill locally")
            return

        # Step 4: pull remote blocks
        print(
            f"[scheduler] Request {req.id}: pulling {len(remote_blocks)} blocks to slot {req.dst_slot}"
        )
        handle = self.workers.pull(req.dst_slot, remote_blocks)
        req.pull_handle = handle

        with self._lock:
            req.state = RequestState.WAITING_FOR_REMOTE_BLOCKS

        print(f"[scheduler] Request {req.id}: state → WAITING_FOR_REMOTE_BLOCKS")

    def drain(self) -> int:
        """
        Poll for completed transfers and resume waiting requests.

        Called by the background drain thread (or manually in tests).

        Returns:
            Number of requests resumed
        """
        arrived = self.workers.check_arrived_blocks()
        resumed = 0

        if arrived:
            print(f"[scheduler] drain(): {len(arrived)} transfers completed")

        with self._lock:
            for handle in arrived:
                for req in self._requests.values():
                    if req.state != RequestState.WAITING_FOR_REMOTE_BLOCKS:
                        continue
                    if req.pull_handle and req.pull_handle.id == handle.id:
                        if handle.status == PullStatus.COMPLETED:
                            self._dispatch_locked(req, reason="remote blocks arrived")
                        else:
                            # Failed — fallback to local prefill
                            self._dispatch_locked(
                                req, reason=f"pull failed: {handle.error}"
                            )
                        resumed += 1

            # Check for timeouts
            now = time.monotonic()
            for req in list(self._requests.values()):
                if req.state == RequestState.WAITING_FOR_REMOTE_BLOCKS:
                    elapsed = now - req.submit_time
                    if elapsed > self.timeout_seconds:
                        print(
                            f"[scheduler] Request {req.id}: timeout after {elapsed:.1f}s"
                        )
                        self._dispatch_locked(
                            req, reason="timeout, fallback to local prefill"
                        )
                        resumed += 1

        return resumed

    def _dispatch(self, req: Request, reason: str = ""):
        """Dispatch a request (acquires lock)."""
        with self._lock:
            self._dispatch_locked(req, reason)

    def _dispatch_locked(self, req: Request, reason: str = ""):
        """Dispatch a request (must hold lock)."""
        if req.state == RequestState.DISPATCHED:
            return
        if req.state == RequestState.ABORTED:
            return

        req.state = RequestState.DISPATCHED
        req.dispatch_time = time.monotonic()
        self._dispatched.append(req)

        elapsed = req.dispatch_time - req.submit_time
        print(
            f"[scheduler] DISPATCHED request {req.id} ({reason}) [{elapsed * 1000:.1f}ms]"
        )

        if self.on_dispatch:
            self.on_dispatch(req)

    # --- Background drain thread ---

    def start_drain_thread(self, interval_seconds: float = 0.01):
        """Start the background drain thread."""
        if self._drain_running:
            return

        self._drain_running = True
        self._drain_thread = threading.Thread(
            target=self._drain_loop, args=(interval_seconds,), daemon=True
        )
        self._drain_thread.start()
        print("[scheduler] Drain thread started")

    def stop_drain_thread(self):
        """Stop the background drain thread."""
        self._drain_running = False
        if self._drain_thread:
            self._drain_thread.join(timeout=1.0)
            self._drain_thread = None
        print("[scheduler] Drain thread stopped")

    def _drain_loop(self, interval: float):
        """Background drain loop."""
        while self._drain_running:
            self.drain()
            time.sleep(interval)


def token_to_block_hash(token: int) -> str:
    """Convert a token ID to its block hash (simplified for PoC)."""
    return f"block_{token}"
