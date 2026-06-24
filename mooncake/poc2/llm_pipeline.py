# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
llm_pipeline.py — Python adapter mirroring the real LLMPipeline::resolveSession.

This mirrors the C++ LLMPipeline from:
  tt-inference-server/tt-media-server/cpp_server/src/services/llm_pipeline.cpp

The key method is resolveSession() which:
1. computeRoutingInfo(req) → tokens → block hashes
2. sessionManager_->tryAcquireByPrefixHash(blocks) → local lookup
3. [NEW] remoteKvIndex_->exist(leftover) → remote lookup via Mooncake
4. [NEW] migrationWorkers_->pull() → async pull, park request
5. createSession() if no match, or dispatch with continuation

This adapter implements the full flow using the real SessionManager pattern
plus the Mooncake remote lookup extension.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Callable
import threading
import time
import hashlib

from session_manager import SessionManager, BlockHashInfo, AcquiredSession
from remote_kv_index import RemoteKvIndex, RemoteBlock
from migration_workers import MigrationWorkers, PullHandle, PullStatus


class ResolveState(Enum):
    """Request resolution state."""

    RESOLVING = 1
    WAITING_FOR_REMOTE_BLOCKS = 2
    DISPATCHED = 3
    ABORTED = 4


@dataclass
class ResolveRequest:
    """A request being resolved by LLMPipeline."""

    request_id: str
    tokens: List[int]
    state: ResolveState = ResolveState.RESOLVING

    # Resolution results
    local_session: Optional[AcquiredSession] = None
    remote_blocks: List[RemoteBlock] = field(default_factory=list)
    pull_handle: Optional[PullHandle] = None
    allocated_slot: Optional[int] = None

    # Timing
    submit_time: float = field(default_factory=time.monotonic)
    dispatch_time: Optional[float] = None

    # Dispatch info
    matched_tokens: int = 0
    is_continuation: bool = False


def compute_block_hashes(
    tokens: List[int], block_size: int = 64
) -> List[BlockHashInfo]:
    """
    Mirrors computePrefixCachingInfoFromTokens from conversation_hasher.hpp.

    Computes per-block hashes using chained xxhash (simplified to sha256 for PoC).
    Real impl uses xxh64 with parent hash as seed.

    Args:
        tokens: Token IDs
        block_size: Tokens per block (from config, typically 64)

    Returns:
        List of BlockHashInfo, one per full block
    """
    blocks = []
    parent_hash = 0

    for i in range(0, len(tokens), block_size):
        block_tokens = tokens[i : i + block_size]
        if len(block_tokens) < block_size:
            break  # partial block not included

        # Chain hash: xxh64(block_tokens, seed=parent_hash)
        # Simplified: use sha256 truncated to 64 bits
        h = hashlib.sha256()
        h.update(parent_hash.to_bytes(8, "little"))
        for t in block_tokens:
            h.update(t.to_bytes(4, "little"))
        block_hash = int.from_bytes(h.digest()[:8], "little")

        blocks.append(BlockHashInfo(hash=block_hash))
        parent_hash = block_hash

    return blocks


class LLMPipeline:
    """
    Python adapter mirroring the real LLMPipeline.

    Implements resolveSession() with the Mooncake extension:
    1. Local lookup via SessionManager
    2. Remote lookup via RemoteKvIndex (Mooncake)
    3. Async pull via MigrationWorkers
    4. Park/resume pattern for waiting requests

    This is the integration point where Mooncake hooks into the real scheduler.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        remote_kv_index: RemoteKvIndex,
        migration_workers: MigrationWorkers,
        endpoint_id: int = 0,
        timeout_seconds: float = 30.0,
        on_dispatch: Optional[Callable[[ResolveRequest], None]] = None,
    ):
        """
        Args:
            session_manager: Local session/slot manager (real component)
            remote_kv_index: Mooncake-backed remote block lookup
            migration_workers: Async KV block transfer
            endpoint_id: This endpoint's ID (for publishing)
            timeout_seconds: Max wait for remote blocks
            on_dispatch: Callback when request dispatches
        """
        self._session_manager = session_manager
        self._remote_kv_index = remote_kv_index
        self._migration_workers = migration_workers
        self._endpoint_id = endpoint_id
        self._timeout_seconds = timeout_seconds
        self._on_dispatch = on_dispatch

        self._requests: Dict[str, ResolveRequest] = {}
        self._lock = threading.Lock()

        # Background drain
        self._drain_running = False
        self._drain_thread: Optional[threading.Thread] = None

    def resolve_session(self, request_id: str, tokens: List[int]) -> ResolveRequest:
        """
        Mirrors LLMPipeline::resolveSession (llm_pipeline.cpp:146-210).

        The full resolution flow:
        1. Compute block hashes from tokens
        2. LOCAL lookup: sessionManager.tryAcquireByPrefixHash()
        3. If miss → REMOTE lookup: remoteKvIndex.exist(leftover)
        4. If remote hit → pull() + park
        5. Otherwise → allocate new slot, dispatch

        Args:
            request_id: Unique request identifier
            tokens: Input token IDs

        Returns:
            ResolveRequest (inspect .state for current status)
        """
        req = ResolveRequest(request_id=request_id, tokens=tokens)

        with self._lock:
            self._requests[request_id] = req

        self._resolve(req)
        return req

    def _resolve(self, req: ResolveRequest):
        """Core resolution logic."""
        # Step 1: tokens → block hashes
        blocks = compute_block_hashes(req.tokens)
        block_hashes = [b.hash for b in blocks]

        print(
            f"[pipeline] Request {req.request_id}: {len(req.tokens)} tokens → {len(blocks)} blocks"
        )

        if not blocks:
            # No full blocks — allocate fresh slot
            self._allocate_and_dispatch(req, reason="no blocks to match")
            return

        # Step 2: LOCAL lookup
        # This is the existing code path in the real LLMPipeline
        local_match = self._session_manager.try_acquire_by_prefix_hash(blocks)

        if local_match:
            req.local_session = local_match
            req.matched_tokens = local_match.matched_tokens
            req.is_continuation = True
            print(
                f"[pipeline] Request {req.request_id}: LOCAL HIT session={local_match.session_id} "
                f"matched={local_match.matched_blocks} blocks"
            )
            self._dispatch(req, reason="local prefix hit")
            return

        print(f"[pipeline] Request {req.request_id}: local miss, checking remote...")

        # ========== NEW: REMOTE LOOKUP (Mooncake integration point) ==========
        # This is where the PoC extends the real LLMPipeline
        #
        # In the real C++ code, this would be:
        #   auto remoteBlocks = remoteKvIndex_->exist(blockHashes);
        #   if (!remoteBlocks.empty()) {
        #       auto handle = migrationWorkers_->pull(slot, remoteBlocks);
        #       parkRequest(req, handle);
        #       return;
        #   }

        remote_blocks = self._remote_kv_index.exist(block_hashes)
        req.remote_blocks = remote_blocks

        if not remote_blocks:
            # Nothing remote — allocate fresh (today's behavior)
            print(
                f"[pipeline] Request {req.request_id}: no remote blocks, allocating fresh slot"
            )
            self._allocate_and_dispatch(req, reason="no remote, fresh prefill")
            return

        print(
            f"[pipeline] Request {req.request_id}: REMOTE HIT {len(remote_blocks)} blocks"
        )

        # Step 3: Allocate slot for incoming blocks
        slot_id = self._session_manager.allocate_slot(req.request_id)
        if slot_id is None:
            print(f"[pipeline] Request {req.request_id}: no free slots!")
            req.state = ResolveState.ABORTED
            return

        req.allocated_slot = slot_id

        # Step 4: Pull remote blocks
        print(f"[pipeline] Request {req.request_id}: pulling to slot {slot_id}")
        handle = self._migration_workers.pull(slot_id, remote_blocks)
        req.pull_handle = handle

        # Check if instant completion (mock)
        if handle.status == PullStatus.COMPLETED:
            req.matched_tokens = len(remote_blocks) * 64  # estimate
            req.is_continuation = True
            self._dispatch(req, reason="remote blocks arrived (instant)")
            return

        # Park the request
        with self._lock:
            req.state = ResolveState.WAITING_FOR_REMOTE_BLOCKS

        print(f"[pipeline] Request {req.request_id}: PARKED waiting for remote blocks")

    def _allocate_and_dispatch(self, req: ResolveRequest, reason: str):
        """Allocate a fresh slot and dispatch."""
        slot_id = self._session_manager.allocate_slot(req.request_id)
        if slot_id is None:
            print(f"[pipeline] Request {req.request_id}: no free slots!")
            req.state = ResolveState.ABORTED
            return

        req.allocated_slot = slot_id
        req.is_continuation = False
        self._dispatch(req, reason=reason)

    def _dispatch(self, req: ResolveRequest, reason: str):
        """Mark request as dispatched."""
        with self._lock:
            if req.state in (ResolveState.DISPATCHED, ResolveState.ABORTED):
                return

            req.state = ResolveState.DISPATCHED
            req.dispatch_time = time.monotonic()

        elapsed = (req.dispatch_time - req.submit_time) * 1000
        cont_str = "continuation" if req.is_continuation else "fresh prefill"
        print(
            f"[pipeline] DISPATCHED {req.request_id} ({reason}) [{cont_str}] [{elapsed:.1f}ms]"
        )

        if self._on_dispatch:
            self._on_dispatch(req)

    def drain(self) -> int:
        """
        Poll for completed transfers and resume waiting requests.

        Called by background drain thread.
        """
        arrived = self._migration_workers.check_arrived_blocks()
        resumed = 0

        if arrived:
            print(f"[pipeline] drain(): {len(arrived)} transfers completed")

        with self._lock:
            for handle in arrived:
                for req in self._requests.values():
                    if req.state != ResolveState.WAITING_FOR_REMOTE_BLOCKS:
                        continue
                    if req.pull_handle and req.pull_handle.id == handle.id:
                        if handle.status == PullStatus.COMPLETED:
                            req.matched_tokens = len(req.remote_blocks) * 64
                            req.is_continuation = True
                            self._dispatch_locked(req, reason="remote blocks arrived")
                        else:
                            # Failed — fallback to fresh prefill
                            self._dispatch_locked(
                                req, reason=f"pull failed: {handle.error}"
                            )
                        resumed += 1

            # Check timeouts
            now = time.monotonic()
            for req in list(self._requests.values()):
                if req.state == ResolveState.WAITING_FOR_REMOTE_BLOCKS:
                    elapsed = now - req.submit_time
                    if elapsed > self._timeout_seconds:
                        print(
                            f"[pipeline] Request {req.request_id}: timeout after {elapsed:.1f}s"
                        )
                        self._dispatch_locked(req, reason="timeout, fallback prefill")
                        resumed += 1

        return resumed

    def _dispatch_locked(self, req: ResolveRequest, reason: str):
        """Dispatch while holding lock."""
        if req.state in (ResolveState.DISPATCHED, ResolveState.ABORTED):
            return

        req.state = ResolveState.DISPATCHED
        req.dispatch_time = time.monotonic()

        elapsed = (req.dispatch_time - req.submit_time) * 1000
        cont_str = "continuation" if req.is_continuation else "fresh prefill"
        print(
            f"[pipeline] DISPATCHED {req.request_id} ({reason}) [{cont_str}] [{elapsed:.1f}ms]"
        )

        if self._on_dispatch:
            self._on_dispatch(req)

    def cancel(self, request_id: str) -> bool:
        """Cancel a request."""
        with self._lock:
            req = self._requests.get(request_id)
            if not req:
                return False
            if req.state == ResolveState.DISPATCHED:
                return False
            req.state = ResolveState.ABORTED
            return True

    def publish_local_blocks(self, session_id: str, slot_id: int):
        """
        Publish local KV blocks to the remote index (Mooncake).

        Called after prefill completes to advertise blocks to other endpoints.

        In real C++, this would be called from the completion handler:
            remoteKvIndex_->publish(hash, endpointId_, slotId, blockIndex);
        """
        hashes = self._session_manager.get_slot_hashes(slot_id)
        for i, h in enumerate(hashes):
            self._remote_kv_index.publish(h, self._endpoint_id, slot_id, i)

        print(f"[pipeline] Published {len(hashes)} blocks for session {session_id}")

    def start_drain_thread(self, interval_seconds: float = 0.01):
        """Start background drain thread."""
        if self._drain_running:
            return

        self._drain_running = True
        self._drain_thread = threading.Thread(
            target=self._drain_loop, args=(interval_seconds,), daemon=True
        )
        self._drain_thread.start()
        print("[pipeline] Drain thread started")

    def stop_drain_thread(self):
        """Stop background drain thread."""
        self._drain_running = False
        if self._drain_thread:
            self._drain_thread.join(timeout=1.0)
            self._drain_thread = None
        print("[pipeline] Drain thread stopped")

    def _drain_loop(self, interval: float):
        """Background drain loop."""
        while self._drain_running:
            self.drain()
            time.sleep(interval)
