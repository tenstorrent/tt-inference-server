# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
session_manager.py — Python adapter mirroring the real SessionManager API.

This mirrors the C++ SessionManager from:
  tt-inference-server/tt-media-server/cpp_server/include/services/session_manager.hpp

Key methods we replicate:
- tryAcquireByPrefixHash(blocks) → AcquiredSession or None
- registerPrefixHash(sessionId, hashes)
- releaseSession(sessionId)

The real C++ uses:
- prefixIndex_: hash → {sessionId, slotId, blockIndex}
- inFlightAcquires_: for parking concurrent requests on same prefix

For the PoC we implement the core lookup logic without the concurrency handling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import threading


@dataclass
class BlockHashInfo:
    """Mirrors utils/conversation_hasher.hpp BlockHashInfo."""

    hash: int
    accumulated_think_tokens: int = 0


@dataclass
class AcquiredSession:
    """
    Mirrors session_manager.hpp AcquiredSession.

    Returned when tryAcquireByPrefixHash finds a local match.
    """

    session_id: str
    slot_id: int
    matched_blocks: int  # how many prefix blocks matched
    matched_tokens: int  # token count of matched prefix


@dataclass
class SlotInfo:
    """Info about a slot's KV cache state."""

    slot_id: int
    session_id: str
    block_hashes: List[int] = field(default_factory=list)
    token_count: int = 0


class SessionManager:
    """
    Python adapter mirroring the real SessionManager.

    The real C++ SessionManager (session_manager.hpp:54) does:
    1. Manages slots (acquire/release)
    2. Maintains prefixIndex_ for content-addressable lookup
    3. Handles in-flight acquire parking

    For the PoC we focus on:
    - tryAcquireByPrefixHash: local prefix cache lookup
    - registerPrefixHash: advertise a session's prefix
    - The data structures that would need remote lookup extension
    """

    def __init__(self, num_slots: int = 8):
        self._num_slots = num_slots
        self._lock = threading.Lock()

        # prefixIndex_: hash → (sessionId, slotId, blockIndex)
        # Real C++ uses std::unordered_map<uint64_t, PrefixEntry>
        self._prefix_index: Dict[int, tuple[str, int, int]] = {}

        # Slot state
        self._slots: Dict[int, Optional[SlotInfo]] = {i: None for i in range(num_slots)}
        self._free_slots: Set[int] = set(range(num_slots))

        # Session → slot mapping
        self._session_to_slot: Dict[str, int] = {}

    def try_acquire_by_prefix_hash(
        self,
        blocks: List[BlockHashInfo],
    ) -> Optional[AcquiredSession]:
        """
        Mirrors SessionManager::tryAcquireByPrefixHash (session_manager.hpp:121).

        Looks up blocks in the LOCAL prefix index. Returns the longest
        matching prefix if found.

        This is where remote lookup would be added — after this returns None.

        Args:
            blocks: Per-block hash info from computePrefixCachingInfoFromTokens

        Returns:
            AcquiredSession if local match found, None otherwise
        """
        if not blocks:
            return None

        with self._lock:
            # Find longest matching prefix
            best_match: Optional[tuple[str, int, int]] = (
                None  # (session_id, slot_id, matched_blocks)
            )

            for i, block_info in enumerate(blocks):
                entry = self._prefix_index.get(block_info.hash)
                if entry is None:
                    break

                session_id, slot_id, block_index = entry

                # Verify the slot still has this session
                slot_info = self._slots.get(slot_id)
                if slot_info is None or slot_info.session_id != session_id:
                    # Stale entry
                    continue

                # Check block index matches position
                if block_index != i:
                    continue

                best_match = (session_id, slot_id, i + 1)

            if best_match is None:
                return None

            session_id, slot_id, matched_blocks = best_match
            slot_info = self._slots[slot_id]

            # Estimate matched tokens (simplified: block_size * matched_blocks)
            # Real impl uses slot_info's actual token count
            matched_tokens = matched_blocks * 64  # assume 64 tokens per block

            return AcquiredSession(
                session_id=session_id,
                slot_id=slot_id,
                matched_blocks=matched_blocks,
                matched_tokens=matched_tokens,
            )

    def register_prefix_hash(
        self,
        session_id: str,
        slot_id: int,
        hashes: List[int],
    ) -> None:
        """
        Mirrors SessionManager::registerPrefixHash (session_manager.hpp:134).

        Called after prefill completes to advertise the new KV cache content.

        Args:
            session_id: The session that owns this prefix
            slot_id: The slot containing the KV cache
            hashes: Per-block hashes to register
        """
        with self._lock:
            for i, h in enumerate(hashes):
                self._prefix_index[h] = (session_id, slot_id, i)

            # Update slot info
            if slot_id in self._slots:
                self._slots[slot_id] = SlotInfo(
                    slot_id=slot_id,
                    session_id=session_id,
                    block_hashes=list(hashes),
                    token_count=len(hashes) * 64,
                )

            self._session_to_slot[session_id] = slot_id

    def allocate_slot(self, session_id: str) -> Optional[int]:
        """
        Allocate a free slot for a new session.

        Returns:
            slot_id if available, None if all slots full
        """
        with self._lock:
            if not self._free_slots:
                return None

            slot_id = self._free_slots.pop()
            self._slots[slot_id] = SlotInfo(slot_id=slot_id, session_id=session_id)
            self._session_to_slot[session_id] = slot_id
            return slot_id

    def release_session(self, session_id: str) -> bool:
        """
        Release a session and its slot.

        Returns:
            True if released, False if not found
        """
        with self._lock:
            slot_id = self._session_to_slot.get(session_id)
            if slot_id is None:
                return False

            # Remove from prefix index
            slot_info = self._slots.get(slot_id)
            if slot_info:
                for h in slot_info.block_hashes:
                    self._prefix_index.pop(h, None)

            # Free the slot
            self._slots[slot_id] = None
            self._free_slots.add(slot_id)
            del self._session_to_slot[session_id]

            return True

    def get_slot_hashes(self, slot_id: int) -> List[int]:
        """Get the block hashes for a slot (for publishing to remote index)."""
        with self._lock:
            slot_info = self._slots.get(slot_id)
            if slot_info:
                return list(slot_info.block_hashes)
            return []

    def get_all_hashes(self) -> Dict[int, tuple[str, int, int]]:
        """Get entire prefix index (for debugging)."""
        with self._lock:
            return dict(self._prefix_index)
