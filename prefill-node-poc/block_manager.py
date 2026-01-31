# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Block manager for paged KV cache: allocates and frees blocks per prefill sequence.

Inspired by nano-vllm BlockManager; simplified for prefill-only (no prefix caching
hash, no can_append/may_append for decode). Each sequence gets a contiguous set of
free block IDs for its prompt length.
"""

from __future__ import annotations

import logging
from collections import deque

from sequence import PrefillSequence

logger = logging.getLogger(__name__)


class BlockManager:
    """
    Manages a pool of KV cache blocks. Allocates blocks to sequences for prefill
    and frees them when sequences are done (e.g. handed off to decode node).
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def can_allocate(self, seq: PrefillSequence) -> bool:
        """True if there are enough free blocks for this sequence."""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: PrefillSequence) -> None:
        """
        Assign block IDs to the sequence. seq.block_table is filled; seq must not
        already have blocks.
        """
        assert not seq.block_table, "Sequence already has blocks"
        assert self.can_allocate(seq), "Not enough free blocks"
        num_blocks = seq.num_blocks
        for _ in range(num_blocks):
            block_id = self.free_block_ids.popleft()
            self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        free_after = len(self.free_block_ids)
        logger.info(
            "allocated blocks req_id=%s num_blocks=%d block_ids=%s free_after=%d",
            getattr(seq, "req_id", "?"),
            num_blocks,
            seq.block_table[:5] if len(seq.block_table) > 5 else seq.block_table,
            free_after,
        )

    def deallocate(self, seq: PrefillSequence) -> None:
        """Return the sequence's blocks to the free pool and clear seq.block_table."""
        num_blocks = len(seq.block_table)
        for block_id in seq.block_table:
            self.used_block_ids.discard(block_id)
            self.free_block_ids.append(block_id)
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        free_after = len(self.free_block_ids)
        logger.info(
            "deallocated blocks req_id=%s num_blocks=%d free_after=%d",
            getattr(seq, "req_id", "?"),
            num_blocks,
            free_after,
        )
