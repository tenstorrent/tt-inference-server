# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Prefill-only scheduler: selects waiting requests that fit in batch and KV blocks.

Inspired by nano-vllm Scheduler; only the prefill branch. No decode, no running
queue for generation. Sequences are added to waiting; schedule() returns a batch
of sequences to prefill (and allocates their blocks). After prefill they are
deallocated and handed off (e.g. to decode node).

num_kvcache_blocks can be set explicitly or computed from available KV cache
memory (num_layers, block_size, kvpe_dim, dtype, etc.).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

from block_manager import BlockManager
from sequence import PrefillSequence
from timing import timed

logger = logging.getLogger(__name__)


def compute_num_kvcache_blocks(
    available_memory_bytes: int,
    num_layers: int,
    block_size: int,
    kvpe_dim: int,
    dtype_bytes: int = 1,
    tensors_per_layer: int = 1,
) -> int:
    """
    Compute max number of KV cache blocks that fit in available memory.

    Memory per block (across all layers): num_layers * tensors_per_layer
    * block_size * kvpe_dim * dtype_bytes.
    tensors_per_layer=1 for DeepSeek (combined KV/MLA); use 2 for separate K and V.
    dtype_bytes=1 for bfloat8 (DeepSeek).
    """
    bytes_per_block = (
        num_layers * tensors_per_layer * block_size * kvpe_dim * dtype_bytes
    )
    if bytes_per_block <= 0:
        return 0
    return max(0, available_memory_bytes // bytes_per_block)


@dataclass
class SchedulerConfig:
    """Configuration for the prefill scheduler."""

    max_num_seqs: int = 32
    max_num_batched_tokens: int = 8192
    num_kvcache_blocks: int | None = None
    block_size: int = 32

    # KV cache memory (used when num_kvcache_blocks is None)
    available_kv_cache_memory_gb: float = 2.0
    num_layers: int = 61
    kvpe_dim: int = 576
    kv_cache_dtype_bytes: int = 1  # bfloat8 for DeepSeek; use 2 for bfloat16
    kv_tensors_per_layer: int = (
        1  # 1 for DeepSeek (combined KV/MLA); 2 for separate K and V
    )

    def get_num_kvcache_blocks(self) -> int:
        if self.num_kvcache_blocks is not None:
            return self.num_kvcache_blocks
        available_bytes = int(self.available_kv_cache_memory_gb * (1024**3))
        num_blocks = compute_num_kvcache_blocks(
            available_memory_bytes=available_bytes,
            num_layers=self.num_layers,
            block_size=self.block_size,
            kvpe_dim=self.kvpe_dim,
            dtype_bytes=self.kv_cache_dtype_bytes,
            tensors_per_layer=self.kv_tensors_per_layer,
        )
        logger.info(
            "kv_cache memory calc available_gb=%.2f num_layers=%d -> num_blocks=%d",
            self.available_kv_cache_memory_gb,
            self.num_layers,
            num_blocks,
        )
        return num_blocks


class PrefillScheduler:
    """
    Scheduler that only handles prefills. Maintains a waiting queue; schedule()
    returns a batch of sequences to prefill, respecting max_num_seqs,
    max_num_batched_tokens, and block availability.
    """

    def __init__(self, config: SchedulerConfig):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        num_blocks = config.get_num_kvcache_blocks()
        logger.info(
            "block_manager num_blocks=%d (from %s)",
            num_blocks,
            "config" if config.num_kvcache_blocks is not None else "memory calc",
        )
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=config.block_size,
        )
        self.waiting: deque[PrefillSequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting

    def add(self, seq: PrefillSequence) -> None:
        self.waiting.append(seq)
        logger.info(
            "request added req_id=%s prompt_len=%d waiting=%d",
            seq.req_id,
            len(seq),
            len(self.waiting),
        )

    @timed()
    def schedule(self) -> list[PrefillSequence]:
        """
        Select a batch of sequences from waiting to prefill. Allocates blocks for
        each. Returns the list of scheduled sequences (may be empty if waiting is
        empty or no sequence fits).
        """
        scheduled: list[PrefillSequence] = []
        num_batched_tokens = 0

        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            prompt_tokens = len(seq) - seq.num_cached_tokens
            if num_batched_tokens + prompt_tokens > self.max_num_batched_tokens:
                logger.debug(
                    "batch token limit seq req_id=%s prompt_tokens=%d would exceed max=%d",
                    seq.req_id,
                    prompt_tokens,
                    self.max_num_batched_tokens,
                )
                break
            if not self.block_manager.can_allocate(seq):
                logger.debug(
                    "not enough blocks for req_id=%s num_blocks=%d",
                    seq.req_id,
                    seq.num_blocks,
                )
                break
            self.block_manager.allocate(seq)
            num_batched_tokens += prompt_tokens
            self.waiting.popleft()
            scheduled.append(seq)

        if scheduled:
            logger.info(
                "scheduled batch num_seqs=%d num_batched_tokens=%d req_ids=%s waiting_left=%d",
                len(scheduled),
                num_batched_tokens,
                [s.req_id for s in scheduled],
                len(self.waiting),
            )
        elif self.waiting:
            logger.debug(
                "schedule produced empty batch (limits hit), waiting=%d",
                len(self.waiting),
            )

        return scheduled

    @timed()
    def release(self, seqs: list[PrefillSequence]) -> None:
        """
        Release blocks for the given sequences (e.g. after prefill is done and
        KV cache has been streamed to decode node).
        """
        if seqs:
            logger.info(
                "releasing seqs num=%d req_ids=%s", len(seqs), [s.req_id for s in seqs]
            )
        for seq in seqs:
            self.block_manager.deallocate(seq)
