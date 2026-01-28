# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Prefill-only scheduler: selects waiting requests that fit in batch and KV blocks.

Inspired by nano-vllm Scheduler; only the prefill branch. No decode, no running
queue for generation. Sequences are added to waiting; schedule() returns a batch
of sequences to prefill (and allocates their blocks). After prefill they are
deallocated and handed off (e.g. to decode node).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from block_manager import BlockManager
from sequence import PrefillSequence


@dataclass
class SchedulerConfig:
    """Configuration for the prefill scheduler."""

    max_num_seqs: int = 32
    max_num_batched_tokens: int = 8192
    num_kvcache_blocks: int = 1024
    block_size: int = 32


class PrefillScheduler:
    """
    Scheduler that only handles prefills. Maintains a waiting queue; schedule()
    returns a batch of sequences to prefill, respecting max_num_seqs,
    max_num_batched_tokens, and block availability.
    """

    def __init__(self, config: SchedulerConfig):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_manager = BlockManager(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.block_size,
        )
        self.waiting: deque[PrefillSequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting

    def add(self, seq: PrefillSequence) -> None:
        self.waiting.append(seq)

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
                break
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            num_batched_tokens += prompt_tokens
            self.waiting.popleft()
            scheduled.append(seq)

        return scheduled

    def release(self, seqs: list[PrefillSequence]) -> None:
        """
        Release blocks for the given sequences (e.g. after prefill is done and
        KV cache has been streamed to decode node).
        """
        for seq in seqs:
            self.block_manager.deallocate(seq)
