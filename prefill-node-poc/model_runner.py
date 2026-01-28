# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Model runner that bridges the prefill scheduler and the prefill simulator.

Builds batched inputs (tokens, prompt_lens, page_table, kv_cache, start_pos)
from scheduled sequences and calls the prefill simulator. Matches the interface
expected by tt_model_runner (tokens, prompt_lens, page_table, kv_cache, start_pos).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from prefill_simulator import (
    DeepSeekPrefillSimulator,
    KVCacheReference,
    PrefillConfig,
)
from sequence import PrefillSequence


class PrefillModelRunner:
    """
    Runs prefill for batches of sequences produced by PrefillScheduler.
    Prepares inputs from sequences (tokens, prompt_lens, block_tables) and
    calls the prefill simulator.
    """

    def __init__(
        self,
        prefill_config: PrefillConfig,
        mesh_device: object = None,
        on_kv_cache_ready: Optional[Callable[[int, KVCacheReference], None]] = None,
    ):
        self.prefill_config = prefill_config
        self.simulator = DeepSeekPrefillSimulator(
            config=prefill_config,
            mesh_device=mesh_device,
            on_kv_cache_ready=on_kv_cache_ready,
        )
        self._kv_caches: Optional[List[KVCacheReference]] = None

    def allocate_kv_cache(self, dtype: str = "bfloat8_b") -> List[KVCacheReference]:
        """Allocate KV cache on device; call once before running prefills."""
        self._kv_caches = self.simulator.allocate_kv_cache(dtype=dtype)
        return self._kv_caches

    def prepare_prefill(
        self,
        seqs: List[PrefillSequence],
    ) -> tuple[torch.Tensor, List[int], torch.Tensor, List[object], torch.Tensor]:
        """
        Build simulator inputs from scheduled sequences.

        Returns:
            tokens: [batch_size, max_prompt_len], padded with pad_token_id
            prompt_lens: list of actual lengths per sequence
            page_table: [batch_size, max_num_blocks_per_req], block IDs per sequence
            kv_cache: list of KV cache tensors (from simulator)
            start_pos: [batch_size], zeros (no prefix cache)
        """
        if not seqs:
            raise ValueError("prepare_prefill requires at least one sequence")

        batch_size = len(seqs)
        prompt_lens = [len(seq) for seq in seqs]
        max_prompt_len = max(prompt_lens)
        max_num_blocks = max(len(seq.block_table) for seq in seqs)
        pad_token_id = self.prefill_config.pad_token_id

        tokens = torch.full(
            (batch_size, max_prompt_len),
            pad_token_id,
            dtype=torch.int64,
        )
        for i, seq in enumerate(seqs):
            tokens[i, : len(seq.token_ids)] = torch.tensor(
                seq.token_ids,
                dtype=torch.int64,
            )

        page_table = torch.zeros(
            (batch_size, max_num_blocks),
            dtype=torch.int32,
        )
        for i, seq in enumerate(seqs):
            if seq.block_table:
                page_table[i, : len(seq.block_table)] = torch.tensor(
                    seq.block_table,
                    dtype=torch.int32,
                )

        if self._kv_caches is None:
            raise RuntimeError("KV cache not allocated; call allocate_kv_cache() first")
        kv_cache = [ref.tensor for ref in self._kv_caches]

        start_pos = torch.zeros(batch_size, dtype=torch.int32)

        return tokens, prompt_lens, page_table, kv_cache, start_pos

    def run_prefill(self, seqs: List[PrefillSequence]) -> torch.Tensor:
        """
        Run prefill for the given scheduled sequences. Prepares inputs and
        calls the simulator. Returns logits [batch_size, max_padded_len, vocab_size].
        """
        if not seqs:
            return torch.empty(0, 0, self.prefill_config.vocab_size)

        tokens, prompt_lens, page_table, kv_cache, start_pos = self.prepare_prefill(
            seqs
        )
        logits = self.simulator.prefill_forward(
            tokens=tokens,
            prompt_lens=prompt_lens,
            page_table=page_table,
            kv_cache=kv_cache,
            start_pos=start_pos,
        )
        return logits

    def cleanup(self) -> None:
        self.simulator.cleanup()
        self._kv_caches = None
