# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Model runner that bridges the prefill scheduler and the prefill simulator.

Builds batched inputs (tokens, prompt_lens, page_table, kv_cache, start_pos)
from scheduled sequences and calls the prefill simulator. Matches the interface
expected by tt_model_runner (tokens, prompt_lens, page_table, kv_cache, start_pos).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

import torch

logger = logging.getLogger(__name__)

from prefill_simulator import (
    DeepSeekPrefillSimulator,
    KVCacheReference,
    PrefillConfig,
)
from sequence import PrefillSequence
from timing import timed


def extract_blocks_from_layer(
    layer_tensor: torch.Tensor,
    block_table: List[int],
    block_size: int,
) -> List[Any]:
    """
    Extract KV blocks for a single request from a layer tensor.

    Layer tensor shape: (num_blocks, num_heads, block_size, kvpe_dim).
    block_table lists physical block IDs for this request.
    Returns a list of block tensors (views or copies depending on backend).
    """
    if not block_table:
        return []
    blocks = []
    for block_id in block_table:
        block = layer_tensor[block_id]
        blocks.append(block)
    return blocks


class PrefillModelRunner:
    """
    Runs prefill for batches of sequences produced by PrefillScheduler.
    Prepares inputs from sequences (tokens, prompt_lens, block_tables) and
    calls the prefill simulator. On each layer-ready callback, extracts blocks
    per request and invokes on_kv_cache_blocks_ready(layer_idx, req_id, blocks).
    """

    def __init__(
        self,
        prefill_config: PrefillConfig,
        mesh_device: object = None,
        on_kv_cache_ready: Optional[Callable[[int, KVCacheReference], None]] = None,
        on_kv_cache_blocks_ready: Optional[
            Callable[[int, Any, List[Any]], None]
        ] = None,
    ):
        self.prefill_config = prefill_config
        self._current_seqs: List[PrefillSequence] = []
        self._on_kv_cache_blocks_ready = on_kv_cache_blocks_ready
        self._on_kv_cache_ready_user = on_kv_cache_ready

        def _callback(layer_idx: int, ref: KVCacheReference) -> None:
            if self._on_kv_cache_blocks_ready is not None:
                self._on_kv_cache_ready_internal(layer_idx, ref)
            if self._on_kv_cache_ready_user is not None:
                self._on_kv_cache_ready_user(layer_idx, ref)

        self.simulator = DeepSeekPrefillSimulator(
            config=prefill_config,
            mesh_device=mesh_device,
            on_kv_cache_ready=_callback
            if (on_kv_cache_blocks_ready or on_kv_cache_ready)
            else None,
        )
        self._kv_caches: Optional[List[KVCacheReference]] = None

    def _on_kv_cache_ready_internal(
        self, layer_idx: int, ref: KVCacheReference
    ) -> None:
        if (
            self._on_kv_cache_blocks_ready is None
            or not self._current_seqs
            or self._kv_caches is None
        ):
            return
        layer_tensor = ref.tensor
        if not isinstance(layer_tensor, torch.Tensor):
            return
        block_size = self.prefill_config.block_size
        for seq in self._current_seqs:
            blocks = extract_blocks_from_layer(
                layer_tensor, seq.block_table, block_size
            )
            self._on_kv_cache_blocks_ready(layer_idx, seq.req_id, blocks)

    def allocate_kv_cache(self, dtype: str = "bfloat8_b") -> List[KVCacheReference]:
        """Allocate KV cache on device; call once before running prefills."""
        self._kv_caches = self.simulator.allocate_kv_cache(dtype=dtype)
        logger.info(
            "kv_cache allocated num_layers=%d dtype=%s", len(self._kv_caches), dtype
        )
        return self._kv_caches

    @timed()
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

        logger.debug(
            "prepare_prefill batch_size=%d max_prompt_len=%d max_num_blocks=%d prompt_lens=%s",
            batch_size,
            max_prompt_len,
            max_num_blocks,
            prompt_lens,
        )
        return tokens, prompt_lens, page_table, kv_cache, start_pos

    @timed()
    def run_prefill(self, seqs: List[PrefillSequence]) -> torch.Tensor:
        """
        Run prefill for the given scheduled sequences. Prepares inputs and
        calls the simulator. Returns logits [batch_size, max_padded_len, vocab_size].
        Holds _current_seqs for the duration so layer-ready callbacks can extract
        blocks per request.
        """
        if not seqs:
            return torch.empty(0, 0, self.prefill_config.vocab_size)

        logger.info(
            "run_prefill start num_seqs=%d req_ids=%s",
            len(seqs),
            [s.req_id for s in seqs],
        )
        self._current_seqs = seqs
        try:
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
            logger.info(
                "run_prefill done logits_shape=%s",
                tuple(logits.shape),
            )
            return logits
        finally:
            self._current_seqs = []

    def cleanup(self) -> None:
        self.simulator.cleanup()
        self._kv_caches = None
        logger.info("model_runner cleanup")
