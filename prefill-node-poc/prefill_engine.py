# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Prefill-only engine: scheduler + model runner. Add requests, call step() to
schedule and run prefill, then release sequences (e.g. for D2D handoff to decode).
run_loop(stop_event) runs until stop_event is set, draining the waiting queue.

Works like vLLM interface with explicit KV cache allocation and access.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

import torch

import ttnn

from model_runner import KvCacheConfig, ModelConfig, PrefillModelRunner
from scheduler import PrefillScheduler, SchedulerConfig
from sequence import PrefillSequence
from timing import timed

logger = logging.getLogger(__name__)


class PrefillEngine:
    """
    Prefill-only engine. Add requests via add_request(); call step() to
    schedule a batch, run prefill, and return logits + scheduled sequences.
    Call release_after_prefill(seqs) after KV cache has been streamed to decode.
    
    Works like vLLM interface:
    - Explicit KV cache allocation with configurable shape
    - Access to KV cache tensors after prefill (for D2D transfer)
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        model_config: ModelConfig,
        mesh_device: ttnn.MeshDevice,
        kv_cache_config: Optional[KvCacheConfig] = None,
    ):
        self.scheduler = PrefillScheduler(scheduler_config)
        self.model_config = model_config
        self.kv_cache_config = kv_cache_config
        
        self.model_runner = PrefillModelRunner(
            model_config=model_config,
            mesh_device=mesh_device,
            kv_cache_config=kv_cache_config,
        )
        
        # Allocate KV cache (with optional explicit config)
        self.model_runner.allocate_kv_cache(kv_cache_config)
        
        num_blocks = scheduler_config.get_num_kvcache_blocks()
        max_seqs = scheduler_config.max_num_seqs or 0
        max_tokens = scheduler_config.max_num_batched_tokens or 0
        logger.info(
            "engine init max_num_seqs=%d max_num_batched_tokens=%d num_kvcache_blocks=%d",
            max_seqs,
            max_tokens,
            num_blocks,
        )

    @property
    def tokenizer(self):
        """Access the tokenizer."""
        return self.model_runner.tokenizer
    
    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        return self.model_runner.num_layers
    
    @property
    def kv_cache(self) -> Optional[List[ttnn.Tensor]]:
        """Access the KV cache tensors (one per layer)."""
        return self.model_runner.kv_cache
    
    def get_kv_cache(self) -> List[ttnn.Tensor]:
        """
        Get the KV cache tensors (like vLLM interface).
        
        Returns:
            List of KV cache tensors, one per layer.
        """
        return self.model_runner.get_kv_cache()
    
    def get_kv_cache_for_layer(self, layer_idx: int) -> ttnn.Tensor:
        """
        Get KV cache tensor for a specific layer.
        
        Args:
            layer_idx: Layer index (0 to num_layers-1)
            
        Returns:
            KV cache tensor for the specified layer.
        """
        return self.model_runner.get_kv_cache_for_layer(layer_idx)

    def add_request(self, token_ids: list[int], req_id: str | None = None) -> None:
        """Add a prefill request (prompt token IDs) to the waiting queue."""
        block_size = self.scheduler.block_manager.block_size
        seq = PrefillSequence(token_ids=token_ids, req_id=req_id, block_size=block_size)
        self.scheduler.add(seq)
        logger.info("add_request req_id=%s prompt_len=%d", seq.req_id, len(seq))

    @timed()
    def step(self) -> tuple[torch.Tensor, list[PrefillSequence]]:
        """
        Schedule a batch of waiting sequences, run prefill, return logits and
        the scheduled sequences.
        
        After this call, KV cache is updated and accessible via get_kv_cache().
        Caller should stream KV cache then call release_after_prefill(seqs).
        """
        scheduled = self.scheduler.schedule()
        if not scheduled:
            logger.debug("step no batch scheduled (waiting empty or limits hit)")
            return torch.empty(0, 0, self.model_config.vocab_size), []
        logits = self.model_runner.run_prefill(scheduled)
        logger.info(
            "step done num_seqs=%d logits_shape=%s req_ids=%s",
            len(scheduled),
            tuple(logits.shape),
            [s.req_id for s in scheduled],
        )
        return logits, scheduled

    @timed()
    def release_after_prefill(self, seqs: list[PrefillSequence]) -> None:
        """Release block allocations for sequences after prefill/KV stream is done."""
        if seqs:
            logger.info(
                "release_after_prefill num_seqs=%d req_ids=%s",
                len(seqs),
                [s.req_id for s in seqs],
            )
        self.scheduler.release(seqs)

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def run_loop(self, stop_event: threading.Event) -> None:
        """
        Run step() in a loop until stop_event is set. Drains the waiting queue.
        When the queue is empty, waits on stop_event (timeout 0.05s) to avoid spinning.
        """
        while not stop_event.is_set():
            if self.is_finished():
                stop_event.wait(timeout=0.05)
                continue
            logits, scheduled = self.step()
            if not scheduled:
                continue
            self.release_after_prefill(scheduled)

    def cleanup(self) -> None:
        self.model_runner.cleanup()
        logger.info("engine cleanup")
