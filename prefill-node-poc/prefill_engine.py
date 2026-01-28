# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Prefill-only engine: scheduler + model runner. Add requests, call step() to
schedule and run prefill, then release sequences (e.g. for D2D handoff to decode).
run_loop(stop_event) runs until stop_event is set, draining the waiting queue.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import torch

from model_runner import PrefillModelRunner
from prefill_simulator import KVCacheReference, PrefillConfig
from scheduler import PrefillScheduler, SchedulerConfig
from sequence import PrefillSequence

logger = logging.getLogger(__name__)


class PrefillEngine:
    """
    Prefill-only engine. Add requests via add_request(); call step() to
    schedule a batch, run prefill, and return logits + scheduled sequences.
    Call release_after_prefill(seqs) after KV cache has been streamed to decode.
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        prefill_config: PrefillConfig,
        mesh_device: object = None,
        on_kv_cache_ready: Optional[Callable[[int, KVCacheReference], None]] = None,
        on_kv_cache_blocks_ready: Optional[Callable[[int, object, list], None]] = None,
    ):
        self.scheduler = PrefillScheduler(scheduler_config)
        self.model_runner = PrefillModelRunner(
            prefill_config=prefill_config,
            mesh_device=mesh_device,
            on_kv_cache_ready=on_kv_cache_ready,
            on_kv_cache_blocks_ready=on_kv_cache_blocks_ready,
        )
        self.model_runner.allocate_kv_cache()
        logger.info(
            "engine init max_num_seqs=%d max_num_batched_tokens=%d num_kvcache_blocks=%d num_layers=%d",
            scheduler_config.max_num_seqs,
            scheduler_config.max_num_batched_tokens,
            scheduler_config.num_kvcache_blocks,
            prefill_config.num_layers,
        )

    def add_request(self, token_ids: list[int], req_id: str | None = None) -> None:
        """Add a prefill request (prompt token IDs) to the waiting queue."""
        block_size = self.scheduler.block_manager.block_size
        seq = PrefillSequence(token_ids=token_ids, req_id=req_id, block_size=block_size)
        self.scheduler.add(seq)
        logger.info("add_request req_id=%s prompt_len=%d", seq.req_id, len(seq))

    def step(self) -> tuple[torch.Tensor, list[PrefillSequence]]:
        """
        Schedule a batch of waiting sequences, run prefill, return logits and
        the scheduled sequences. Caller should stream KV cache (via callback)
        then call release_after_prefill(seqs).
        """
        scheduled = self.scheduler.schedule()
        if not scheduled:
            logger.debug("step no batch scheduled (waiting empty or limits hit)")
            return torch.empty(0, 0, self.model_runner.prefill_config.vocab_size), []
        logits = self.model_runner.run_prefill(scheduled)
        logger.info(
            "step done num_seqs=%d logits_shape=%s req_ids=%s",
            len(scheduled),
            tuple(logits.shape),
            [s.req_id for s in scheduled],
        )
        return logits, scheduled

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
