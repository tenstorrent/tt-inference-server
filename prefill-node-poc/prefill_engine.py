# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
from typing import List

import torch
import ttnn

from model_runner import KvCacheConfig, ModelConfig, PrefillModelRunner
from scheduler import PrefillScheduler, SchedulerConfig
from sequence import PrefillSequence
from timing import timed

logger = logging.getLogger(__name__)


class PrefillEngine:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        model_config: ModelConfig,
        mesh_device: ttnn.MeshDevice,
        kv_cache_config: KvCacheConfig,
    ):
        self.scheduler = PrefillScheduler(scheduler_config)
        self.model_config = model_config
        self.kv_cache_config = kv_cache_config
        
        self.model_runner = PrefillModelRunner(
            model_config=model_config,
            mesh_device=mesh_device,
        )
        
        self.model_runner.allocate_kv_cache(kv_cache_config)
        
        num_blocks = scheduler_config.get_num_kvcache_blocks()
        logger.info(
            "engine init max_num_seqs=%d max_num_batched_tokens=%d num_kvcache_blocks=%d",
            scheduler_config.max_num_seqs or 0,
            scheduler_config.max_num_batched_tokens or 0,
            num_blocks,
        )

    @property
    def tokenizer(self):
        return self.model_runner.tokenizer
    
    @property
    def num_layers(self) -> int:
        return self.model_runner.num_layers
    
    @property
    def kv_cache(self) -> List[ttnn.Tensor]:
        return self.model_runner.kv_cache
    
    def get_kv_cache(self) -> List[ttnn.Tensor]:
        return self.model_runner.get_kv_cache()
    
    def get_kv_cache_for_layer(self, layer_idx: int) -> ttnn.Tensor:
        return self.model_runner.get_kv_cache_for_layer(layer_idx)

    def add_request(self, token_ids: list[int], req_id: str | None = None) -> None:
        block_size = self.scheduler.block_manager.block_size
        seq = PrefillSequence(token_ids=token_ids, req_id=req_id, block_size=block_size)
        self.scheduler.add(seq)
        logger.info("add_request req_id=%s prompt_len=%d", seq.req_id, len(seq))

    @timed()
    def step(self) -> tuple[torch.Tensor, list[PrefillSequence]]:
        scheduled = self.scheduler.schedule()
        if not scheduled:
            logger.debug("step no batch scheduled")
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
