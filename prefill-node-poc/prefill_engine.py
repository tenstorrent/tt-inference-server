# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Prefill-only engine: scheduler + model runner. Add requests, call step() to
schedule and run prefill, then release sequences (e.g. for D2D handoff to decode).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from model_runner import PrefillModelRunner
from prefill_simulator import KVCacheReference, PrefillConfig
from scheduler import PrefillScheduler, SchedulerConfig
from sequence import PrefillSequence


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

    def add_request(self, token_ids: list[int], req_id: str | None = None) -> None:
        """Add a prefill request (prompt token IDs) to the waiting queue."""
        block_size = self.scheduler.block_manager.block_size
        seq = PrefillSequence(token_ids=token_ids, req_id=req_id, block_size=block_size)
        self.scheduler.add(seq)

    def step(self) -> tuple[torch.Tensor, list[PrefillSequence]]:
        """
        Schedule a batch of waiting sequences, run prefill, return logits and
        the scheduled sequences. Caller should stream KV cache (via callback)
        then call release_after_prefill(seqs).
        """
        scheduled = self.scheduler.schedule()
        if not scheduled:
            return torch.empty(0, 0, self.model_runner.prefill_config.vocab_size), []
        logits = self.model_runner.run_prefill(scheduled)
        return logits, scheduled

    def release_after_prefill(self, seqs: list[PrefillSequence]) -> None:
        """Release block allocations for sequences after prefill/KV stream is done."""
        self.scheduler.release(seqs)

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def cleanup(self) -> None:
        self.model_runner.cleanup()


if __name__ == "__main__":
    # Example: run from prefill-node-poc directory: python prefill_engine.py
    def on_kv_ready(layer_idx: int, ref: KVCacheReference) -> None:
        print(f"  KV layer {layer_idx} ready for whole batch")

    def on_blocks_ready(layer_idx: int, req_id: object, blocks: list) -> None:
        print(
            f"  KV layer {layer_idx} blocks ready for req_id={req_id}, num_blocks={len(blocks)}"
        )

    sched_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=512,
        num_kvcache_blocks=64,
        block_size=32,
    )
    prefill_config = PrefillConfig(num_layers=4, vocab_size=1000, block_size=32)
    engine = PrefillEngine(
        scheduler_config=sched_config,
        prefill_config=prefill_config,
        on_kv_cache_ready=on_kv_ready,
        on_kv_cache_blocks_ready=on_blocks_ready,
    )

    engine.add_request(list(range(0, 128)), req_id="req_0")
    engine.add_request(list(range(100, 200)), req_id="req_1")
    print("Added 2 requests; running step()...")
    logits, scheduled = engine.step()
    # Same as tt_model_runner: full logits [batch, max_padded_len, vocab_size]
    print(f"Full logits shape: {logits.shape}, scheduled: {len(scheduled)} seqs")
    # Last-position logits for sampling (one 1000-dim vector per request)
    last_logits = logits[:, -1, :]
    print(f"Last-position logits shape: {last_logits.shape}  (for sampling)")
    engine.release_after_prefill(scheduled)
    print("Released. Done.")
    engine.cleanup()
