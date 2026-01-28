# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Entry point for the prefill-node-poc. Uses a config object, DeepSeek V3 tokenizer,
concrete prompt strings, and runs the engine in a loop until main signals quit.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from prefill_engine import PrefillEngine
from prefill_simulator import KVCacheReference, PrefillConfig
from scheduler import SchedulerConfig


@dataclass
class AppConfig:
    """Hardcoded config for prefill-node-poc demo."""

    # Scheduler
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 512
    num_kvcache_blocks: int = 64
    block_size: int = 32

    # Simulator / model
    num_layers: int = 4
    vocab_size: int = 129280
    num_layers_sim: int = 4
    vocab_size_sim: int = 1000

    # Tokenizer (DeepSeek V3 on HuggingFace)
    tokenizer_model: str = "deepseek-ai/DeepSeek-V3"

    # Concrete prompts to tokenize and send (more than max_num_seqs to get waiting)
    prompts: list[str] = field(
        default_factory=lambda: [
            "What is the capital of France?",
            "Explain quantum computing in one sentence.",
            "Write a haiku about coding.",
            "List three benefits of open source software.",
            "What is the speed of light?",
            "Describe the water cycle briefly.",
        ]
    )

    # Run loop (main sleeps this long then signals engine to quit)
    run_seconds: float = 2.0
    verbose_kv: bool = False


def main() -> None:
    cfg = AppConfig()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_model,
            trust_remote_code=True,
        )
    except Exception as e:
        logging.warning("Could not load DeepSeek tokenizer (%s), using dummy tokens", e)
        tokenizer = None

    def on_kv_ready(layer_idx: int, ref: KVCacheReference) -> None:
        if cfg.verbose_kv:
            print(f"  KV layer {layer_idx} ready for whole batch")

    def on_blocks_ready(layer_idx: int, req_id: object, blocks: list) -> None:
        if cfg.verbose_kv:
            print(
                f"  KV layer {layer_idx} blocks ready for req_id={req_id}, num_blocks={len(blocks)}"
            )

    sched_config = SchedulerConfig(
        max_num_seqs=cfg.max_num_seqs,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        num_kvcache_blocks=cfg.num_kvcache_blocks,
        block_size=cfg.block_size,
    )
    prefill_config = PrefillConfig(
        num_layers=cfg.num_layers_sim,
        vocab_size=cfg.vocab_size_sim,
        block_size=cfg.block_size,
    )
    engine = PrefillEngine(
        scheduler_config=sched_config,
        prefill_config=prefill_config,
        on_kv_cache_ready=on_kv_ready,
        on_kv_cache_blocks_ready=on_blocks_ready,
    )

    if tokenizer is not None:
        for i, text in enumerate(cfg.prompts):
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            engine.add_request(token_ids, req_id=f"req_{i}")
            print(f"Added: {text[:50]}... -> {len(token_ids)} tokens")
    else:
        import random

        for i in range(len(cfg.prompts)):
            token_ids = [
                random.randint(0, prefill_config.vocab_size - 1) for _ in range(32)
            ]
            engine.add_request(token_ids, req_id=f"req_{i}")
            print(f"Added dummy req_{i} ({len(token_ids)} tokens)")

    stop_event = threading.Event()
    loop_thread = threading.Thread(
        target=engine.run_loop, args=(stop_event,), daemon=True
    )
    loop_thread.start()
    print(f"Engine run_loop started; will stop in {cfg.run_seconds}s ...")
    time.sleep(cfg.run_seconds)
    stop_event.set()
    loop_thread.join(timeout=2.0)
    print("Engine stopped.")
    engine.cleanup()


if __name__ == "__main__":
    main()
