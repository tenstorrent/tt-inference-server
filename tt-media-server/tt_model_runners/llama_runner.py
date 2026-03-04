# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Llama-3.1-8B-Instruct runner for cpp_server LLM flow.

Wraps LlamaForCausalLM from tt-metal (models.tt_transformers.tt.generator_vllm).
Designed to be called from C++ PybindLlamaModelRunner which provides block_table
from the C++ BlockManager on each step.

KV cache block allocation and lifecycle are managed entirely by the C++ Scheduler
and BlockManager.  This runner only:
  - allocates the on-device KV cache tensors during warmup,
  - builds page_table tensors from the block_table supplied per-sequence, and
  - runs prefill_forward / decode_forward on the model.

Batching:
- Prefill: batch at API level (run(prefill, sequences)); metal runs sequential
  forwards per sequence.
- Decode: when max_batch_size > 1, a single batched decode_forward is used.

Requires: PYTHONPATH to include TT_METAL_HOME and tt-media-server root.
Environment: HF_MODEL (e.g. meta-llama/Llama-3.1-8B-Instruct), TT_VISIBLE_DEVICES.
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Any

_tt_metal = os.environ.get("TT_METAL_HOME")
if _tt_metal and _tt_metal not in sys.path:
    sys.path.insert(0, _tt_metal)

from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner

# Llama 3.1 8B stop tokens (must match C++ config in settings.cpp):
#   128001 = <|end_of_text|>
#   128008 = <|eom_id|>   (end-of-message)
#   128009 = <|eot_id|>   (end-of-turn, emitted after every assistant response)
EOS_TOKEN_ID = 128001
STOP_TOKEN_IDS: frozenset[int] = frozenset({128001, 128008, 128009})
DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NUM_BLOCKS = 512
KV_CACHE_BLOCK_SIZE = 32


@dataclass
class StepSequence:
    """One sequence in a step request (mirrors C++ Sequence)."""

    task_id: str
    token_ids: list[int]
    max_tokens: int
    temperature: float
    ignore_eos: bool
    block_table: list[int]
    current_pos: int
    prompt_len: int
    seed: int | None = None


@dataclass
class StepResult:
    """One token result (mirrors C++ TokenResult)."""

    task_id: str
    token_id: int
    error: str = ""


def _sample(logits_1d, temperature: float, seed: int | None = None) -> int:
    """Host-side sampling. Argmax when temperature <= 0."""
    import torch

    if temperature <= 0:
        return int(torch.argmax(logits_1d).item())
    probs = torch.softmax(logits_1d.float() / temperature, dim=-1)
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        idx = torch.multinomial(probs, 1, generator=gen)
    else:
        idx = torch.multinomial(probs, 1)
    return int(idx.item())


class Llama31_8BRunner(BaseMetalDeviceRunner):
    """
    Runner that wraps tt-metal LlamaForCausalLM for use with cpp_server LLM engine.

    KV cache block management is owned by the C++ BlockManager.  Each StepSequence
    carries a block_table (list of block IDs) that this runner converts to a
    page_table tensor for the metal model.

    Calling convention mirrors the vLLM TT plugin (tt_model_runner.py):
    - warmup_prefill=True (default) compiles operators and captures traces;
      skipping it causes "unsafe device buffer allocation" that corrupts output.
    - enable_trace=True for decode to match vLLM behaviour.
    - Sampling is done on host from returned logits.
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.model = None
        self.hf_model_name = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
        self.max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", 16))
        self.max_seq_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
        self._kv_cache = None
        self._max_num_blocks_per_seq = 0

    def get_pipeline_device_params(self):
        return {"num_command_queues": 2, "trace_region_size": 32 * 1024 * 1024}

    def _page_table_from_block_ids(self, block_ids: list[int], torch) -> Any:
        """Build page_table tensor from block IDs (single sequence, shape [1, max_blocks])."""
        max_blocks = self._max_num_blocks_per_seq
        page_table = torch.zeros((1, max_blocks), dtype=torch.int32)
        n = min(len(block_ids), max_blocks)
        page_table[0, :n] = torch.tensor(block_ids[:n], dtype=torch.int32)
        return page_table

    def _allocate_kv_cache(self) -> None:
        import torch

        a = self.model.model_args[0]
        self._kv_cache = self.model.allocate_kv_cache(
            (MAX_NUM_BLOCKS, a.n_kv_heads, KV_CACHE_BLOCK_SIZE, a.head_dim),
            torch.bfloat16,
            a.n_layers,
        )
        self._block_size = int(self._kv_cache[0][0][0].shape[2])
        self._max_num_blocks_per_seq = min(
            math.ceil(self.max_seq_len / self._block_size), MAX_NUM_BLOCKS
        )

    def _load_model(self):
        from models.tt_transformers.tt.generator_vllm import LlamaForCausalLM
        from transformers import AutoConfig

        self.logger.info(f"Device {self.device_id}: Loading Llama-3.1-8B-Instruct...")
        hf_config = AutoConfig.from_pretrained(self.hf_model_name)
        mesh_device = self.set_device()
        self.model = LlamaForCausalLM.initialize_vllm_model(
            hf_config,
            mesh_device,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            tt_data_parallel=1,
            optimizations="performance",
        )
        self.logger.info(f"Device {self.device_id}: Model loaded")

    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Model warmup...")
        os.environ["HF_MODEL"] = self.hf_model_name
        self._load_model()
        self._allocate_kv_cache()
        if self._kv_cache is None:
            raise RuntimeError(
                f"Device {self.device_id}: KV cache allocation returned None — "
                "model.allocate_kv_cache() failed silently"
            )
        self.logger.info(
            f"Device {self.device_id}: Warmup done (max_batch_size={self.max_batch_size}, "
            "batched decode enabled when multiple sequences per step)"
        )
        return True

    def run(
        self, is_prefill: bool = True, sequences: list | None = None
    ) -> list[StepResult]:
        """Run one scheduler step (prefill or decode), returning one token per sequence.

        Called from C++ as runner.run(is_prefill, sequences) with positional args.
        """
        import torch

        if sequences is None:
            sequences = []
        if self._kv_cache is None:
            raise RuntimeError("KV cache not allocated; warmup may have failed")
        if is_prefill:
            return self._run_prefill_batch(sequences, torch)
        if self.max_batch_size > 1 and len(sequences) > 0:
            return self._run_decode_batch(sequences, torch)
        return [self._run_decode(s, torch) for s in sequences]

    def _run_prefill_batch(
        self, sequences: list[StepSequence], torch
    ) -> list[StepResult]:
        return [self._run_prefill(s, torch) for s in sequences]

    def _run_prefill(self, seq: StepSequence, torch) -> StepResult:
        if not seq.block_table:
            return StepResult(
                task_id=seq.task_id,
                token_id=EOS_TOKEN_ID,
                error="empty block_table for prefill",
            )

        prompt_len = len(seq.token_ids)
        page_table = self._page_table_from_block_ids(seq.block_table, torch)
        tokens = torch.tensor([seq.token_ids], dtype=torch.int64)

        logits = self.model.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=self._kv_cache,
            prompt_lens=[prompt_len],
        )

        logits_1d = logits[0, -1, :] if logits.dim() >= 3 else logits.flatten()
        next_token = _sample(logits_1d, seq.temperature, seq.seed)
        return StepResult(task_id=seq.task_id, token_id=next_token)

    def _run_decode(self, seq: StepSequence, torch) -> StepResult:
        if not seq.block_table:
            return StepResult(
                task_id=seq.task_id,
                token_id=EOS_TOKEN_ID,
                error="empty block_table for decode",
            )

        page_table = self._page_table_from_block_ids(seq.block_table, torch)
        last_token = seq.token_ids[-1]
        tokens = torch.tensor([[last_token]], dtype=torch.int64)
        current_pos = torch.tensor([seq.current_pos], dtype=torch.int64)

        logits, _log_probs = self.model.decode_forward(
            tokens,
            current_pos,
            page_table=page_table,
            kv_cache=self._kv_cache,
            enable_trace=True,
        )

        logits_1d = logits[0, -1, :]
        next_token = _sample(logits_1d, seq.temperature, seq.seed)
        return StepResult(task_id=seq.task_id, token_id=next_token)

    def _run_decode_batch(
        self, sequences: list[StepSequence], torch
    ) -> list[StepResult]:
        """Batched decode when len(sequences) <= max_batch_size.
        tt-metal requires batch == max_batch_size. Fill unused slots by repeating
        real sequences so every slot points to valid KV blocks (no reserved
        block 0 needed). Results from repeated slots are discarded.
        """
        B = self.max_batch_size
        if len(sequences) > B:
            return [self._run_decode(s, torch) for s in sequences]

        n = len(sequences)
        tokens_list = []
        start_pos_list = []
        page_tables = []
        for i in range(B):
            seq = sequences[i % n]
            tokens_list.append(seq.token_ids[-1])
            start_pos_list.append(seq.current_pos)
            page_tables.append(self._page_table_from_block_ids(seq.block_table, torch))

        tokens_batch = torch.tensor([[t] for t in tokens_list], dtype=torch.int64)
        start_pos_batch = torch.tensor(start_pos_list, dtype=torch.int64)
        page_table_batch = torch.cat(page_tables, dim=0)

        logits, _log_probs = self.model.decode_forward(
            tokens_batch,
            start_pos_batch,
            page_table=page_table_batch,
            kv_cache=self._kv_cache,
            enable_trace=True,
        )

        return [
            StepResult(
                task_id=seq.task_id,
                token_id=_sample(logits[i, -1, :], seq.temperature, seq.seed),
            )
            for i, seq in enumerate(sequences)
        ]
