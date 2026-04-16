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

from models.common.sampling import SamplingParams
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

    task_id: int
    token_ids: list[int]
    temperature: float
    ignore_eos: bool
    block_table: list[int]
    current_pos: int
    prompt_len: int
    seed: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@dataclass
class StepResult:
    """One token result (mirrors C++ TokenResult)."""

    task_id: int
    token_id: int
    error: str = ""


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
    - Sampling is done on device using SamplingParams forwarded from C++.
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.model = None
        self.hf_model_name = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
        self.max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", 32))
        self.max_seq_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
        self._kv_cache = None
        self._max_num_blocks_per_seq = 0

    def get_pipeline_device_params(self):
        device_params = super().get_pipeline_device_params()
        device_params["trace_region_size"] = 50000000
        return device_params

    def _page_table_from_block_ids(self, block_ids: list[int], torch) -> Any:
        """Build page_table tensor from block IDs (single sequence, shape [1, max_blocks]).

        Unused entries are set to -1 (SKIP_PAGE_TABLE_ENTRY in the paged_fill_cache
        kernel).
        """
        max_blocks = self._max_num_blocks_per_seq
        page_table = torch.full((1, max_blocks), -1, dtype=torch.int32)
        n = min(len(block_ids), max_blocks)
        page_table[0, :n] = torch.tensor(block_ids[:n], dtype=torch.int32)
        return page_table

    def _build_sampling_params(self, seq: StepSequence) -> SamplingParams:
        """Build SamplingParams from StepSequence sampling fields."""
        return SamplingParams(
            temperature=seq.temperature,
            top_p=seq.top_p if seq.top_p is not None else 1.0,
            top_k=seq.top_k if seq.top_k is not None else 1,
            presence_penalty=seq.presence_penalty,
            frequency_penalty=seq.frequency_penalty,
            repetition_penalty=seq.repetition_penalty
            if seq.repetition_penalty is not None
            else 1.0,
            seed=seq.seed,
        )

    def _build_batch_sampling_params(
        self, sequences: list[StepSequence]
    ) -> SamplingParams:
        """Build SamplingParams with list-valued fields for batched decode.

        Each field is a list where element i corresponds to sequences[i].
        This enables per-sequence sampling parameters in batched mode.
        """
        # Build lists for each parameter
        temperatures = [seq.temperature for seq in sequences]
        top_ps = [seq.top_p if seq.top_p is not None else 1.0 for seq in sequences]
        top_ks = [seq.top_k if seq.top_k is not None else 1 for seq in sequences]
        repetition_penalties = [
            seq.repetition_penalty if seq.repetition_penalty is not None else 1.0
            for seq in sequences
        ]
        presence_penalties = [seq.presence_penalty for seq in sequences]
        frequency_penalties = [seq.frequency_penalty for seq in sequences]
        seeds = [seq.seed for seq in sequences]

        return SamplingParams(
            temperature=temperatures,
            top_p=top_ps,
            top_k=top_ks,
            presence_penalty=presence_penalties,
            frequency_penalty=frequency_penalties,
            repetition_penalty=repetition_penalties,
            seed=seeds if any(s is not None for s in seeds) else None,
        )

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

        # Warmup prefill
        self.model.warmup_model_prefill(
            kv_cache=self._kv_cache,
            enable_trace=True,
            can_sample_on_device=True,
            non_greedy_decoding_on_device=True,
        )

        # Warmup decode
        self.model.warmup_model_decode(
            kv_cache=self._kv_cache,
            enable_trace=True,
            max_batch_size=self.max_batch_size,
            num_blocks=self._max_num_blocks_per_seq,
            can_sample_on_device=True,
            non_greedy_decoding_on_device=True,
        )

        self.logger.info(
            f"Device {self.device_id}: Warmup done (max_batch_size={self.max_batch_size}, "
            "batched decode enabled when multiple sequences per step)"
        )
        return True

    def run(
        self,
        is_prefill: bool = True,
        sequences: list | None = None,
        reset_batch: bool = False,
    ) -> list[StepResult]:
        """Run one scheduler step (prefill or decode), returning one token per sequence.

        Called from C++ as runner.run(is_prefill, sequences, reset_batch) with positional args.
        reset_batch: passed from C++; True on the first decode step after prefill to reset
        on-device sampling state (prompt/output for penalties).
        """
        import torch

        if sequences is None:
            sequences = []
        if self._kv_cache is None:
            raise RuntimeError("KV cache not allocated; warmup may have failed")
        if is_prefill:
            return self._run_prefill_batch(sequences, torch)
        if self.max_batch_size > 1 and len(sequences) > 0:
            return self._run_decode_batch(sequences, torch, reset_batch=reset_batch)
        return [self._run_decode(s, torch, reset_batch=reset_batch) for s in sequences]

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
        sampling_params = self._build_sampling_params(seq)

        output_tokens, _log_probs = self.model.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=self._kv_cache,
            prompt_lens=[prompt_len],
            warmup_prefill=False,
            sampling_params=sampling_params,
        )

        next_token = int(output_tokens[0].item())
        return StepResult(task_id=seq.task_id, token_id=next_token)

    def _run_decode(
        self, seq: StepSequence, torch, reset_batch: bool = False
    ) -> StepResult:
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
        sampling_params = self._build_sampling_params(seq)

        output_tokens, _log_probs = self.model.decode_forward(
            tokens,
            current_pos,
            page_table=page_table,
            kv_cache=self._kv_cache,
            enable_trace=True,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )

        next_token = int(output_tokens[0].item())
        return StepResult(task_id=seq.task_id, token_id=next_token)

    def _run_decode_batch(
        self, sequences: list[StepSequence], torch, reset_batch: bool = False
    ) -> list[StepResult]:
        """Batched decode when len(sequences) <= max_batch_size.

        tt-metal requires batch == max_batch_size. Fill unused slots by repeating
        real sequences so every slot points to valid KV blocks (no reserved
        block 0 needed). Results from repeated slots are discarded.

        Per-sequence sampling parameters are supported: each sequence can have
        different temperature, top_p, top_k, etc. Parameters are passed as
        list-valued fields in SamplingParams.
        """
        B = self.max_batch_size
        if len(sequences) > B:
            return [self._run_decode(s, torch, reset_batch) for s in sequences]

        n = len(sequences)
        tokens_list = []
        start_pos_list = []
        page_tables = []
        padded_sequences = []
        for i in range(B):
            seq = sequences[i % n]
            tokens_list.append(seq.token_ids[-1])
            start_pos_list.append(seq.current_pos)
            page_tables.append(self._page_table_from_block_ids(seq.block_table, torch))
            padded_sequences.append(seq)

        tokens_batch = torch.tensor([[t] for t in tokens_list], dtype=torch.int64)
        start_pos_batch = torch.tensor(start_pos_list, dtype=torch.int64)
        page_table_batch = torch.cat(page_tables, dim=0)

        # Build per-sequence sampling parameters for batched decode (including padded slots)
        # Each field in SamplingParams is a list where element i corresponds to padded_sequences[i]
        sampling_params = self._build_batch_sampling_params(padded_sequences)

        output_tokens, _log_probs = self.model.decode_forward(
            tokens_batch,
            start_pos_batch,
            page_table=page_table_batch,
            kv_cache=self._kv_cache,
            enable_trace=True,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )

        # Device sampling returns tokens directly, shape [batch]
        return [
            StepResult(
                task_id=seq.task_id,
                token_id=int(output_tokens[i].item()),
            )
            for i, seq in enumerate(sequences)
        ]
