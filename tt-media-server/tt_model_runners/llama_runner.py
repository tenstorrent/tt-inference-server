# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Llama-3.1-8B runner for cpp_server LLM flow.

Wraps LlamaForCausalLM from tt-metal (models.tt_transformers.tt.generator_vllm).
Designed to be run as a subprocess: reads JSON requests from stdin, writes JSON
responses to stdout (length-prefixed). Used by C++ PipeLlamaModelRunner.

Requires: PYTHONPATH to include TT_METAL_HOME and tt-media-server root.
Environment: HF_MODEL (e.g. meta-llama/Llama-3.1-8B-Instruct), TT_VISIBLE_DEVICES.
"""

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

# Add tt-metal to path when run as subprocess (C++ sets PYTHONPATH; fallback for local test)
_tt_metal = os.environ.get("TT_METAL_HOME")
if _tt_metal and _tt_metal not in sys.path:
    sys.path.insert(0, _tt_metal)

from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.logger import TTLogger


@dataclass
class StepSequence:
    """One sequence in a step request (mirrors C++ Sequence)."""

    task_id: str
    token_ids: list[int]
    max_tokens: int
    temperature: float
    ignore_eos: bool


@dataclass
class StepResult:
    """One token result (mirrors C++ DecodeResult)."""

    task_id: str
    token_id: int
    finished: bool


# Default model name for Llama-3.1-8B (reuse HF_MODEL env)
DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

BLOCK_SIZE = 32
MAX_NUM_BLOCKS = 512


def _num_blocks_for_seq(seq_len: int) -> int:
    return math.ceil(seq_len / BLOCK_SIZE)


def _padded_prefill_len(seq_len: int) -> int:
    return ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE


@dataclass
class _SeqState:
    page_table: Any
    current_len: int
    num_blocks_used: int
    prompt_len: int


class Llama31_8BRunner(BaseMetalDeviceRunner):
    """
    Runner that wraps tt-metal LlamaForCausalLM for use with cpp_server LLM engine.
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.model = None
        self.hf_model_name = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
        self.max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "1"))
        self.max_seq_len = int(os.environ.get("MAX_MODEL_LEN", "2048"))
        self._kv_cache = None
        self._seq_state: dict[str, _SeqState] = {}
        self._next_free_block = 0

    def get_pipeline_device_params(self):
        # Trace region required for prefill/decode; 0 causes TT_FATAL in mesh_trace
        return {"num_command_queues": 2, "trace_region_size": 32 * 1024 * 1024}

    def _allocate_kv_cache(self) -> None:
        import torch

        a = self.model.model_args[0]
        self._kv_cache = self.model.allocate_kv_cache(
            (MAX_NUM_BLOCKS, a.n_kv_heads, BLOCK_SIZE, a.head_dim),
            torch.bfloat16,
            a.n_layers,
        )

    def _load_model(self):
        from models.tt_transformers.tt.generator_vllm import LlamaForCausalLM
        from transformers import AutoConfig

        self.logger.info(f"Device {self.device_id}: Loading Llama-3.1-8B...")
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
        self.logger.info(f"Device {self.device_id}: Warmup done")
        return True

    def run(self, *args: Any, **kwargs: Any) -> list[StepResult]:
        """Required by BaseDeviceRunner. Pipe protocol uses run_step() instead."""
        is_prefill = kwargs.get("is_prefill", True)
        sequences = kwargs.get("sequences", [])
        return self.run_step(is_prefill, sequences)

    # Llama 3.1 EOS token id (stub uses this so C++ stream stops cleanly).
    EOS_TOKEN_ID = 128001

    def run_step(
        self, is_prefill: bool, sequences: list[StepSequence]
    ) -> list[StepResult]:
        """
        Run one scheduler step: prefill or decode.
        Returns one token per sequence (next token; for prefill, first generated token).

        Calls tt-metal prefill_forward / decode_forward with paged KV cache. The model is loaded in warmup()
        but never run. To produce real tokens you must:

        1. Allocate KV cache (self.model.allocate_kv_cache with block manager / config).
        2. Build paged-attention state per sequence (page_table, block allocation).
        3. Prefill: call self.model.prefill_forward_text(
             tokens=..., page_table=..., kv_cache=..., sampling_params=SamplingParams(...)
           ) and take the returned sampled token(s).
        4. Decode: call self.model.decode_forward_text(
             tokens=last_token, start_pos=..., page_table=..., kv_cache=..., sampling_params=...
           ) in a loop; each call returns the next token until EOS or max_tokens.

        The tt-metal Generator (prefill_forward_text / decode_forward_text) expects
        vLLM-style inputs: batched tokens, page tables, and SamplingParams. Wiring
        that into this pipe protocol requires a minimal block manager and per-sequence
        state (token list, position, page_table) or reusing tt-metal/vLLM scheduler logic.
        """
        import torch
        from models.tt_transformers.tt.generator import SamplingParams

        if self._kv_cache is None:
            return [
                StepResult(task_id=s.task_id, token_id=self.EOS_TOKEN_ID, finished=True)
                for s in sequences
            ]
        return [
            self._run_prefill(s, torch, SamplingParams)
            if is_prefill
            else self._run_decode(s, torch, SamplingParams)
            for s in sequences
        ]

    def _run_prefill(self, seq: StepSequence, torch, SamplingParams) -> StepResult:
        prompt_len = len(seq.token_ids)
        num_blocks = _num_blocks_for_seq(prompt_len)
        if self._next_free_block + num_blocks > MAX_NUM_BLOCKS:
            return StepResult(
                task_id=seq.task_id, token_id=self.EOS_TOKEN_ID, finished=True
            )
        block_ids = list(
            range(self._next_free_block, self._next_free_block + num_blocks)
        )
        self._next_free_block += num_blocks
        page_table = torch.tensor([block_ids], dtype=torch.int32)
        padded_len = _padded_prefill_len(prompt_len)
        tokens = torch.tensor(
            [seq.token_ids + [0] * (padded_len - prompt_len)],
            dtype=torch.int64,
        )
        # 0 = greedy; respect client value (API default 1.0 when not set)
        temp = max(0.0, float(seq.temperature))
        sampling_params = SamplingParams(temperature=temp, top_k=50, top_p=1.0)
        out_tokens, _ = self.model.prefill_forward(
            tokens=tokens,
            page_table=page_table,
            kv_cache=self._kv_cache,
            prompt_lens=torch.tensor([prompt_len]),
            empty_slots=[0],
            sampling_params=sampling_params,
            start_pos=torch.tensor([0]),
            warmup_prefill=False,
        )
        next_token = int(out_tokens.flatten()[0].item())
        self._seq_state[seq.task_id] = _SeqState(
            page_table=page_table,
            current_len=prompt_len + 1,
            num_blocks_used=num_blocks,
            prompt_len=prompt_len,
        )
        finished = next_token == self.EOS_TOKEN_ID and not seq.ignore_eos
        return StepResult(task_id=seq.task_id, token_id=next_token, finished=finished)

    def _run_decode(self, seq: StepSequence, torch, SamplingParams) -> StepResult:
        state = self._seq_state.get(seq.task_id)
        if state is None:
            return StepResult(
                task_id=seq.task_id, token_id=self.EOS_TOKEN_ID, finished=True
            )
        last_token = seq.token_ids[-1]
        current_len = state.current_len
        num_blocks_used = state.num_blocks_used
        if (current_len + 1) > num_blocks_used * BLOCK_SIZE:
            if self._next_free_block >= MAX_NUM_BLOCKS:
                return StepResult(
                    task_id=seq.task_id, token_id=self.EOS_TOKEN_ID, finished=True
                )
            new_block = torch.tensor([[self._next_free_block]], dtype=torch.int32)
            state = _SeqState(
                page_table=torch.cat([state.page_table, new_block], dim=1),
                current_len=state.current_len,
                num_blocks_used=num_blocks_used + 1,
                prompt_len=state.prompt_len,
            )
            self._next_free_block += 1
            self._seq_state[seq.task_id] = state
        tokens = torch.tensor([[last_token]], dtype=torch.int64)
        # RoPE/position_id must be the position of the token we're feeding (0-indexed), not the next slot
        position_of_input_token = current_len - 1
        temp = max(0.0, float(seq.temperature))
        sampling_params = SamplingParams(temperature=temp, top_k=50, top_p=1.0)
        out_tokens, _ = self.model.decode_forward(
            tokens=tokens,
            start_pos=torch.tensor([position_of_input_token]),
            page_table=state.page_table,
            kv_cache=self._kv_cache,
            sampling_params=sampling_params,
            enable_trace=False,
        )
        next_token = int(out_tokens.flatten()[0].item())
        new_len = current_len + 1
        self._seq_state[seq.task_id] = _SeqState(
            page_table=state.page_table,
            current_len=new_len,
            num_blocks_used=state.num_blocks_used,
            prompt_len=state.prompt_len,
        )
        num_generated = new_len - state.prompt_len
        hit_eos = next_token == self.EOS_TOKEN_ID and not seq.ignore_eos
        hit_max = num_generated >= seq.max_tokens
        finished = hit_eos or hit_max
        return StepResult(task_id=seq.task_id, token_id=next_token, finished=finished)


def _run_async_warmup(runner: Llama31_8BRunner) -> bool:
    import asyncio

    return asyncio.run(runner.warmup())


def main() -> int:
    """Pipe protocol: read length-prefixed JSON from stdin, write length-prefixed JSON to stdout."""
    # Redirect fd 1 (stdout) to stderr so any code (including C extensions) writing to stdout
    # does not corrupt the pipe. Keep the pipe write end on a duplicated fd for protocol only.
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    protocol_out = open(protocol_fd, "wb", closefd=False)

    logger = TTLogger()
    # Use TT_VISIBLE_DEVICES (set by C++ parent) so setup_runner_environment does not overwrite it with "device_0"
    device_id = os.environ.get("TT_VISIBLE_DEVICES", "1")
    runner = Llama31_8BRunner(device_id)

    if not _run_async_warmup(runner):
        logger.error("Warmup failed")
        return 1
    logger.info("Llama runner ready")

    while True:
        try:
            # Read 4-byte length (big-endian) then JSON body
            raw_len = sys.stdin.buffer.read(4)
            if not raw_len or len(raw_len) < 4:
                break
            msg_len = int.from_bytes(raw_len, "big")
            if msg_len <= 0 or msg_len > 10 * 1024 * 1024:
                logger.error(f"Invalid message length: {msg_len}")
                break
            body = sys.stdin.buffer.read(msg_len)
            if len(body) != msg_len:
                break
            req = json.loads(body.decode("utf-8"))
            if req.get("exit"):
                break
            is_prefill = req.get("is_prefill", True)
            seqs = [
                StepSequence(
                    task_id=s["task_id"],
                    token_ids=s["token_ids"],
                    max_tokens=s.get("max_tokens", 64),
                    temperature=float(s.get("temperature", 1.0)),
                    ignore_eos=bool(s.get("ignore_eos", False)),
                )
                for s in req.get("sequences", [])
            ]
            step_results = runner.run_step(is_prefill, seqs)
            response = [
                {
                    "task_id": r.task_id,
                    "token_id": r.token_id,
                    "finished": r.finished,
                }
                for r in step_results
            ]
            out_bytes = json.dumps(response).encode("utf-8")
            protocol_out.write(len(out_bytes).to_bytes(4, "big"))
            protocol_out.write(out_bytes)
            protocol_out.flush()
        except Exception as e:
            logger.error(f"Step failed: {e}")
            err = json.dumps({"error": str(e)}).encode("utf-8")
            protocol_out.write(len(err).to_bytes(4, "big"))
            protocol_out.write(err)
            protocol_out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
