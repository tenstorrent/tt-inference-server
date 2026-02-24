# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Llama-3.1-8B runner for cpp_server LLM flow.

Wraps LlamaForCausalLM from tt-metal (models.tt_transformers.tt.generator_vllm).
Designed to be called from C++ PybindLlamaModelRunner which provides block_table
from the C++ BlockManager on each step.

KV cache block allocation and lifecycle are managed entirely by the C++ Scheduler
and BlockManager.  This runner only:
  - allocates the on-device KV cache tensors during warmup,
  - builds page_table tensors from the block_table supplied per-sequence, and
  - runs prefill_forward / decode_forward on the model.

Batching:
- Prefill: batch at API level (run_step(prefill, sequences)); metal runs sequential
  forwards per sequence.
- Decode: when max_batch_size > 1, a single batched decode_forward is used.

Requires: PYTHONPATH to include TT_METAL_HOME and tt-media-server root.
Environment: HF_MODEL (e.g. meta-llama/Llama-3.1-8B), TT_VISIBLE_DEVICES.
"""

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

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
    block_table: list[int]
    current_pos: int
    prompt_len: int
    seed: int | None = None


@dataclass
class StepResult:
    """One token result (mirrors C++ TokenResult)."""

    task_id: str
    token_id: int
    finished: bool
    error: str = ""


DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B"

MAX_NUM_BLOCKS = 512
KV_CACHE_BLOCK_SIZE = 32


def _block_size_from_kv_cache(kv_cache) -> int:
    """Read block size from the first KV cache tensor. Structure: kv_cache[dp][layer] = [k_tt, v_tt]."""
    return int(kv_cache[0][0][0].shape[2])


def _sample_greedy(logits):
    """Host-side greedy sampling from a 1-D logits vector."""
    import torch

    return int(torch.argmax(logits).item())


def _sample(logits, temperature: float = 1.0, seed: int | None = None):
    """Host-side sampling with temperature and optional seed for reproducibility.
    Falls back to greedy when temperature is near zero."""
    import torch

    if temperature < 1e-6:
        return _sample_greedy(logits)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def _pad_page_table(page_table, max_width: int, torch):
    """Pad page table to fixed width for constant shape required by tracing.
    Use 0 for padding (matching vLLM convention); unused columns are never
    written to because the position mask prevents writes past current_pos."""
    current_width = page_table.shape[1]
    if current_width < max_width:
        pad = torch.zeros(
            (page_table.shape[0], max_width - current_width),
            dtype=page_table.dtype,
        )
        page_table = torch.cat([page_table, pad], dim=1)
    return page_table


def _clamp_token_ids(token_ids: list[int], vocab_size: int, logger) -> list[int]:
    """Replace out-of-range token IDs with 0 and log a warning."""
    clamped = []
    for tid in token_ids:
        if tid < 0 or tid >= vocab_size:
            logger.warning(
                f"Token ID {tid} out of range [0, {vocab_size}), replacing with 0"
            )
            clamped.append(0)
        else:
            clamped.append(tid)
    return clamped


class Llama31_8BRunner(BaseMetalDeviceRunner):
    """
    Runner that wraps tt-metal LlamaForCausalLM for use with cpp_server LLM engine.

    KV cache block management is owned by the C++ BlockManager.  Each StepSequence
    carries a block_table (list of block IDs) that this runner converts to a
    page_table tensor for the metal model.

    Calling convention mirrors the vLLM TT plugin (tt_model_runner.py):
    - prefill_forward / decode_forward are called WITHOUT sampling_params
      so the model returns logits; sampling happens on host.
    - warmup_prefill=True (default) compiles operators and captures traces;
      skipping it causes "unsafe device buffer allocation" that corrupts output.
    - enable_trace=True for decode to match vLLM behaviour.
    - Token IDs are clamped to [0, vocab_size) to prevent OOB embedding lookups.
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.model = None
        self.hf_model_name = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
        self.max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", 16))
        self.max_seq_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
        self._kv_cache = None
        self._vocab_size = 0
        self._max_num_blocks_per_seq = 0

    def get_pipeline_device_params(self):
        return {"num_command_queues": 2, "trace_region_size": 32 * 1024 * 1024}

    def _page_table_from_block_ids(self, block_ids: list[int], torch) -> Any:
        """Build padded page_table tensor from a list of block ids (single sequence)."""
        row = torch.tensor([block_ids], dtype=torch.int32)
        return _pad_page_table(row, self._max_num_blocks_per_seq, torch)

    def _allocate_kv_cache(self) -> None:
        import torch

        a = self.model.model_args[0]
        self._vocab_size = a.vocab_size
        self._kv_cache = self.model.allocate_kv_cache(
            (MAX_NUM_BLOCKS, a.n_kv_heads, KV_CACHE_BLOCK_SIZE, a.head_dim),
            torch.bfloat16,
            a.n_layers,
        )
        self._block_size = _block_size_from_kv_cache(self._kv_cache)
        self._max_num_blocks_per_seq = min(
            math.ceil(self.max_seq_len / self._block_size), MAX_NUM_BLOCKS
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

    def run(self, *args: Any, **kwargs: Any) -> list[StepResult]:
        """Required by BaseDeviceRunner. Pipe protocol uses run_step() instead."""
        is_prefill = kwargs.get("is_prefill", True)
        sequences = kwargs.get("sequences", [])
        return self.run_step(is_prefill, sequences)

    EOS_TOKEN_ID = 128001
    STOP_TOKEN_IDS: frozenset[int] = frozenset({128001, 128008, 128009})

    def run_step(
        self,
        is_prefill: bool,
        sequences: list[StepSequence],
    ) -> list[StepResult]:
        """Run one scheduler step (prefill or decode), returning one token per sequence.

        block_table is provided per-sequence by the C++ BlockManager via StepSequence.
        This runner builds page_table tensors from block_table and calls the model.
        """
        import torch

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
                token_id=self.EOS_TOKEN_ID,
                finished=True,
                error="empty block_table for prefill",
            )

        safe_ids = _clamp_token_ids(seq.token_ids, self._vocab_size, self.logger)
        prompt_len = len(safe_ids)
        page_table = self._page_table_from_block_ids(seq.block_table, torch)
        tokens = torch.tensor([safe_ids], dtype=torch.int64)

        logits_output = self.model.prefill_forward(
            tokens=tokens,
            page_table=page_table,
            kv_cache=self._kv_cache,
            prompt_lens=[prompt_len],
            empty_slots=[0],
        )

        logits_1d = (
            logits_output[0, -1, :]
            if logits_output.dim() >= 3
            else logits_output.flatten()
        )
        next_token = _sample(logits_1d, seq.temperature, seq.seed)
        finished = next_token in self.STOP_TOKEN_IDS and not seq.ignore_eos
        return StepResult(task_id=seq.task_id, token_id=next_token, finished=finished)

    def _run_decode(self, seq: StepSequence, torch) -> StepResult:
        if not seq.block_table:
            return StepResult(
                task_id=seq.task_id,
                token_id=self.EOS_TOKEN_ID,
                finished=True,
                error="empty block_table for decode",
            )

        page_table = self._page_table_from_block_ids(seq.block_table, torch)
        last_token = seq.token_ids[-1]
        tokens = torch.tensor([[last_token]], dtype=torch.int64)

        logits, _ = self.model.decode_forward(
            tokens=tokens,
            start_pos=torch.tensor([seq.current_pos], dtype=torch.int64),
            page_table=page_table,
            kv_cache=self._kv_cache,
            enable_trace=True,
        )

        logits_1d = logits[0, -1, :]
        next_token = _sample(logits_1d, seq.temperature, seq.seed)
        finished = next_token in self.STOP_TOKEN_IDS and not seq.ignore_eos
        return StepResult(task_id=seq.task_id, token_id=next_token, finished=finished)

    def _run_decode_batch(
        self, sequences: list[StepSequence], torch
    ) -> list[StepResult]:
        """Single batched decode forward when len(sequences) <= max_batch_size."""
        B = self.max_batch_size
        if len(sequences) > B:
            return [self._run_decode(s, torch) for s in sequences]

        tokens_list = [0] * B
        start_pos_list = [0] * B
        page_tables = []

        for i, seq in enumerate(sequences):
            tokens_list[i] = seq.token_ids[-1]
            start_pos_list[i] = seq.current_pos
            page_tables.append(self._page_table_from_block_ids(seq.block_table, torch))

        tokens_batch = torch.tensor([[t] for t in tokens_list], dtype=torch.int64)
        start_pos_batch = torch.tensor(start_pos_list, dtype=torch.int64)
        page_table_batch = torch.cat(page_tables, dim=0)
        pad_rows = B - page_table_batch.shape[0]
        if pad_rows > 0:
            pad = torch.zeros(
                (pad_rows, self._max_num_blocks_per_seq), dtype=torch.int32
            )
            page_table_batch = torch.cat([page_table_batch, pad], dim=0)

        logits, _ = self.model.decode_forward(
            tokens=tokens_batch,
            start_pos=start_pos_batch,
            page_table=page_table_batch,
            kv_cache=self._kv_cache,
            enable_trace=True,
        )

        results = []
        for i, seq in enumerate(sequences):
            logits_1d = logits[i, -1, :]
            next_token = _sample(logits_1d, seq.temperature, seq.seed)
            finished = next_token in self.STOP_TOKEN_IDS and not seq.ignore_eos
            results.append(
                StepResult(task_id=seq.task_id, token_id=next_token, finished=finished)
            )
        return results


def _run_async_warmup(runner: Llama31_8BRunner) -> bool:
    import asyncio

    return asyncio.run(runner.warmup())


def main() -> int:
    """Pipe protocol: read length-prefixed JSON from stdin, write length-prefixed JSON to stdout."""
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    protocol_out = open(protocol_fd, "wb", closefd=False)

    logger = TTLogger()
    device_id = os.environ.get("TT_VISIBLE_DEVICES", "1")
    runner = Llama31_8BRunner(device_id)

    if not _run_async_warmup(runner):
        logger.error("Warmup failed")
        return 1
    logger.info("Llama runner ready")

    while True:
        try:
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
                    block_table=s.get("block_table", []),
                    current_pos=s.get("current_pos", 0),
                    prompt_len=s.get("prompt_len", len(s.get("token_ids", []))),
                    seed=s.get("seed"),
                )
                for s in req.get("sequences", [])
            ]
            step_results = runner.run_step(is_prefill, seqs)
            response = [
                {
                    "task_id": r.task_id,
                    "token_id": r.token_id,
                    "finished": r.finished,
                    **({"error": r.error} if r.error else {}),
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
