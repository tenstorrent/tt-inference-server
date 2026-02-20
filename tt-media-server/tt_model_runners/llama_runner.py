# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Llama-3.1-8B runner for cpp_server LLM flow.

Wraps LlamaForCausalLM from tt-metal (models.tt_transformers.tt.generator_vllm).
Designed to be run as a subprocess: reads JSON requests from stdin, writes JSON
responses to stdout (length-prefixed). Used by C++ PipeLlamaModelRunner.

Batching:
- Pipe protocol is batch-oriented: one request can carry multiple sequences;
  response is one result per sequence (same order).
- Prefill: batch at API level (run_step(prefill, sequences)); metal runs sequential
  forwards per sequence.
- Decode: when max_batch_size > 1, a single batched decode_forward is used (logs show
  DECODE[0], DECODE[1], ...). Default max_batch_size=10; set MAX_BATCH_SIZE if needed.
  If max_batch_size=1 or unset in env with older code, decode is sequential (logs show
  "DECODE top-5" with ~35ms between lines).

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
    seed: int | None = None


@dataclass
class StepResult:
    """One token result (mirrors C++ DecodeResult)."""

    task_id: str
    token_id: int
    finished: bool
    error: str = ""


DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

MAX_NUM_BLOCKS = 512
KV_CACHE_BLOCK_SIZE = 32


def _block_size_from_kv_cache(kv_cache) -> int:
    """Read block size from the first KV cache tensor. Structure: kv_cache[dp][layer] = [k_tt, v_tt]."""
    return int(kv_cache[0][0][0].shape[2])


def _num_blocks_for_seq(seq_len: int, block_size: int) -> int:
    return math.ceil(seq_len / block_size)


@dataclass
class _SeqState:
    """Per-sequence state for decode. Runner allocates blocks; page_table is stored here."""

    page_table: Any
    current_len: int
    num_blocks_used: int
    prompt_len: int


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


def _log_top_tokens(logger, label, logits, k=5):
    """Log the top-k tokens from a 1-D logits tensor for debugging."""
    import torch

    topk = torch.topk(logits.float(), k)
    pairs = [(int(idx), float(val)) for idx, val in zip(topk.indices, topk.values)]
    logger.info(f"{label} top-{k}: {pairs}")


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
        self.max_seq_len = 1024
        self._kv_cache = None
        self._vocab_size = 0
        self._max_num_blocks_per_seq = 0
        self._next_free_block = 0
        self._free_blocks: list[int] = []
        self._seq_state: dict[str, _SeqState] = {}

    def get_pipeline_device_params(self):
        return {"num_command_queues": 2, "trace_region_size": 32 * 1024 * 1024}

    def _page_table_from_block_ids(self, block_ids: list[int], torch) -> Any:
        """Build padded page_table tensor from a list of block ids (single sequence)."""
        row = torch.tensor([block_ids], dtype=torch.int32)
        return _pad_page_table(row, self._max_num_blocks_per_seq, torch)

    def _allocate_blocks(self, count: int) -> list[int] | None:
        """Allocate `count` blocks from the free list or next_free_block. Returns None if not enough blocks."""
        blocks: list[int] = []
        for _ in range(count):
            if self._free_blocks:
                blocks.append(self._free_blocks.pop())
            elif self._next_free_block < MAX_NUM_BLOCKS:
                blocks.append(self._next_free_block)
                self._next_free_block += 1
            else:
                self._free_blocks.extend(blocks)
                return None
        return blocks

    def _allocate_single_block(self) -> int | None:
        """Allocate one block. Returns None if no blocks available."""
        if self._free_blocks:
            return self._free_blocks.pop()
        if self._next_free_block < MAX_NUM_BLOCKS:
            block_id = self._next_free_block
            self._next_free_block += 1
            return block_id
        return None

    def _free_seq_blocks(self, page_table, num_blocks_used: int) -> None:
        """Return allocated block IDs from a page table back to the free list."""
        block_ids = page_table[0, :num_blocks_used].tolist()
        self._free_blocks.extend(block_ids)

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
        # Block 0 is reserved as a null/scratch block: padding rows in batched
        # decode point their page tables at block 0, so writes from inactive
        # batch positions land here instead of corrupting a real sequence's KV.
        self._next_free_block = 1
        self._free_blocks = []

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

    # Llama 3.1 Instruct stop tokens (must match C++ config in settings.cpp):
    #   128001 = <|end_of_text|>
    #   128008 = <|eom_id|>   (end-of-message)
    #   128009 = <|eot_id|>   (end-of-turn, emitted after every assistant response)
    EOS_TOKEN_ID = 128001
    STOP_TOKEN_IDS: frozenset[int] = frozenset({128001, 128008, 128009})

    def run_step(
        self,
        is_prefill: bool,
        sequences: list[StepSequence],
        request_block_size: int | None = None,
    ) -> list[StepResult]:
        """Run one scheduler step (prefill or decode), returning one token per sequence.
        Prefill: batch from API POV; metal runs sequential forwards per sequence.
        Decode: single batched metal forward when max_batch_size > 1, else sequential.
        KV cache blocks are allocated by this runner (_next_free_block).
        """
        import torch

        if (
            request_block_size is not None
            and getattr(self, "_block_size", None) is not None
        ):
            if request_block_size != self._block_size:
                self.logger.warning(
                    f"block_size mismatch: request={request_block_size} runner={self._block_size}"
                )
        if self._kv_cache is None:
            return [
                StepResult(task_id=s.task_id, token_id=self.EOS_TOKEN_ID, finished=True)
                for s in sequences
            ]
        if is_prefill:
            return self._run_prefill_batch(sequences, torch)
        if self.max_batch_size > 1 and len(sequences) > 0:
            return self._run_decode_batch(sequences, torch)
        return [self._run_decode(s, torch) for s in sequences]

    def _run_prefill_batch(
        self, sequences: list[StepSequence], torch
    ) -> list[StepResult]:
        """Batch prefill: one result per sequence. Metal runs sequential forwards."""
        return [self._run_prefill(s, torch) for s in sequences]

    def _run_prefill(self, seq: StepSequence, torch) -> StepResult:
        block_size = self._block_size
        safe_ids = _clamp_token_ids(seq.token_ids, self._vocab_size, self.logger)
        prompt_len = len(safe_ids)
        num_blocks = _num_blocks_for_seq(prompt_len, block_size)
        block_ids = self._allocate_blocks(num_blocks)
        if block_ids is None:
            return StepResult(
                task_id=seq.task_id, token_id=self.EOS_TOKEN_ID, finished=True
            )
        page_table = self._page_table_from_block_ids(block_ids, torch)
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
        if finished:
            self._free_seq_blocks(page_table, num_blocks)
        else:
            self._seq_state[seq.task_id] = _SeqState(
                page_table=page_table,
                current_len=prompt_len + 1,
                num_blocks_used=num_blocks,
                prompt_len=prompt_len,
            )
        return StepResult(task_id=seq.task_id, token_id=next_token, finished=finished)

    def _run_decode(self, seq: StepSequence, torch) -> StepResult:
        state = self._seq_state.get(seq.task_id)
        if state is None:
            return StepResult(
                task_id=seq.task_id, token_id=self.EOS_TOKEN_ID, finished=True
            )
        block_size = self._block_size
        last_token = seq.token_ids[-1]
        current_len = state.current_len
        num_blocks_used = state.num_blocks_used
        page_table = state.page_table

        if (current_len + 1) > num_blocks_used * block_size:
            new_block_id = self._allocate_single_block()
            if new_block_id is None:
                self._free_seq_blocks(state.page_table, num_blocks_used)
                self._seq_state.pop(seq.task_id, None)
                return StepResult(
                    task_id=seq.task_id, token_id=self.EOS_TOKEN_ID, finished=True
                )
            block_ids = state.page_table[0, :num_blocks_used].tolist()
            block_ids.append(new_block_id)
            page_table = self._page_table_from_block_ids(block_ids, torch)
            num_blocks_used += 1
            state = _SeqState(
                page_table=page_table,
                current_len=state.current_len,
                num_blocks_used=num_blocks_used,
                prompt_len=state.prompt_len,
            )

        tokens = torch.tensor([[last_token]], dtype=torch.int64)
        position = current_len - 1

        logits, _ = self.model.decode_forward(
            tokens=tokens,
            start_pos=torch.tensor([position], dtype=torch.int64),
            page_table=page_table,
            kv_cache=self._kv_cache,
            enable_trace=True,
        )

        logits_1d = logits[0, -1, :]
        next_token = _sample(logits_1d, seq.temperature, seq.seed)

        new_len = current_len + 1
        num_generated = new_len - state.prompt_len
        hit_eos = next_token in self.STOP_TOKEN_IDS and not seq.ignore_eos
        hit_max = num_generated >= seq.max_tokens
        finished = hit_eos or hit_max

        if finished:
            self._free_seq_blocks(state.page_table, num_blocks_used)
            self._seq_state.pop(seq.task_id, None)
        else:
            self._seq_state[seq.task_id] = _SeqState(
                page_table=state.page_table,
                current_len=new_len,
                num_blocks_used=num_blocks_used,
                prompt_len=state.prompt_len,
            )
        return StepResult(task_id=seq.task_id, token_id=next_token, finished=finished)

    def _run_decode_batch(
        self, sequences: list[StepSequence], torch
    ) -> list[StepResult]:
        """Single batched decode forward when len(sequences) <= max_batch_size.
        Page tables come from runner-owned _seq_state."""
        B = self.max_batch_size
        block_size = self._block_size
        if len(sequences) > B:
            return [self._run_decode(s, torch) for s in sequences]
        early_results: dict[int, StepResult] = {}
        valid_indices: list[int] = []
        states_after_check: list[_SeqState] = []

        for i, seq in enumerate(sequences):
            state = self._seq_state.get(seq.task_id)
            if state is None:
                early_results[i] = StepResult(
                    task_id=seq.task_id,
                    token_id=self.EOS_TOKEN_ID,
                    finished=True,
                )
                continue
            if (state.current_len + 1) > state.num_blocks_used * block_size:
                new_block_id = self._allocate_single_block()
                if new_block_id is None:
                    self._free_seq_blocks(state.page_table, state.num_blocks_used)
                    self._seq_state.pop(seq.task_id, None)
                    early_results[i] = StepResult(
                        task_id=seq.task_id,
                        token_id=self.EOS_TOKEN_ID,
                        finished=True,
                    )
                    continue
                block_ids = state.page_table[0, : state.num_blocks_used].tolist()
                block_ids.append(new_block_id)
                page_table = self._page_table_from_block_ids(block_ids, torch)
                state = _SeqState(
                    page_table=page_table,
                    current_len=state.current_len,
                    num_blocks_used=state.num_blocks_used + 1,
                    prompt_len=state.prompt_len,
                )
                self._seq_state[seq.task_id] = state
            valid_indices.append(i)
            states_after_check.append(state)

        out_order: list[StepResult] = []
        for i in range(len(sequences)):
            if i in early_results:
                out_order.append(early_results[i])
                continue
            out_order.append(None)

        if not valid_indices:
            return [out_order[i] for i in range(len(sequences))]

        tokens_list = [0] * B
        start_pos_list = [0] * B
        for pos, idx in enumerate(valid_indices):
            seq = sequences[idx]
            state = states_after_check[pos]
            tokens_list[pos] = seq.token_ids[-1]
            start_pos_list[pos] = int(state.current_len - 1)

        tokens_batch = torch.tensor([[t] for t in tokens_list], dtype=torch.int64)
        start_pos_batch = torch.tensor(
            start_pos_list, dtype=torch.int64
        )  # tt-metal expects int64
        page_tables = [
            states_after_check[pos].page_table for pos in range(len(valid_indices))
        ]
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

        for pos, idx in enumerate(valid_indices):
            seq = sequences[idx]
            state = states_after_check[pos]
            logits_1d = logits[pos, -1, :]
            next_token = _sample(logits_1d, seq.temperature, seq.seed)
            new_len = state.current_len + 1
            num_generated = new_len - state.prompt_len
            hit_eos = next_token in self.STOP_TOKEN_IDS and not seq.ignore_eos
            hit_max = num_generated >= seq.max_tokens
            finished = hit_eos or hit_max

            if finished:
                self._free_seq_blocks(state.page_table, state.num_blocks_used)
                self._seq_state.pop(seq.task_id, None)
            else:
                self._seq_state[seq.task_id] = _SeqState(
                    page_table=state.page_table,
                    current_len=new_len,
                    num_blocks_used=state.num_blocks_used,
                    prompt_len=state.prompt_len,
                )
            out_order[idx] = StepResult(
                task_id=seq.task_id, token_id=next_token, finished=finished
            )

        return out_order


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
                    seed=s.get("seed"),
                )
                for s in req.get("sequences", [])
            ]
            step_results = runner.run_step(
                is_prefill, seqs, request_block_size=req.get("block_size")
            )
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
