# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
Qwen-3-1.5B-Instruct CPU-only runner for cpp_server LLM flow.

Uses HuggingFace transformers (no tt-metal).  KV cache is managed via
DynamicCache (past_key_values).  Sampling is done in Python.

Designed to be called from C++ QwenModelRunner via pybind11.
"""

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EOS_TOKEN_ID = 151643  # <|endoftext|>
STOP_TOKEN_IDS: frozenset[int] = frozenset({151643, 151645, 151644})
# 151645 = <|im_end|>, 151644 = <|im_start|> (sometimes used as turn boundary)


@dataclass
class StepSequence:
    """One sequence in a step request (mirrors C++ Sequence)."""

    task_id: int
    token_ids: list[int]
    temperature: float
    ignore_eos: bool
    block_table: list[int]        # ignored for CPU runner (no paged KV)
    current_pos: int
    prompt_len: int
    seed: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    allowed_token_ids: list[int] | None = None


@dataclass
class StepResult:
    """One token result (mirrors C++ TokenResult)."""

    task_id: int
    token_id: int
    error: str = ""


class Qwen15BRunner:
    """
    CPU-only runner for Qwen-3-1.5B-Instruct.

    KV cache is kept per-sequence in self._kv_caches (task_id -> DynamicCache).
    No block_table / paged cache — this is a POC to show cpp_server can run
    non-tt-metal models.
    """

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.model = None
        self.tokenizer = None
        self.hf_model_name = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
        self._kv_caches: dict[int, DynamicCache] = {}
        self._max_seq_len = int(os.environ.get("MAX_MODEL_LEN", "32768"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_from_logits(
        self, logits_1d: torch.Tensor, seq: StepSequence
    ) -> int:
        """Host-side sampling with grammar mask, penalties, top-k/top-p."""
        if seq.allowed_token_ids is not None:
            mask = torch.full_like(logits_1d, float("-inf"))
            allowed = torch.tensor(seq.allowed_token_ids, dtype=torch.long)
            mask[allowed] = 0.0
            logits_1d = logits_1d + mask

        rep_pen = (
            seq.repetition_penalty if seq.repetition_penalty is not None else 1.0
        )
        if rep_pen != 1.0:
            positive = logits_1d > 0
            logits_1d = torch.where(
                positive, logits_1d / rep_pen, logits_1d * rep_pen
            )

        pres_pen = seq.presence_penalty
        freq_pen = seq.frequency_penalty
        if pres_pen != 0.0 or freq_pen != 0.0:
            token_counts = torch.zeros_like(logits_1d)
            for tid in seq.token_ids:
                if 0 <= tid < token_counts.size(0):
                    token_counts[tid] += 1
            appeared = (token_counts > 0).float()
            logits_1d = (
                logits_1d - freq_pen * token_counts - pres_pen * appeared
            )

        temp = seq.temperature
        if temp == 0 or temp is None:
            return int(logits_1d.argmax().item())

        scaled = logits_1d / temp

        k = seq.top_k if seq.top_k is not None and seq.top_k > 0 else 50
        top_vals, top_idx = torch.topk(scaled, min(k, scaled.size(0)))

        top_p = seq.top_p if seq.top_p is not None else 1.0
        if top_p < 1.0:
            sorted_probs, sorted_order = torch.sort(
                torch.softmax(top_vals, dim=-1), descending=True
            )
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumulative - sorted_probs) >= top_p
            sorted_probs[cutoff] = 0.0
            sorted_probs /= sorted_probs.sum()
            orig_idx = sorted_order[torch.multinomial(sorted_probs, 1)]
            chosen = top_idx[orig_idx]
        else:
            probs = torch.softmax(top_vals, dim=-1)
            if seq.seed is not None:
                g = torch.Generator()
                g.manual_seed(seq.seed)
                chosen = top_idx[torch.multinomial(probs, 1, generator=g)]
            else:
                chosen = top_idx[torch.multinomial(probs, 1)]

        return int(chosen.item())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def warmup(self) -> bool:
        print(f"[Qwen15BRunner] Loading {self.hf_model_name} on CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        print(f"[Qwen15BRunner] Model loaded on CPU")
        return True

    def run(
        self,
        is_prefill: bool = True,
        sequences: list | None = None,
        reset_batch: bool = False,
    ) -> list[StepResult]:
        """Run one scheduler step (prefill or decode), returning one token per sequence."""
        if sequences is None:
            sequences = []
        if self.model is None:
            raise RuntimeError("Model not loaded; warmup() must be called first")

        if is_prefill:
            return [self._run_prefill(seq) for seq in sequences]
        else:
            return [self._run_decode(seq) for seq in sequences]

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------
    def _run_prefill(self, seq: StepSequence) -> StepResult:
        """Process prompt tokens, build KV cache, return last token."""
        input_ids = torch.tensor([seq.token_ids], dtype=torch.long)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        logits = outputs.logits[0, -1, :]  # [vocab]
        cache: DynamicCache = outputs.past_key_values  # type: ignore[assignment]
        self._kv_caches[seq.task_id] = cache

        next_token = self._sample_from_logits(logits, seq)
        return StepResult(task_id=seq.task_id, token_id=next_token)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def _run_decode(self, seq: StepSequence) -> StepResult:
        """Generate one token using cached KV."""
        cache = self._kv_caches.get(seq.task_id)
        if cache is None:
            # Fallback: no cache -> treat as prefill (should not happen in normal flow)
            return self._run_prefill(seq)

        input_ids = torch.tensor([[seq.token_ids[-1]]], dtype=torch.long)
        with torch.no_grad():
            outputs = self.model(
                input_ids, past_key_values=cache, use_cache=True
            )
        logits = outputs.logits[0, -1, :]
        new_cache: DynamicCache = outputs.past_key_values  # type: ignore[assignment]
        self._kv_caches[seq.task_id] = new_cache

        next_token = self._sample_from_logits(logits, seq)
        return StepResult(task_id=seq.task_id, token_id=next_token)

    def exit(self) -> None:
        """Release model and caches."""
        self.model = None
        self.tokenizer = None
        self._kv_caches.clear()
        print("[Qwen15BRunner] Runner exited")
