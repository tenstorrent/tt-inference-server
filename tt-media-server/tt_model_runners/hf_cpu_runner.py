# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
Generic HuggingFace CPU runner for cpp_server LLM flow.

Works with any AutoModelForCausalLM model. Configuration via environment:
  HF_MODEL              - Model name or path (default: Qwen/Qwen2.5-1.5B-Instruct)
  HF_STOP_TOKEN_IDS     - Comma-separated stop token IDs (default: model-specific)
  HF_EOS_TOKEN_ID       - EOS token ID (default: from tokenizer)
  HF_MAX_MODEL_LEN      - Max sequence length (default: 32768)
  HF_DEVICE             - torch device (default: cpu)
  HF_TRUST_REMOTE_CODE  - Trust remote code (default: true)
  HF_TORCH_DTYPE        - torch dtype: float32, float16, bfloat16 (default: float32)

Uses HuggingFace transformers (no tt-metal). KV cache via DynamicCache.
Sampling done in Python.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _parse_stop_token_ids(default_ids: list[int]) -> frozenset[int]:
    env = os.environ.get("HF_STOP_TOKEN_IDS", "")
    if env:
        return frozenset(int(x.strip()) for x in env.split(",") if x.strip())
    return frozenset(default_ids)


def _get_model_defaults(model_name: str) -> dict[str, Any]:
    """Return sensible defaults for known models."""
    # Extract model family from name for default stop tokens
    lowered = model_name.lower()
    if "qwen" in lowered:
        return {"stop_ids": [151643, 151645, 151644], "eos_id": 151643}
    if "llama" in lowered or "meta-llama" in lowered:
        return {"stop_ids": [128001, 128008, 128009], "eos_id": 128001}
    if "deepseek" in lowered:
        return {"stop_ids": [1], "eos_id": 1}
    if "gemma" in lowered:
        return {"stop_ids": [1, 107], "eos_id": 1}
    if "mistral" in lowered or "mixtral" in lowered:
        return {"stop_ids": [2], "eos_id": 2}
    if "phi" in lowered:
        return {"stop_ids": [32000, 32001, 32007], "eos_id": 32000}
    # Generic fallback
    return {"stop_ids": [], "eos_id": None}


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


class HFCPURunner:
    """
    Generic CPU runner for any HuggingFace causal LM.

    KV cache kept per-sequence in self._kv_caches (task_id -> DynamicCache).
    No block_table / paged cache — this is a POC to show cpp_server can run
    non-tt-metal models.
    """

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.model = None
        self.tokenizer = None
        self.hf_model_name = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)

        defaults = _get_model_defaults(self.hf_model_name)
        self.stop_token_ids = _parse_stop_token_ids(defaults["stop_ids"])
        self.eos_token_id = int(
            os.environ.get("HF_EOS_TOKEN_ID", str(defaults["eos_id"] or 0))
        )
        self._max_seq_len = int(os.environ.get("HF_MAX_MODEL_LEN", "32768"))
        self._device = os.environ.get("HF_DEVICE", "cpu")
        self._trust_remote = os.environ.get("HF_TRUST_REMOTE_CODE", "true").lower() == "true"
        dtype_str = os.environ.get("HF_TORCH_DTYPE", "float32")
        self._dtype = getattr(torch, dtype_str, torch.float32)

        self._kv_caches: dict[int, DynamicCache] = {}

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
        print(f"[HFCPURunner] Loading {self.hf_model_name} on {self._device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_name, trust_remote_code=self._trust_remote
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name,
            trust_remote_code=self._trust_remote,
            torch_dtype=self._dtype,
        )
        if self._device != "cpu":
            self.model = self.model.to(self._device)
        self.model.eval()

        # Override EOS from tokenizer if not set via env
        if self.eos_token_id == 0 and hasattr(self.tokenizer, "eos_token_id"):
            self.eos_token_id = self.tokenizer.eos_token_id or 0

        print(f"[HFCPURunner] Model loaded on {self._device}")
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
        if self._device != "cpu":
            input_ids = input_ids.to(self._device)
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
            return self._run_prefill(seq)

        input_ids = torch.tensor([[seq.token_ids[-1]]], dtype=torch.long)
        if self._device != "cpu":
            input_ids = input_ids.to(self._device)
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
        print("[HFCPURunner] Runner exited")
