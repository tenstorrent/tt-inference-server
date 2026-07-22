# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from transformers import TopPLogitsWarper


@dataclass(frozen=True)
class TTSamplingParams:
    """
    Used by TTModelInput.

    The penalty fields default to no-op values (``presence_penalty`` /
    ``frequency_penalty`` == 0.0, ``repetition_penalty`` == 1.0, ``min_p`` == 0.0)
    so existing callers and on-device sampling paths are unaffected. They are
    honored host-side by :func:`sample_tokens` when token-id context is supplied
    (see TS-10). Each may be a scalar (applied to every row) or a per-row list.
    """

    temperature: Union[float, list[float]]
    top_k: Union[int, list[int]]
    top_p: Union[float, list[float]]
    presence_penalty: Union[float, list[float]] = 0.0
    frequency_penalty: Union[float, list[float]] = 0.0
    repetition_penalty: Union[float, list[float]] = 1.0
    min_p: Union[float, list[float]] = 0.0


@dataclass(frozen=True)
class TTModelInput:
    """
    Used by the TTModelRunner.
    """

    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    prompt_lens: list[int] | None
    seq_groups: list[int] | None  # Not used in V1
    block_tables: torch.Tensor
    unpadded_batch_size: Union[int, list[int]]  # List is used for DP in V1
    perform_device_sampling: bool | None  # Currently unused in v1
    tt_sampling_params: Union[
        None, TTSamplingParams, list[TTSamplingParams | None]
    ]  # List is used for DP in V1
    compat_sampling_used: bool
    sampling_metadata: None  # Not used in V1
    multi_modal_kwargs: dict
    cross_block_tables: torch.Tensor | None  # Not yet supported in V1
    # TS-8 (prefix caching): per-request count of already-cached prompt tokens
    # (block-aligned). The full prompt is still sent; a prefix-cache-aware model
    # (dots.ocr S2) computes only the uncached suffix and attends over the
    # resident prefix KV. ``None`` => no prefix info (compute the whole prompt).
    num_computed_tokens: list[int] | None = None


def top_pk_logits_efficient(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    # Do not keep the entire vocab size after top k.
    # Instead, keep the k size tensor and record the associated indices.
    if k < 1:  # no top-k sampling if set to -1 or 0
        top_k_values, top_k_indices = (
            logits,
            torch.arange(logits.shape[-1]).unsqueeze(0).repeat(logits.shape[0], 1),
        )
    else:
        top_k_values, top_k_indices = torch.topk(logits, k=k)
    top_p_values = TopPLogitsWarper(top_p=p)(None, top_k_values)
    probs = F.softmax(top_p_values / temperature, dim=-1)
    probs = torch.nan_to_num(
        probs
    )  # convert nan to num to prevent error in multinomial
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    if return_probs:
        return token, (probs, top_k_indices)
    else:
        return token


def _as_row_tensor(value, num_seqs: int, dtype=torch.float32) -> torch.Tensor:
    """Broadcast a scalar or per-row sequence into a ``[num_seqs]`` tensor."""
    if isinstance(value, (list, tuple)):
        vals = list(value)
        if len(vals) == 1:
            vals = vals * num_seqs
        assert len(vals) == num_seqs, (
            f"per-row sampling value has length {len(vals)} != num_seqs {num_seqs}"
        )
        return torch.tensor(vals, dtype=dtype)
    return torch.full((num_seqs,), float(value), dtype=dtype)


def _penalties_active(params: "TTSamplingParams") -> bool:
    """True iff any penalty field deviates from its no-op default."""

    def nonzero(v, default):
        if isinstance(v, (list, tuple)):
            return any(float(x) != default for x in v)
        return float(v) != default

    return (
        nonzero(params.presence_penalty, 0.0)
        or nonzero(params.frequency_penalty, 0.0)
        or nonzero(params.repetition_penalty, 1.0)
    )


def _min_p_active(params: "TTSamplingParams") -> bool:
    v = params.min_p
    if isinstance(v, (list, tuple)):
        return any(float(x) != 0.0 for x in v)
    return float(v) != 0.0


def _bin_counts_and_mask(token_ids_per_row, vocab_size: int, num_seqs: int):
    """Per-row token histogram and presence mask over the vocab."""
    bin_counts = torch.zeros((num_seqs, vocab_size), dtype=torch.float32)
    for i, ids in enumerate(token_ids_per_row):
        if ids is None or len(ids) == 0:
            continue
        t = torch.as_tensor(ids, dtype=torch.long)
        # Defensive: drop any out-of-range ids (e.g. padding sentinels).
        t = t[(t >= 0) & (t < vocab_size)]
        if t.numel() == 0:
            continue
        bin_counts[i].scatter_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    mask = bin_counts > 0
    return bin_counts, mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_token_ids_per_row: Sequence,
    output_token_ids_per_row: Sequence,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """Apply repetition/presence/frequency penalties (vLLM semantics).

    Repetition penalty considers tokens in the prompt OR the generated output;
    frequency/presence penalties consider only generated (output) tokens.
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = _bin_counts_and_mask(
        prompt_token_ids_per_row, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = _bin_counts_and_mask(
        output_token_ids_per_row, vocab_size, num_seqs
    )
    rep = repetition_penalties.to(logits.dtype).unsqueeze(1).repeat(1, vocab_size)
    rep[~(prompt_mask | output_mask)] = 1.0
    logits = torch.where(logits > 0, logits / rep, logits * rep)
    logits = logits - frequency_penalties.to(logits.dtype).unsqueeze(1) * output_bin_counts
    logits = logits - presence_penalties.to(logits.dtype).unsqueeze(1) * output_mask.to(
        logits.dtype
    )
    return logits


def apply_min_p(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """Mask out tokens whose probability is below ``min_p * max_prob`` per row."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    threshold = min_p.to(logits.dtype).unsqueeze(1) * top_probs
    return logits.masked_fill(probs < threshold, float("-inf"))


def sample_tokens(
    logits,
    tt_sampling_params: TTSamplingParams,
    prompt_token_ids: Optional[Sequence] = None,
    output_token_ids: Optional[Sequence] = None,
):
    """Host-side sampling with optional repetition/presence/frequency penalties.

    Penalties + ``min_p`` are applied only when per-row token-id context is
    supplied and the corresponding fields deviate from their no-op defaults, so
    the legacy greedy / top-p-top-k behavior is byte-for-byte preserved when no
    penalties are requested.
    """
    num_seqs = logits.shape[0]
    has_context = prompt_token_ids is not None or output_token_ids is not None

    if has_context and _penalties_active(tt_sampling_params):
        prompt_ctx = prompt_token_ids if prompt_token_ids is not None else [None] * num_seqs
        output_ctx = output_token_ids if output_token_ids is not None else [None] * num_seqs
        logits = apply_penalties(
            logits,
            prompt_ctx,
            output_ctx,
            _as_row_tensor(tt_sampling_params.presence_penalty, num_seqs),
            _as_row_tensor(tt_sampling_params.frequency_penalty, num_seqs),
            _as_row_tensor(tt_sampling_params.repetition_penalty, num_seqs),
        )

    if _min_p_active(tt_sampling_params):
        logits = apply_min_p(logits, _as_row_tensor(tt_sampling_params.min_p, num_seqs))

    if tt_sampling_params.temperature == 0:  # greedy decoding
        return torch.argmax(logits, dim=-1)
    else:  # top-k top-p sampling
        return top_pk_logits_efficient(
            logits,
            p=tt_sampling_params.top_p,
            k=tt_sampling_params.top_k,
            temperature=tt_sampling_params.temperature,
        )
