# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F
from transformers import TopPLogitsWarper


@dataclass(frozen=True)
class TTSamplingParams:
    """
    Used by TTModelInput.
    """

    temperature: Union[float, list[float]]
    top_k: Union[int, list[int]]
    top_p: Union[float, list[float]]
    seed: Union[int, None] = None


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


def top_pk_logits_efficient(
    logits, p=0.9, k=10, temperature=1.0, return_probs=False, generator=None
):
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
    top_k_id = torch.multinomial(
        probs, num_samples=1, generator=generator
    ).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    if return_probs:
        return token, (probs, top_k_indices)
    else:
        return token


# Module-level generator for seeded host-side sampling.
# Persists across decode steps so the RNG state advances deterministically
# from the initial seed set on the first call.
_host_sampling_generator: torch.Generator | None = None
_host_sampling_seed: int | None = None


def sample_tokens(logits, tt_sampling_params: TTSamplingParams):
    global _host_sampling_generator, _host_sampling_seed

    if tt_sampling_params.temperature == 0:  # greedy decoding
        return torch.argmax(logits, dim=-1)

    # top-k top-p sampling
    generator = None
    seed = tt_sampling_params.seed
    if seed is not None:
        # Create or re-seed the generator when a new seed value is provided.
        # Once seeded, the generator persists so subsequent decode steps
        # advance the RNG state deterministically.
        if _host_sampling_generator is None or _host_sampling_seed != seed:
            _host_sampling_generator = torch.Generator(device="cpu")
            _host_sampling_generator.manual_seed(seed)
            _host_sampling_seed = seed
        generator = _host_sampling_generator

    return top_pk_logits_efficient(
        logits,
        p=tt_sampling_params.top_p,
        k=tt_sampling_params.top_k,
        temperature=tt_sampling_params.temperature,
        generator=generator,
    )
