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


def sample_tokens(logits, tt_sampling_params: TTSamplingParams):
    if tt_sampling_params.temperature == 0:  # greedy decoding
        return torch.argmax(logits, dim=-1)
    else:  # top-k top-p sampling
        return top_pk_logits_efficient(
            logits,
            p=tt_sampling_params.top_p,
            k=tt_sampling_params.top_k,
            temperature=tt_sampling_params.temperature,
        )
