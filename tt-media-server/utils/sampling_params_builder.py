# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import _DEFAULT_SAMPLING_PARAMS
from domain.completion_request import CompletionRequest
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind


def build_sampling_params(request: CompletionRequest) -> SamplingParams:
    """
    Build SamplingParams from request, applying defaults for unspecified values.

    Extracts sampling parameters from the request object, applies default values
    when parameters are not specified, validates parameter combinations,
    and constructs SamplingParams with the appropriate values.
    """
    defaults = _DEFAULT_SAMPLING_PARAMS

    # Extract and resolve parameters from request, falling back to defaults
    temperature = (
        request.temperature
        if request.temperature is not None
        else defaults["temperature"]
    )
    top_p = request.top_p if request.top_p is not None else defaults["top_p"]
    top_k = request.top_k if request.top_k is not None else defaults["top_k"]
    min_p = request.min_p if request.min_p is not None else defaults["min_p"]

    # We check falsey here because that is what we used to do, if user passes 0, he will get 0 tokens back
    max_tokens = request.max_tokens if request.max_tokens else defaults["max_tokens"]
    n = request.n if request.n is not None else defaults["n"]
    seed = request.seed if request.seed is not None else defaults["seed"]
    logprobs = (
        request.logprobs if request.logprobs is not None else defaults["logprobs"]
    )
    repetition_penalty = (
        request.repetition_penalty
        if request.repetition_penalty is not None
        else defaults["repetition_penalty"]
    )
    presence_penalty = (
        request.presence_penalty
        if request.presence_penalty is not None
        else defaults["presence_penalty"]
    )
    frequency_penalty = (
        request.frequency_penalty
        if request.frequency_penalty is not None
        else defaults["frequency_penalty"]
    )

    # Handle stop sequences - normalize to list
    stop = request.stop if request.stop is not None else defaults["stop"]
    if isinstance(stop, str):
        stop = [stop] if stop else []

    stop_token_ids = (
        request.stop_token_ids
        if request.stop_token_ids is not None
        else defaults["stop_token_ids"]
    )

    # Validate parameter combinations
    # When temperature is 0, sampling is deterministic (greedy decoding)
    if temperature == 0.0:
        top_p = 1.0
        top_k = 0

    return SamplingParams(
        n=n,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        seed=seed,
        stop=stop,
        stop_token_ids=stop_token_ids,
        bad_words=defaults["bad_words"],
        include_stop_str_in_output=request.include_stop_str_in_output,
        ignore_eos=request.ignore_eos,
        min_tokens=request.min_tokens,
        logprobs=logprobs,
        prompt_logprobs=request.prompt_logprobs,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        skip_special_tokens=request.skip_special_tokens,
        spaces_between_special_tokens=request.spaces_between_special_tokens,
        truncate_prompt_tokens=defaults["truncate_prompt_tokens"],
        guided_decoding=defaults["guided_decoding"],
        extra_args=defaults["extra_args"],
        output_kind=RequestOutputKind.DELTA,
    )
