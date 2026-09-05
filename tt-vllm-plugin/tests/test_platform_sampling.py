from contextlib import contextmanager

import pytest
from tt_vllm_plugin.platform import TTPlatform
from vllm.sampling_params import SamplingParams


@contextmanager
def _sampling_mode(mode):
    previous = getattr(TTPlatform, "sample_on_device_mode", None)
    TTPlatform.sample_on_device_mode = mode
    try:
        yield
    finally:
        TTPlatform.sample_on_device_mode = previous


@contextmanager
def _non_greedy_support(supported):
    previous = getattr(TTPlatform, "non_greedy_decoding_on_device", None)
    TTPlatform.non_greedy_decoding_on_device = supported
    try:
        yield
    finally:
        TTPlatform.non_greedy_decoding_on_device = previous


@pytest.mark.parametrize(
    "modifier",
    [
        {"presence_penalty": 0.5},
        {"frequency_penalty": 0.5},
        {"repetition_penalty": 1.1},
        {"min_p": 0.1},
        {"logit_bias": {7: 1.0}},
        {"allowed_token_ids": [7, 11]},
        {"logprobs": 1},
    ],
)
def test_device_sampling_rejects_options_that_need_compatibility_sampling(modifier):
    with _sampling_mode("all"), pytest.raises(
        ValueError, match="require vLLM compatibility sampling"
    ):
        TTPlatform.validate_request(
            prompt="hello",
            params=SamplingParams(**modifier),
            processed_inputs={},
        )


def test_device_sampling_accepts_native_temperature_top_k_top_p_when_supported():
    with _sampling_mode("all"), _non_greedy_support(True):
        TTPlatform.validate_request(
            prompt="hello",
            params=SamplingParams(temperature=0.7, top_k=10, top_p=0.9),
            processed_inputs={},
        )


@pytest.mark.parametrize("mode", ["all", "decode_only"])
def test_device_sampling_rejects_non_greedy_for_greedy_only_model(mode):
    with _sampling_mode(mode), _non_greedy_support(False), pytest.raises(
        ValueError, match="Non-greedy sampling is unsupported by this TT model"
    ):
        TTPlatform.validate_request(
            prompt="hello",
            params=SamplingParams(temperature=1.0, top_k=128256, top_p=0.9),
            processed_inputs={},
        )


def test_device_sampling_accepts_greedy_for_greedy_only_model():
    with _sampling_mode("all"), _non_greedy_support(False):
        TTPlatform.validate_request(
            prompt="hello",
            params=SamplingParams(temperature=0.0),
            processed_inputs={},
        )


def test_host_sampling_mode_keeps_compatibility_sampling_behavior():
    with _sampling_mode(None), _non_greedy_support(False):
        TTPlatform.validate_request(
            prompt="hello",
            params=SamplingParams(presence_penalty=0.5),
            processed_inputs={},
        )
