from contextlib import contextmanager

import pytest
from vllm.sampling_params import SamplingParams

from tt_vllm_plugin.platform import TTPlatform


@contextmanager
def _sampling_mode(mode):
    previous = getattr(TTPlatform, "sample_on_device_mode", None)
    TTPlatform.sample_on_device_mode = mode
    try:
        yield
    finally:
        TTPlatform.sample_on_device_mode = previous


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


def test_device_sampling_accepts_native_temperature_top_k_top_p():
    with _sampling_mode("all"):
        TTPlatform.validate_request(
            prompt="hello",
            params=SamplingParams(temperature=0.7, top_k=10, top_p=0.9),
            processed_inputs={},
        )


def test_host_sampling_mode_keeps_compatibility_sampling_behavior():
    with _sampling_mode(None):
        TTPlatform.validate_request(
            prompt="hello",
            params=SamplingParams(presence_penalty=0.5),
            processed_inputs={},
        )
