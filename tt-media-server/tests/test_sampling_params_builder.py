# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from unittest.mock import patch

from config.constants import _DEFAULT_SAMPLING_PARAMS
from domain.completion_request import CompletionRequest


class TestBuildSamplingParamsDefaults:
    """Test that defaults are applied correctly when request has None/default values."""

    def test_default_temperature_applied_when_none(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", temperature=None)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["temperature"] == _DEFAULT_SAMPLING_PARAMS["temperature"]

    def test_default_top_p_applied_when_none(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", top_p=None)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            # When temp=0 (default), top_p is forced to 1.0
            assert kwargs["top_p"] == 1.0

    def test_default_max_tokens_applied_when_falsy(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", max_tokens=0)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["max_tokens"] == _DEFAULT_SAMPLING_PARAMS["max_tokens"]

    def test_default_max_tokens_applied_when_none(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", max_tokens=None)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["max_tokens"] == _DEFAULT_SAMPLING_PARAMS["max_tokens"]


class TestBuildSamplingParamsRequestValues:
    """Test that request values are used when provided."""

    def test_temperature_from_request(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", temperature=0.7)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["temperature"] == 0.7

    def test_top_p_from_request(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", temperature=0.7, top_p=0.9)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["top_p"] == 0.9

    def test_top_k_from_request(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", temperature=0.7, top_k=50)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["top_k"] == 50

    def test_max_tokens_from_request(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", max_tokens=100)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["max_tokens"] == 100

    def test_seed_from_request(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", seed=42)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["seed"] == 42

    def test_presence_penalty_from_request(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", presence_penalty=0.5)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["presence_penalty"] == 0.5

    def test_frequency_penalty_from_request(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", frequency_penalty=0.3)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["frequency_penalty"] == 0.3


class TestBuildSamplingParamsGreedyDecoding:
    """Test greedy decoding validation when temperature is 0."""

    def test_greedy_decoding_forces_top_p_to_1(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", temperature=0.0, top_p=0.5)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["temperature"] == 0.0
            assert kwargs["top_p"] == 1.0

    def test_greedy_decoding_forces_top_k_to_0(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", temperature=0.0, top_k=50)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["temperature"] == 0.0
            assert kwargs["top_k"] == 0

    def test_non_zero_temperature_preserves_top_p(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", temperature=0.5, top_p=0.8)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["top_p"] == 0.8


class TestBuildSamplingParamsStopSequences:
    """Test stop sequence handling."""

    def test_stop_string_converted_to_list(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", stop="STOP")
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["stop"] == ["STOP"]

    def test_stop_list_preserved(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", stop=["STOP1", "STOP2"])
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["stop"] == ["STOP1", "STOP2"]

    def test_empty_stop_string_becomes_empty_list(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", stop="")
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["stop"] == []

    def test_stop_token_ids_from_request(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", stop_token_ids=[1, 2, 3])
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["stop_token_ids"] == [1, 2, 3]


class TestBuildSamplingParamsDirectPassthrough:
    """Test parameters that pass directly from request without defaults lookup."""

    def test_include_stop_str_in_output_passthrough(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", include_stop_str_in_output=True)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["include_stop_str_in_output"] is True

    def test_ignore_eos_passthrough(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", ignore_eos=True)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["ignore_eos"] is True

    def test_min_tokens_passthrough(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", min_tokens=10)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["min_tokens"] == 10

    def test_skip_special_tokens_passthrough(self):
        with patch("utils.sampling_params_builder.SamplingParams") as mock_sp:
            from utils.sampling_params_builder import build_sampling_params

            request = CompletionRequest(prompt="test", skip_special_tokens=False)
            build_sampling_params(request)

            kwargs = mock_sp.call_args.kwargs
            assert kwargs["skip_special_tokens"] is False
