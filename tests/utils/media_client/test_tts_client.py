# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pytest

from utils.media_clients.tts_client import DEFAULT_TTS_TEXT, TtsClientStrategy
from utils.media_clients.test_status import TtsTestStatus


class MockAsyncResponse:
    """Mock async response for aiohttp."""

    def __init__(self, status=200, json_data=None):
        self.status = status
        self._json_data = json_data or {}

    async def json(self):
        return self._json_data

    async def text(self):
        return "Error text"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockAsyncSession:
    """Mock async session for aiohttp."""

    def __init__(self, response):
        self._response = response

    def post(self, *args, **kwargs):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class TestTtsClientStrategyInit(unittest.TestCase):
    """Tests for TtsClientStrategy.__init__ method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def test_init_tokenizer_success(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "microsoft/speecht5_tts"
        device = MagicMock()

        strategy = TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

        assert strategy.tokenizer is not None
        mock_tokenizer.assert_called_once_with("microsoft/speecht5_tts")

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def test_init_tokenizer_failure(self, mock_tokenizer):
        mock_tokenizer.side_effect = Exception("Tokenizer error")
        model_spec = MagicMock()
        model_spec.hf_model_repo = "microsoft/speecht5_tts"
        device = MagicMock()

        strategy = TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

        assert strategy.tokenizer is None


class TestTtsClientStrategyGetNumCalls(unittest.TestCase):
    """Tests for _get_tts_num_calls method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    @patch("utils.media_clients.tts_client.get_num_calls")
    def test_get_tts_num_calls_benchmark_default(
        self, mock_get_num_calls, mock_tokenizer
    ):
        mock_tokenizer.return_value = MagicMock()
        mock_get_num_calls.return_value = 2  # Default value
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        strategy = TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

        result = strategy._get_tts_num_calls(is_eval=False)

        assert result == 25  # TTS benchmark default

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    @patch("utils.media_clients.tts_client.get_num_calls")
    def test_get_tts_num_calls_eval_default(self, mock_get_num_calls, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_get_num_calls.return_value = 2  # Default value
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        strategy = TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

        result = strategy._get_tts_num_calls(is_eval=True)

        assert result == 12  # TTS eval default

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    @patch("utils.media_clients.tts_client.get_num_calls")
    def test_get_tts_num_calls_respects_configured_value(
        self, mock_get_num_calls, mock_tokenizer
    ):
        mock_tokenizer.return_value = MagicMock()
        mock_get_num_calls.return_value = 10  # Custom value
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        strategy = TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

        result = strategy._get_tts_num_calls(is_eval=False)

        assert result == 10  # Respects configured value


class TestTtsClientStrategyComputeWer(unittest.TestCase):
    """Tests for _compute_wer method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    # All times tested on MAX_STEPS = 10
    def test_compute_wer_identical_strings(self):
        strategy = self._create_strategy()
        result = strategy._compute_wer("hello world", "hello world")
        assert result == 0.0

    def test_compute_wer_one_deletion(self):
        strategy = self._create_strategy()
        result = strategy._compute_wer("hello world", "hello")
        assert abs(result - 0.5) < 0.001  # 1 error / 2 words

    def test_compute_wer_one_insertion(self):
        strategy = self._create_strategy()
        result = strategy._compute_wer("hello", "hello world")
        assert result == 1.0  # 1 insertion / 1 word

    def test_compute_wer_one_substitution(self):
        strategy = self._create_strategy()
        result = strategy._compute_wer("hello world", "hello there")
        assert abs(result - 0.5) < 0.001  # 1 substitution / 2 words

    def test_compute_wer_multiple_errors(self):
        strategy = self._create_strategy()
        result = strategy._compute_wer("hello this is a test", "hello this is test")
        assert abs(result - 0.2) < 0.001  # 1 deletion / 5 words

    def test_compute_wer_empty_reference(self):
        strategy = self._create_strategy()
        result = strategy._compute_wer("", "hello")
        assert result == 1.0

    def test_compute_wer_empty_both(self):
        strategy = self._create_strategy()
        result = strategy._compute_wer("", "")
        assert result == 0.0

    def test_compute_wer_empty_hypothesis(self):
        strategy = self._create_strategy()
        result = strategy._compute_wer("hello world", "")
        assert result == 1.0  # All words deleted

    def test_compute_wer_case_insensitive(self):
        strategy = self._create_strategy()
        result = strategy._compute_wer("Hello World", "hello world")
        assert result == 0.0  # Case insensitive match

    def test_compute_wer_space_optimized(self):
        """Test that space-optimized DP works correctly."""
        strategy = self._create_strategy()
        # Test with longer hypothesis (should swap for memory efficiency)
        result = strategy._compute_wer("a test", "this is a longer test")
        assert result > 0  # Should have errors

        assert result >= 0  # Should be non-negative


class TestTtsClientStrategyCalculateTtft(unittest.TestCase):
    """Tests for _calculate_ttft_value method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_calculate_ttft_with_valid_values(self):
        strategy = self._create_strategy()
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=100.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=200.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=300.0),
        ]
        result = strategy._calculate_ttft_value(status_list)
        assert result == 200.0  # Average: (100 + 200 + 300) / 3

    def test_calculate_ttft_empty_list(self):
        strategy = self._create_strategy()
        result = strategy._calculate_ttft_value([])
        assert result == 0

    def test_calculate_ttft_with_none_values(self):
        strategy = self._create_strategy()
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=100.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=None),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=300.0),
        ]
        result = strategy._calculate_ttft_value(status_list)
        assert result == 200.0  # Average of valid values

    def test_calculate_ttft_all_none_values(self):
        strategy = self._create_strategy()
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=None),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=None),
        ]
        result = strategy._calculate_ttft_value(status_list)
        assert result == 0


class TestTtsClientStrategyCalculateRtr(unittest.TestCase):
    """Tests for _calculate_rtr_value method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_calculate_rtr_with_valid_values(self):
        strategy = self._create_strategy()
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, rtr=2.0),
            TtsTestStatus(status=True, elapsed=1.0, rtr=4.0),
        ]
        result = strategy._calculate_rtr_value(status_list)
        assert result == 3.0  # Average: (2.0 + 4.0) / 2

    def test_calculate_rtr_empty_list(self):
        strategy = self._create_strategy()
        result = strategy._calculate_rtr_value([])
        assert result == 0

    def test_calculate_rtr_with_none_values(self):
        strategy = self._create_strategy()
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, rtr=2.0),
            TtsTestStatus(status=True, elapsed=1.0, rtr=None),
        ]
        result = strategy._calculate_rtr_value(status_list)
        assert result == 2.0

    def test_calculate_rtr_all_none_values(self):
        strategy = self._create_strategy()
        status_list = [TtsTestStatus(status=True, elapsed=1.0, rtr=None)]
        result = strategy._calculate_rtr_value(status_list)
        assert result == 0


class TestTtsClientStrategyCalculateWerValue(unittest.TestCase):
    """Tests for _calculate_wer_value method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_calculate_wer_value_with_valid_values(self):
        strategy = self._create_strategy()
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, wer=0.1),
            TtsTestStatus(status=True, elapsed=1.0, wer=0.2),
            TtsTestStatus(status=True, elapsed=1.0, wer=0.3),
        ]
        result = strategy._calculate_wer_value(status_list)
        assert abs(result - 0.2) < 0.001  # Average: (0.1 + 0.2 + 0.3) / 3

    def test_calculate_wer_value_empty_list(self):
        strategy = self._create_strategy()
        result = strategy._calculate_wer_value([])
        assert result is None

    def test_calculate_wer_value_with_none_values(self):
        strategy = self._create_strategy()
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, wer=0.1),
            TtsTestStatus(status=True, elapsed=1.0, wer=None),
            TtsTestStatus(status=True, elapsed=1.0, wer=0.3),
        ]
        result = strategy._calculate_wer_value(status_list)
        assert abs(result - 0.2) < 0.001  # Average of valid values

    def test_calculate_wer_value_all_none_values(self):
        strategy = self._create_strategy()
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, wer=None),
            TtsTestStatus(status=True, elapsed=1.0, wer=None),
        ]
        result = strategy._calculate_wer_value(status_list)
        assert result is None


class TestTtsClientStrategyCalculateTailLatency(unittest.TestCase):
    """Tests for _calculate_tail_latency method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_calculate_tail_latency_with_valid_values(self):
        strategy = self._create_strategy()
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=100.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=200.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=300.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=400.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=500.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=600.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=700.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=800.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=900.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=1000.0),
        ]
        p90, p95 = strategy._calculate_tail_latency(status_list)
        # P90 should be 9th value (index 8) = 900.0
        # P95 should be 10th value (index 9) = 1000.0
        assert p90 == 900.0
        assert p95 == 1000.0

    def test_calculate_tail_latency_empty_list(self):
        strategy = self._create_strategy()
        p90, p95 = strategy._calculate_tail_latency([])
        assert p90 == 0.0  # Returns 0.0 for empty list
        assert p95 == 0.0

    def test_calculate_tail_latency_with_none_values(self):
        strategy = self._create_strategy()
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=100.0),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=None),
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=300.0),
        ]
        p90, p95 = strategy._calculate_tail_latency(status_list)
        # Only 2 valid values, P90 and P95 should be the same (300.0)
        assert p90 == 300.0
        assert p95 == 300.0

    def test_calculate_tail_latency_single_value(self):
        strategy = self._create_strategy()
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=100.0),
        ]
        p90, p95 = strategy._calculate_tail_latency(status_list)
        assert p90 == 100.0
        assert p95 == 100.0


class TestTtsClientStrategyCalculateAccuracyCheck(unittest.TestCase):
    """Tests for _calculate_accuracy_check method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test"
        model_spec.hf_model_repo = "test/model"
        model_spec.model_type.name = "TEXT_TO_SPEECH"
        model_spec.cli_args = {"device": "n150"}
        device = MagicMock()
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.tts_client.get_performance_targets")
    def test_accuracy_check_all_pass(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(ttft_ms=100, rtr=2.0, tolerance=0.05)

        result = strategy._calculate_accuracy_check(
            ttft_value=0.09, rtr_value=2.5, wer_value=None
        )  # ttft_value in seconds

        assert result == 2  # PASS

    @patch("utils.media_clients.tts_client.get_performance_targets")
    def test_accuracy_check_ttft_fail(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(ttft_ms=100, rtr=None, tolerance=0.05)

        result = strategy._calculate_accuracy_check(
            ttft_value=0.2, rtr_value=None, wer_value=None
        )  # 200ms > 105ms threshold

        assert result == 3  # FAIL

    @patch("utils.media_clients.tts_client.get_performance_targets")
    def test_accuracy_check_rtr_fail(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(ttft_ms=100, rtr=5.0, tolerance=0.05)

        result = strategy._calculate_accuracy_check(
            ttft_value=0.09, rtr_value=1.0, wer_value=None
        )  # 1.0 < 4.75 threshold

        assert result == 3  # FAIL

    @patch("utils.media_clients.tts_client.get_performance_targets")
    def test_accuracy_check_no_targets(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(ttft_ms=None, rtr=None, tolerance=None)

        result = strategy._calculate_accuracy_check(
            ttft_value=0.1, rtr_value=2.0, wer_value=None
        )

        assert result == 0  # UNDEFINED

    @patch("utils.media_clients.tts_client.get_performance_targets")
    def test_accuracy_check_default_tolerance(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(ttft_ms=100, rtr=None, tolerance=None)

        result = strategy._calculate_accuracy_check(
            ttft_value=0.09, rtr_value=None, wer_value=None
        )

        assert result == 2  # PASS with default 5% tolerance


class TestTtsClientStrategyTranscribeAudioForWer(unittest.TestCase):
    """Tests for _transcribe_audio_for_wer method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_transcribe_audio_for_wer_success(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(status=200, json_data={"text": "hello world"})
        mock_session = MockAsyncSession(mock_response)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(
                strategy._transcribe_audio_for_wer("base64audio", "hello world")
            )

        assert result is not None
        assert result == 0.0  # Perfect match

    def test_transcribe_audio_for_wer_failure_status(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(status=500)
        mock_session = MockAsyncSession(mock_response)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(
                strategy._transcribe_audio_for_wer("base64audio", "hello world")
            )

        assert result is None

    def test_transcribe_audio_for_wer_empty_transcription(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(status=200, json_data={"text": ""})
        mock_session = MockAsyncSession(mock_response)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(
                strategy._transcribe_audio_for_wer("base64audio", "hello world")
            )

        assert result is None

    def test_transcribe_audio_for_wer_exception(self):
        import asyncio

        strategy = self._create_strategy()

        class FailingSession:
            def post(self, *args, **kwargs):
                raise Exception("Connection error")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        result = asyncio.run(
            strategy._transcribe_audio_for_wer("base64audio", "hello world")
        )

        assert result is None


class TestTtsClientStrategyGenerateSpeech(unittest.TestCase):
    """Tests for _generate_speech method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_generate_speech_success(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(
            status=200,
            json_data={"audio": "base64audio", "duration": 0.32},
        )
        mock_session = MockAsyncSession(mock_response)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(strategy, "_transcribe_audio_for_wer", return_value=None):
                result = asyncio.run(strategy._generate_speech(calculate_wer=False))

        assert result[0] is True  # success
        assert result[1] > 0  # elapsed time
        assert result[2] is not None  # ttft_ms
        assert result[3] is not None  # rtr
        assert result[4] == DEFAULT_TTS_TEXT  # reference_text
        assert result[5] == 0.32  # audio_duration
        assert result[6] is None  # wer (not calculated)

    def test_generate_speech_with_wer(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(
            status=200,
            json_data={"audio": "base64audio", "duration": 0.32},
        )
        mock_session = MockAsyncSession(mock_response)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(strategy, "_transcribe_audio_for_wer", return_value=0.1):
                result = asyncio.run(strategy._generate_speech(calculate_wer=True))

        assert result[0] is True
        assert result[6] == 0.1  # wer calculated

    def test_generate_speech_failure_status(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(status=500)
        mock_session = MockAsyncSession(mock_response)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(strategy._generate_speech(calculate_wer=False))

        assert result[0] is False

    def test_generate_speech_no_duration(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(
            status=200, json_data={"audio": "base64audio"}
        )
        mock_session = MockAsyncSession(mock_response)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(strategy._generate_speech(calculate_wer=False))

        assert result[0] is True
        assert result[3] is None  # rtr should be None without duration


class TestTtsClientStrategyRunTtsBenchmark(unittest.TestCase):
    """Tests for _run_tts_benchmark method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_run_tts_benchmark(self):
        strategy = self._create_strategy()

        with patch(
            "asyncio.run", return_value=(True, 1.5, 100.0, 2.0, "text", 0.32, None)
        ):
            result = strategy._run_tts_benchmark(3, calculate_wer=False)

        assert len(result) == 3
        assert all(isinstance(s, TtsTestStatus) for s in result)


class TestTtsClientStrategyRunEval(unittest.TestCase):
    """Tests for run_eval method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        model_spec.hf_model_repo = "org/model"
        device = MagicMock()
        device.name = "test_device"
        all_params = MagicMock()
        all_params.tasks = [
            MagicMock(
                task_name="test_task",
                score=MagicMock(
                    tolerance=0.1, published_score=0.9, published_score_ref="ref"
                ),
            )
        ]
        return TtsClientStrategy(all_params, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.tts_client.get_num_calls", return_value=2)
    @patch("utils.media_clients.tts_client.get_performance_targets")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_eval_success(
        self, mock_mkdir, mock_file, mock_targets, mock_num_calls
    ):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(ttft_ms=100, rtr=2.0, tolerance=0.05)
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=100.0, rtr=2.0, wer=0.1),
            TtsTestStatus(status=True, elapsed=1.5, ttft_ms=200.0, rtr=3.0, wer=0.2),
        ]

        with patch.object(
            strategy, "get_health", return_value=(True, "tt-speecht5-tts")
        ):
            with patch.object(strategy, "_run_tts_benchmark", return_value=status_list):
                strategy.run_eval()

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify file path pattern
        open_call_args = mock_file.call_args[0][0]
        path_str = str(open_call_args)
        assert "/tmp/eval_test_id/org__model/results_" in path_str
        assert path_str.endswith(".json")

        # Verify JSON content
        write_calls = mock_file().write.call_args_list
        written_content = "".join(call[0][0] for call in write_calls)
        report_data = json.loads(written_content)

        assert isinstance(report_data, list)
        assert len(report_data) == 1
        eval_result = report_data[0]

        # Verify required keys
        required_keys = [
            "model",
            "device",
            "timestamp",
            "task_type",
            "task_name",
            "tolerance",
            "published_score",
            "score",
            "published_score_ref",
            "accuracy_check",
            "rtr",
            "wer",
        ]
        for key in required_keys:
            assert key in eval_result, f"Missing required key: {key}"

        # Verify calculated averages
        assert eval_result["score"] == 150.0  # TTFT: (100 + 200) / 2 (in ms)
        assert abs(eval_result["rtr"] - 2.5) < 0.001  # (2.0 + 3.0) / 2
        assert abs(eval_result["wer"] - 0.15) < 0.001  # (0.1 + 0.2) / 2

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def test_run_eval_health_check_failed(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test"
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        device.name = "test"
        all_params = MagicMock()
        all_params.tasks = [MagicMock()]
        strategy = TtsClientStrategy(all_params, model_spec, device, "/tmp", 8000)

        with patch.object(strategy, "get_health", return_value=(False, None)):
            with pytest.raises(Exception):
                strategy.run_eval()


class TestTtsClientStrategyRunBenchmark(unittest.TestCase):
    """Tests for run_benchmark method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        model_spec.hf_model_repo = "test/model"
        model_spec.model_type.name = "TEXT_TO_SPEECH"
        model_spec.cli_args = {"device": "n150"}
        device = MagicMock()
        device.name = "test_device"
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.tts_client.get_num_calls", return_value=2)
    @patch("utils.media_clients.tts_client.get_performance_targets")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_benchmark_success(
        self, mock_mkdir, mock_file, mock_targets, mock_num_calls
    ):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(ttft_ms=100, rtr=2.0, tolerance=0.05)
        status_list = [
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=100.0, rtr=2.0),
            TtsTestStatus(status=True, elapsed=1.5, ttft_ms=200.0, rtr=3.0),
        ]

        with patch.object(
            strategy, "get_health", return_value=(True, "tt-speecht5-tts")
        ):
            with patch.object(strategy, "_run_tts_benchmark", return_value=status_list):
                result = strategy.run_benchmark()

        assert result == status_list  # run_benchmark returns status_list
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify file path pattern
        open_call_args = mock_file.call_args[0][0]
        path_str = str(open_call_args)
        assert path_str.startswith("/tmp/benchmark_test_id_")
        assert path_str.endswith(".json")

        # Verify JSON content
        write_calls = mock_file().write.call_args_list
        written_content = "".join(call[0][0] for call in write_calls)
        report_data = json.loads(written_content)

        # Verify required top-level keys
        assert "benchmarks" in report_data
        assert "model" in report_data
        assert "device" in report_data
        assert "timestamp" in report_data
        assert "task_type" in report_data

        # Verify benchmarks structure
        benchmarks = report_data["benchmarks"]
        assert benchmarks["num_requests"] == 2
        assert benchmarks["ttft"] == 0.15  # (100 + 200) / 2 / 1000
        assert benchmarks["rtr"] == 2.5  # (2.0 + 3.0) / 2
        assert "ttft_p90" in benchmarks
        assert "ttft_p95" in benchmarks
        assert "accuracy_check" in benchmarks

        assert report_data["model"] == "test_model"
        assert report_data["device"] == "test_device"
        assert report_data["task_type"] == "tts"

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def test_run_benchmark_health_check_failed(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test"
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        device.name = "test"
        strategy = TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

        with patch.object(strategy, "get_health", return_value=(False, None)):
            with pytest.raises(Exception):
                strategy.run_benchmark()


class TestTtsClientStrategyGenerateReport(unittest.TestCase):
    """Tests for _generate_report method."""

    @patch("utils.media_clients.tts_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        model_spec.hf_model_repo = "test/model"
        model_spec.model_type.name = "TEXT_TO_SPEECH"
        model_spec.cli_args = {"device": "n150"}
        device = MagicMock()
        device.name = "test_device"
        return TtsClientStrategy({}, model_spec, device, "/tmp/output", 8000)

    @patch("utils.media_clients.tts_client.get_performance_targets")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_report(self, mock_mkdir, mock_file, mock_targets):
        mock_targets.return_value = MagicMock(ttft_ms=100, rtr=2.0, tolerance=0.05)
        strategy = self._create_strategy()

        status_list = [TtsTestStatus(status=True, elapsed=1.0, ttft_ms=100.0, rtr=2.0)]

        strategy._generate_report(status_list)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        open_call_args = mock_file.call_args[0][0]
        assert str(open_call_args).startswith("/tmp/output/benchmark_test_id_")
        assert str(open_call_args).endswith(".json")

        write_calls = mock_file().write.call_args_list
        written_content = "".join(call[0][0] for call in write_calls)
        report_data = json.loads(written_content)

        assert "benchmarks" in report_data
        assert "model" in report_data
        assert "device" in report_data
        assert "timestamp" in report_data
        assert "task_type" in report_data

        benchmarks = report_data["benchmarks"]
        assert benchmarks["num_requests"] == 1
        assert benchmarks["ttft"] == 0.1  # 100ms / 1000
        assert benchmarks["rtr"] == 2.0
        assert "ttft_p90" in benchmarks
        assert "ttft_p95" in benchmarks
        assert "accuracy_check" in benchmarks

        assert report_data["model"] == "test_model"
        assert report_data["device"] == "test_device"
        assert report_data["task_type"] == "tts"
