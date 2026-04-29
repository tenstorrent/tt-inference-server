# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pytest

from utils.media_clients.tts_client import TtsClientStrategy
from utils.media_clients.test_status import TtsTestStatus
from workflows.workflow_types import ReportCheckTypes


class MockAsyncResponse:
    """Mock async response for aiohttp."""

    def __init__(self, status=200, json_data=None, headers=None):
        self.status = status
        self._json_data = json_data or {}
        self.headers = headers or {}

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

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_init_tokenizer_success(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "microsoft/speecht5_tts"
        device = MagicMock()

        strategy = TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

        assert strategy.tokenizer is not None
        mock_tokenizer.assert_called_once_with("microsoft/speecht5_tts")

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_init_tokenizer_failure(self, mock_tokenizer):
        mock_tokenizer.side_effect = Exception("Tokenizer error")
        model_spec = MagicMock()
        model_spec.hf_model_repo = "microsoft/speecht5_tts"
        device = MagicMock()

        strategy = TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

        assert strategy.tokenizer is None


class TestTtsClientStrategyGetNumCalls(unittest.TestCase):
    """Tests for _get_tts_num_calls method."""

    @patch("transformers.AutoTokenizer.from_pretrained")
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

        assert result == 10  # TTS benchmark default

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("utils.media_clients.tts_client.get_num_calls")
    def test_get_tts_num_calls_eval_default(self, mock_get_num_calls, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_get_num_calls.return_value = 2  # Default value
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        strategy = TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

        result = strategy._get_tts_num_calls(is_eval=True)

        assert result == 5  # TTS eval default

    @patch("transformers.AutoTokenizer.from_pretrained")
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


class TestTtsClientStrategyCalculateTtft(unittest.TestCase):
    """Tests for _calculate_ttft_value method."""

    @patch("transformers.AutoTokenizer.from_pretrained")
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

    @patch("transformers.AutoTokenizer.from_pretrained")
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


class TestTtsClientStrategyCalculateTailLatency(unittest.TestCase):
    """Tests for _calculate_tail_latency method."""

    @patch("transformers.AutoTokenizer.from_pretrained")
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


class TestTtsClientStrategyCalculatePerformanceCheck(unittest.TestCase):
    """Tests for _calculate_performance_check method (TTFT, RTR)."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test"
        model_spec.hf_model_repo = "test/model"
        model_spec.model_type.name = "TEXT_TO_SPEECH"
        model_spec.device_type = MagicMock()
        model_spec.device_type.name = "N150"
        device = MagicMock()
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.tts_client.get_performance_targets")
    def test_performance_check_all_pass(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(ttft_ms=100, rtr=2.0, tolerance=0.05)

        result = strategy._calculate_performance_check(ttft_value=90.0, rtr_value=2.5)

        assert result == 2  # PASS

    @patch("utils.media_clients.tts_client.get_performance_targets")
    def test_performance_check_ttft_fail(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(ttft_ms=100, rtr=None, tolerance=0.05)

        result = strategy._calculate_performance_check(ttft_value=200.0, rtr_value=None)

        assert result == 3  # FAIL

    @patch("utils.media_clients.tts_client.get_performance_targets")
    def test_performance_check_rtr_fail(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(ttft_ms=100, rtr=5.0, tolerance=0.05)

        result = strategy._calculate_performance_check(ttft_value=90.0, rtr_value=1.0)

        assert result == 3  # FAIL

    @patch("utils.media_clients.tts_client.get_performance_targets")
    def test_performance_check_no_targets(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(ttft_ms=None, rtr=None, tolerance=None)

        result = strategy._calculate_performance_check(ttft_value=100.0, rtr_value=2.0)

        assert result == ReportCheckTypes.NA

    @patch("utils.media_clients.tts_client.get_performance_targets")
    def test_performance_check_default_tolerance(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(ttft_ms=100, rtr=None, tolerance=None)

        result = strategy._calculate_performance_check(ttft_value=90.0, rtr_value=None)

        assert result == 2  # PASS with default 5% tolerance


class TestTtsClientStrategyCalculateAccuracyCheck(unittest.TestCase):
    """Tests for _calculate_accuracy_check method."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "speecht5_tts"
        model_spec.hf_model_repo = "microsoft/speecht5_tts"
        model_spec.model_type.name = "TEXT_TO_SPEECH"
        model_spec.device_type = MagicMock()
        model_spec.device_type.name = "N150"
        device = MagicMock()
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_accuracy_check_returns_undefined(self):
        """TTS has no quality metric implemented yet, so it always reports N/A."""
        strategy = self._create_strategy()
        result = strategy._calculate_accuracy_check()
        assert result == ReportCheckTypes.NA


class TestTtsClientStrategyGenerateSpeech(unittest.TestCase):
    """Tests for _generate_speech method."""

    @patch("transformers.AutoTokenizer.from_pretrained")
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
            headers={"Content-Type": "application/json"},
        )
        mock_session = MockAsyncSession(mock_response)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(strategy._generate_speech())

        assert result[0] is True  # success
        assert result[1] > 0  # elapsed time
        assert result[2] is not None  # ttft_ms
        assert result[3] is not None  # rtr
        assert result[4] == 0.32  # audio_duration

    def test_generate_speech_failure_status(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(status=500)
        mock_session = MockAsyncSession(mock_response)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(strategy._generate_speech())

        assert result[0] is False

    def test_generate_speech_no_duration(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(
            status=200,
            json_data={"audio": "base64audio"},
            headers={"Content-Type": "application/json"},
        )
        mock_session = MockAsyncSession(mock_response)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(strategy._generate_speech())

        assert result[0] is True
        assert result[3] is None  # rtr should be None without duration


class TestTtsClientStrategyRunTtsBenchmark(unittest.TestCase):
    """Tests for _run_tts_benchmark method."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_run_tts_benchmark(self):
        strategy = self._create_strategy()

        # Return tuple: (success, elapsed, ttft_ms, rtr, audio_duration)
        with patch("asyncio.run", return_value=(True, 1.5, 100.0, 2.0, 0.32)):
            result = strategy._run_tts_benchmark(3)

        assert len(result) == 3
        assert all(isinstance(s, TtsTestStatus) for s in result)


class TestTtsClientStrategyRunEval(unittest.TestCase):
    """Tests for run_eval method."""

    @patch("transformers.AutoTokenizer.from_pretrained")
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
            TtsTestStatus(status=True, elapsed=1.0, ttft_ms=100.0, rtr=2.0),
            TtsTestStatus(status=True, elapsed=1.5, ttft_ms=200.0, rtr=3.0),
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

        # run_eval wraps data in a list (same as other media clients)
        assert isinstance(report_data, list)
        assert len(report_data) == 1
        eval_data = report_data[0]

        # Verify required fields
        assert "score" in eval_data
        assert "rtr" in eval_data
        assert "p90_ttft" in eval_data
        assert "p95_ttft" in eval_data
        assert "performance_check" in eval_data  # TTFT/RTR check
        assert "accuracy_check" in eval_data

        # Verify calculated averages
        assert eval_data["score"] == 150.0  # TTFT: (100 + 200) / 2 (in ms)
        assert abs(eval_data["rtr"] - 2.5) < 0.001  # (2.0 + 3.0) / 2

    @patch("transformers.AutoTokenizer.from_pretrained")
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

    @patch("transformers.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        model_spec.hf_model_repo = "test/model"
        model_spec.model_type.name = "TEXT_TO_SPEECH"
        model_spec.device_type = MagicMock()
        model_spec.device_type.name = "N150"
        device = MagicMock()
        device.name = "test_device"
        return TtsClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.tts_client.get_num_calls", return_value=2)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_benchmark_success(self, mock_mkdir, mock_file, mock_num_calls):
        strategy = self._create_strategy()
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
        # accuracy_check is calculated in run_reports.py, not in tts_client.py

        assert report_data["model"] == "test_model"
        assert report_data["device"] == "test_device"
        assert report_data["task_type"] == "text_to_speech"

    @patch("transformers.AutoTokenizer.from_pretrained")
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

    @patch("transformers.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        model_spec.hf_model_repo = "test/model"
        model_spec.model_type.name = "TEXT_TO_SPEECH"
        model_spec.device_type = MagicMock()
        model_spec.device_type.name = "N150"
        device = MagicMock()
        device.name = "test_device"
        return TtsClientStrategy({}, model_spec, device, "/tmp/output", 8000)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_report(self, mock_mkdir, mock_file):
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
        # accuracy_check is calculated in run_reports.py, not in tts_client.py

        assert report_data["model"] == "test_model"
        assert report_data["device"] == "test_device"
        assert report_data["task_type"] == "text_to_speech"
