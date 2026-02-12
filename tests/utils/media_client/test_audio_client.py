# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from utils.media_clients.audio_client import AudioClientStrategy
from utils.media_clients.test_status import AudioTestStatus


class MockAsyncResponse:
    """Mock async response for aiohttp."""

    def __init__(self, status=200, content_lines=None):
        self.status = status
        self._content_lines = content_lines or []
        self._index = 0

    @property
    def content(self):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._content_lines):
            raise StopAsyncIteration
        line = self._content_lines[self._index]
        self._index += 1
        return line

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


class TestAudioClientStrategyInit(unittest.TestCase):
    """Tests for AudioClientStrategy.__init__ method."""

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def test_init_tokenizer_success(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "openai/whisper-large-v3"
        device = MagicMock()

        strategy = AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

        assert strategy.tokenizer is not None
        mock_tokenizer.assert_called_once_with("openai/whisper-large-v3")

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def test_init_tokenizer_failure(self, mock_tokenizer):
        mock_tokenizer.side_effect = Exception("Tokenizer error")
        model_spec = MagicMock()
        model_spec.hf_model_repo = "openai/whisper-large-v3"
        device = MagicMock()

        strategy = AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

        assert strategy.tokenizer is None


class TestAudioClientStrategyCalculateTtft(unittest.TestCase):
    """Tests for _calculate_ttft_value method."""

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_calculate_ttft_with_valid_values(self):
        strategy = self._create_strategy()
        status_list = [
            AudioTestStatus(status=True, elapsed=1.0, ttft=1.0, tsu=10.0, rtr=2.0),
            AudioTestStatus(status=True, elapsed=1.0, ttft=2.0, tsu=10.0, rtr=2.0),
            AudioTestStatus(status=True, elapsed=1.0, ttft=3.0, tsu=10.0, rtr=2.0),
        ]
        result = strategy._calculate_ttft_value(status_list)
        assert result == 2.0

    def test_calculate_ttft_empty_list(self):
        strategy = self._create_strategy()
        result = strategy._calculate_ttft_value([])
        assert result == 0

    def test_calculate_ttft_with_none_values(self):
        strategy = self._create_strategy()
        status_list = [
            MagicMock(ttft=1.0),
            MagicMock(ttft=None),
            MagicMock(ttft=3.0),
        ]
        result = strategy._calculate_ttft_value(status_list)
        assert result == 2.0

    def test_calculate_ttft_all_none_values(self):
        strategy = self._create_strategy()
        status_list = [
            MagicMock(ttft=None),
            MagicMock(ttft=None),
        ]
        result = strategy._calculate_ttft_value(status_list)
        assert result == 0


class TestAudioClientStrategyCalculateRtr(unittest.TestCase):
    """Tests for _calculate_rtr_value method."""

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_calculate_rtr_with_valid_values(self):
        strategy = self._create_strategy()
        status_list = [
            MagicMock(rtr=2.0),
            MagicMock(rtr=4.0),
        ]
        result = strategy._calculate_rtr_value(status_list)
        assert result == 3.0

    def test_calculate_rtr_empty_list(self):
        strategy = self._create_strategy()
        result = strategy._calculate_rtr_value([])
        assert result == 0

    def test_calculate_rtr_with_none_values(self):
        strategy = self._create_strategy()
        status_list = [
            MagicMock(rtr=2.0),
            MagicMock(rtr=None),
        ]
        result = strategy._calculate_rtr_value(status_list)
        assert result == 2.0

    def test_calculate_rtr_all_none_values(self):
        strategy = self._create_strategy()
        status_list = [MagicMock(rtr=None)]
        result = strategy._calculate_rtr_value(status_list)
        assert result == 0


class TestAudioClientStrategyCalculateTsu(unittest.TestCase):
    """Tests for _calculate_tsu_value method."""

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_calculate_tsu_with_valid_values(self):
        strategy = self._create_strategy()
        status_list = [
            MagicMock(tsu=10.0),
            MagicMock(tsu=20.0),
        ]
        result = strategy._calculate_tsu_value(status_list)
        assert result == 15.0

    def test_calculate_tsu_empty_list(self):
        strategy = self._create_strategy()
        result = strategy._calculate_tsu_value([])
        assert result == 0

    def test_calculate_tsu_with_none_values(self):
        strategy = self._create_strategy()
        status_list = [
            MagicMock(tsu=10.0),
            MagicMock(tsu=None),
        ]
        result = strategy._calculate_tsu_value(status_list)
        assert result == 10.0

    def test_calculate_tsu_all_none_values(self):
        strategy = self._create_strategy()
        status_list = [MagicMock(tsu=None)]
        result = strategy._calculate_tsu_value(status_list)
        assert result == 0


class TestAudioClientStrategyCountTokens(unittest.TestCase):
    """Tests for _count_tokens method."""

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def test_count_tokens_with_tokenizer(self, mock_tokenizer_class):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_class.return_value = mock_tokenizer

        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        strategy = AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

        result = strategy._count_tokens("hello world test")
        assert result == 5

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def test_count_tokens_tokenizer_fails(self, mock_tokenizer_class):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = Exception("Encoding error")
        mock_tokenizer_class.return_value = mock_tokenizer

        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        strategy = AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

        result = strategy._count_tokens("hello world test")
        assert result == 3  # Falls back to word count

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def test_count_tokens_no_tokenizer(self, mock_tokenizer_class):
        mock_tokenizer_class.side_effect = Exception("Load error")

        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        strategy = AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

        result = strategy._count_tokens("hello world test four")
        assert result == 4  # Word count

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def test_count_tokens_empty_text(self, mock_tokenizer_class):
        mock_tokenizer_class.return_value = MagicMock()

        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        strategy = AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

        result = strategy._count_tokens("")
        assert result == 0

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def test_count_tokens_whitespace_only(self, mock_tokenizer_class):
        mock_tokenizer_class.return_value = MagicMock()

        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        strategy = AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

        result = strategy._count_tokens("   ")
        assert result == 0


class TestAudioClientStrategyRunEval(unittest.TestCase):
    """Tests for run_eval method."""

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
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
        return AudioClientStrategy(all_params, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.audio_client.get_num_calls", return_value=2)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_eval_success(self, mock_mkdir, mock_file, mock_num_calls):
        strategy = self._create_strategy()
        # Multiple status entries to verify averaging of TTFT, RTR, TSU
        status_list = [
            AudioTestStatus(status=True, elapsed=1.0, ttft=0.4, tsu=8.0, rtr=1.5),
            AudioTestStatus(status=True, elapsed=1.5, ttft=0.6, tsu=12.0, rtr=2.5),
        ]

        with patch.object(strategy, "get_health", return_value=(True, "tt-whisper")):
            with patch.object(
                strategy,
                "_run_audio_transcription_benchmark",
                return_value=status_list,
            ):
                strategy.run_eval()

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify file path pattern: {output_path}/eval_{model_id}/{hf_repo}/results_{timestamp}.json
        open_call_args = mock_file.call_args[0][0]
        path_str = str(open_call_args)
        assert "/tmp/eval_test_id/org__model/results_" in path_str
        assert path_str.endswith(".json")

        # Verify JSON content
        write_calls = mock_file().write.call_args_list
        written_content = "".join(call[0][0] for call in write_calls)
        report_data = json.loads(written_content)

        # run_eval wraps data in a list
        assert isinstance(report_data, list)
        assert len(report_data) == 1
        eval_result = report_data[0]

        # Verify all required keys exist
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
            "t/s/u",
            "rtr",
        ]
        for key in required_keys:
            assert key in eval_result, f"Missing required key: {key}"

        # Verify calculated averages (TTFT, RTR, TSU)
        # TTFT: (0.4 + 0.6) / 2 = 0.5
        assert eval_result["score"] == 0.5
        # TSU: (8.0 + 12.0) / 2 = 10.0
        assert eval_result["t/s/u"] == 10.0
        # RTR: (1.5 + 2.5) / 2 = 2.0
        assert eval_result["rtr"] == 2.0

        # Verify metadata from model_spec and all_params
        assert eval_result["model"] == "test_model"
        assert eval_result["device"] == "test_device"
        assert eval_result["task_type"] == "audio"
        assert eval_result["task_name"] == "test_task"
        assert eval_result["tolerance"] == 0.1
        assert eval_result["published_score"] == 0.9
        assert eval_result["published_score_ref"] == "ref"
        assert eval_result["accuracy_check"] == 2

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def test_run_eval_health_check_failed(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test"
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        device.name = "test"
        all_params = MagicMock()
        all_params.tasks = [MagicMock()]
        strategy = AudioClientStrategy(all_params, model_spec, device, "/tmp", 8000)

        with patch.object(strategy, "get_health", return_value=(False, None)):
            with pytest.raises(Exception):
                strategy.run_eval()

    @patch("utils.media_clients.audio_client.get_num_calls", return_value=1)
    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def test_run_eval_propagates_benchmark_exception(
        self, mock_tokenizer, mock_num_calls
    ):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test"
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        device.name = "test"
        all_params = MagicMock()
        all_params.tasks = [MagicMock()]
        strategy = AudioClientStrategy(all_params, model_spec, device, "/tmp", 8000)

        with patch.object(strategy, "get_health", return_value=(True, "tt-whisper")):
            with patch.object(
                strategy,
                "_run_audio_transcription_benchmark",
                side_effect=RuntimeError("Error"),
            ):
                with pytest.raises(RuntimeError):
                    strategy.run_eval()


class TestAudioClientStrategyRunBenchmark(unittest.TestCase):
    """Tests for run_benchmark method."""

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        model_spec.hf_model_repo = "test/model"
        model_spec.model_type.name = "AUDIO"
        model_spec.cli_args = {"device": "n150"}
        device = MagicMock()
        device.name = "test_device"
        return AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.audio_client.get_num_calls", return_value=2)
    @patch(
        "utils.media_clients.audio_client.is_streaming_enabled_for_whisper",
        return_value=True,
    )
    @patch(
        "utils.media_clients.audio_client.is_preprocessing_enabled_for_whisper",
        return_value=True,
    )
    @patch("utils.media_clients.audio_client.get_performance_targets")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_benchmark_success(
        self,
        mock_mkdir,
        mock_file,
        mock_targets,
        mock_preproc,
        mock_streaming,
        mock_num_calls,
    ):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(
            ttft_ms=100, tput_user=10.0, rtr=2.0, tolerance=0.05
        )
        # Multiple status entries to verify averaging of TTFT, RTR, TSU
        status_list = [
            AudioTestStatus(status=True, elapsed=1.0, ttft=0.4, tsu=8.0, rtr=1.5),
            AudioTestStatus(status=True, elapsed=1.5, ttft=0.6, tsu=12.0, rtr=2.5),
        ]

        with patch.object(strategy, "get_health", return_value=(True, "tt-whisper")):
            with patch.object(
                strategy,
                "_run_audio_transcription_benchmark",
                return_value=status_list,
            ):
                result = strategy.run_benchmark()

        assert result is True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify file path pattern: {output_path}/benchmark_{model_id}_{timestamp}.json
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
        assert "streaming_enabled" in report_data
        assert "preprocessing_enabled" in report_data

        # Verify benchmarks structure and computed averages
        benchmarks = report_data["benchmarks"]
        assert benchmarks["num_requests"] == 2
        # TTFT: (0.4 + 0.6) / 2 = 0.5
        assert benchmarks["ttft"] == 0.5
        # TSU: (8.0 + 12.0) / 2 = 10.0
        assert benchmarks["t/s/u"] == 10.0
        # RTR: (1.5 + 2.5) / 2 = 2.0
        assert benchmarks["rtr"] == 2.0
        assert "accuracy_check" in benchmarks

        # Verify metadata
        assert report_data["model"] == "test_model"
        assert report_data["device"] == "test_device"
        assert report_data["task_type"] == "audio"
        assert report_data["streaming_enabled"] is True
        assert report_data["preprocessing_enabled"] is True

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def test_run_benchmark_health_check_failed(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test"
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        device.name = "test"
        strategy = AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

        with patch.object(strategy, "get_health", return_value=(False, None)):
            with pytest.raises(Exception):
                strategy.run_benchmark()

    @patch("utils.media_clients.audio_client.get_num_calls", return_value=1)
    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def test_run_benchmark_propagates_benchmark_exception(
        self, mock_tokenizer, mock_num_calls
    ):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test"
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        device.name = "test"
        strategy = AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

        with patch.object(strategy, "get_health", return_value=(True, "tt-whisper")):
            with patch.object(
                strategy,
                "_run_audio_transcription_benchmark",
                side_effect=RuntimeError("Error"),
            ):
                with pytest.raises(RuntimeError):
                    strategy.run_benchmark()


class TestAudioClientStrategyGenerateReport(unittest.TestCase):
    """Tests for _generate_report method."""

    @patch(
        "utils.media_clients.audio_client.is_streaming_enabled_for_whisper",
        return_value=False,
    )
    @patch(
        "utils.media_clients.audio_client.is_preprocessing_enabled_for_whisper",
        return_value=False,
    )
    @patch("utils.media_clients.audio_client.get_performance_targets")
    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_report(
        self,
        mock_mkdir,
        mock_file,
        mock_tokenizer,
        mock_targets,
        mock_preproc,
        mock_streaming,
    ):
        mock_tokenizer.return_value = MagicMock()
        mock_targets.return_value = MagicMock(
            ttft_ms=100, tput_user=None, rtr=None, tolerance=0.05
        )
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        model_spec.hf_model_repo = "test/model"
        model_spec.model_type.name = "AUDIO"
        model_spec.cli_args = {"device": "n150"}
        device = MagicMock()
        device.name = "test_device"
        strategy = AudioClientStrategy({}, model_spec, device, "/tmp/output", 8000)

        status_list = [
            AudioTestStatus(status=True, elapsed=1.0, ttft=0.5, tsu=10.0, rtr=2.0)
        ]

        result = strategy._generate_report(status_list)

        assert result is True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify file path pattern: {output_path}/benchmark_{model_id}_{timestamp}.json
        open_call_args = mock_file.call_args[0][0]
        assert str(open_call_args).startswith("/tmp/output/benchmark_test_id_")
        assert str(open_call_args).endswith(".json")

        # Verify JSON content structure
        write_calls = mock_file().write.call_args_list
        written_content = "".join(call[0][0] for call in write_calls)
        report_data = json.loads(written_content)

        # Verify required top-level keys
        assert "benchmarks" in report_data
        assert "model" in report_data
        assert "device" in report_data
        assert "timestamp" in report_data
        assert "task_type" in report_data
        assert "streaming_enabled" in report_data
        assert "preprocessing_enabled" in report_data

        # Verify benchmarks structure and computed values
        benchmarks = report_data["benchmarks"]
        assert benchmarks["num_requests"] == 1
        assert benchmarks["ttft"] == 0.5
        assert benchmarks["t/s/u"] == 10.0
        assert benchmarks["rtr"] == 2.0
        assert "accuracy_check" in benchmarks

        # Verify model/device metadata
        assert report_data["model"] == "test_model"
        assert report_data["device"] == "test_device"
        assert report_data["task_type"] == "audio"
        assert report_data["streaming_enabled"] is False
        assert report_data["preprocessing_enabled"] is False


class TestAudioClientStrategyTranscribeAudio(unittest.TestCase):
    """Tests for _transcribe_audio and related methods."""

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch(
        "utils.media_clients.audio_client.is_streaming_enabled_for_whisper",
        return_value=False,
    )
    @patch(
        "utils.media_clients.audio_client.is_preprocessing_enabled_for_whisper",
        return_value=False,
    )
    def test_transcribe_audio_routes_to_streaming_off(
        self, mock_preproc, mock_streaming
    ):
        import asyncio

        strategy = self._create_strategy()

        with patch.object(
            strategy,
            "_transcribe_audio_streaming_off",
            return_value=(True, 1.0, 0.5, None, 2.0),
        ) as mock_method:
            result = asyncio.run(strategy._transcribe_audio())

        mock_method.assert_called_once_with(False)
        assert result == (True, 1.0, 0.5, None, 2.0)

    @patch(
        "utils.media_clients.audio_client.is_streaming_enabled_for_whisper",
        return_value=True,
    )
    @patch(
        "utils.media_clients.audio_client.is_preprocessing_enabled_for_whisper",
        return_value=True,
    )
    def test_transcribe_audio_routes_to_streaming_on(
        self, mock_preproc, mock_streaming
    ):
        import asyncio

        strategy = self._create_strategy()

        with patch.object(
            strategy,
            "_transcribe_audio_streaming_on",
            new_callable=AsyncMock,
            return_value=(True, 1.0, 0.5, 10.0, 2.0),
        ) as mock_method:
            result = asyncio.run(strategy._transcribe_audio())

        mock_method.assert_called_once_with(True)
        assert result == (True, 1.0, 0.5, 10.0, 2.0)


class TestAudioClientStrategyStreamingOff(unittest.TestCase):
    """Tests for _transcribe_audio_streaming_off method."""

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    @patch("utils.media_clients.audio_client.requests.post")
    def test_streaming_off_success_with_duration(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"duration": 10.0}
        mock_post.return_value = mock_response

        status, elapsed, ttft, tsu, rtr = strategy._transcribe_audio_streaming_off(
            False
        )

        assert status is True
        assert ttft == elapsed
        assert tsu is None
        assert rtr is not None

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    @patch("utils.media_clients.audio_client.requests.post")
    def test_streaming_off_success_no_duration(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        status, elapsed, ttft, tsu, rtr = strategy._transcribe_audio_streaming_off(True)

        assert status is True
        assert rtr is None

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    @patch("utils.media_clients.audio_client.requests.post")
    def test_streaming_off_failure(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        status, elapsed, ttft, tsu, rtr = strategy._transcribe_audio_streaming_off(
            False
        )

        assert status is False

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    @patch("utils.media_clients.audio_client.requests.post")
    def test_streaming_off_json_parse_error(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = Exception("Parse error")
        mock_post.return_value = mock_response

        status, elapsed, ttft, tsu, rtr = strategy._transcribe_audio_streaming_off(
            False
        )

        assert status is True
        assert rtr is None


class TestAudioClientStrategyStreamingOn(unittest.TestCase):
    """Tests for _transcribe_audio_streaming_on method."""

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.encode.return_value = [1, 2, 3]
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    def test_streaming_on_success(self):
        import asyncio

        strategy = self._create_strategy()

        chunk_data = [
            json.dumps({"text": "Hello", "chunk_id": 1}).encode(),
            json.dumps({"text": "world", "chunk_id": 2, "duration": 5.0}).encode(),
        ]
        mock_response = MockAsyncResponse(status=200, content_lines=chunk_data)

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock()
            mock_session.return_value.post.return_value = mock_response

            result = asyncio.run(strategy._transcribe_audio_streaming_on(False))

        assert result[0] is True  # success
        assert result[1] > 0  # elapsed time
        assert result[4] is not None  # rtr

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    def test_streaming_on_failure_status(self):
        import asyncio

        strategy = self._create_strategy()

        mock_response = MockAsyncResponse(status=500)

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock()
            mock_session.return_value.post.return_value = mock_response

            result = asyncio.run(strategy._transcribe_audio_streaming_on(False))

        assert result == (False, 0.0, None, None, None)

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    def test_streaming_on_exception(self):
        import asyncio

        strategy = self._create_strategy()

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                side_effect=Exception("Connection error")
            )

            result = asyncio.run(strategy._transcribe_audio_streaming_on(False))

        assert result == (False, 0.0, None, None, None)

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    def test_streaming_on_empty_lines(self):
        import asyncio

        strategy = self._create_strategy()

        chunk_data = [
            b"",
            b"   ",
            json.dumps({"text": "Hello", "chunk_id": 1, "duration": 5.0}).encode(),
        ]
        mock_response = MockAsyncResponse(status=200, content_lines=chunk_data)

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock()
            mock_session.return_value.post.return_value = mock_response

            result = asyncio.run(strategy._transcribe_audio_streaming_on(False))

        assert result[0] is True

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    def test_streaming_on_decoded_empty_line(self):
        """Test handling of line that decodes to empty string after strip."""
        import asyncio

        strategy = self._create_strategy()

        # Use NBSP (non-breaking space) in UTF-8: b'\xc2\xa0' doesn't strip as bytes,
        # but '\xa0' is Unicode whitespace that str.strip() removes
        chunk_data = [
            b"\xc2\xa0",  # NBSP - bytes.strip() keeps it, str.strip() removes it
            json.dumps({"text": "Hello", "chunk_id": 1, "duration": 5.0}).encode(),
        ]
        mock_response = MockAsyncResponse(status=200, content_lines=chunk_data)

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock()
            mock_session.return_value.post.return_value = mock_response

            result = asyncio.run(strategy._transcribe_audio_streaming_on(False))

        assert result[0] is True

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    def test_streaming_on_json_decode_error(self):
        import asyncio

        strategy = self._create_strategy()

        chunk_data = [
            b"not valid json",
            json.dumps({"text": "Hello", "chunk_id": 1, "duration": 5.0}).encode(),
        ]
        mock_response = MockAsyncResponse(status=200, content_lines=chunk_data)

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock()
            mock_session.return_value.post.return_value = mock_response

            result = asyncio.run(strategy._transcribe_audio_streaming_on(False))

        assert result[0] is True

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    def test_streaming_on_speaker_marker_skipped_for_ttft(self):
        import asyncio

        strategy = self._create_strategy()

        chunk_data = [
            json.dumps({"text": "[SPEAKER_01]", "chunk_id": 1}).encode(),
            json.dumps(
                {"text": "Hello world", "chunk_id": 2, "duration": 5.0}
            ).encode(),
        ]
        mock_response = MockAsyncResponse(status=200, content_lines=chunk_data)

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock()
            mock_session.return_value.post.return_value = mock_response

            result = asyncio.run(strategy._transcribe_audio_streaming_on(True))

        assert result[0] is True
        # TTFT should be set on "Hello world", not "[SPEAKER_01]"

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    def test_streaming_on_no_audio_duration(self):
        import asyncio

        strategy = self._create_strategy()

        chunk_data = [
            json.dumps({"text": "Hello", "chunk_id": 1}).encode(),
        ]
        mock_response = MockAsyncResponse(status=200, content_lines=chunk_data)

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock()
            mock_session.return_value.post.return_value = mock_response

            result = asyncio.run(strategy._transcribe_audio_streaming_on(False))

        assert result[0] is True
        assert result[4] is None  # rtr should be None without duration

    @patch("builtins.open", mock_open(read_data='{"file": "base64audio"}'))
    def test_streaming_on_empty_text_chunk(self):
        """Test branch where chunk has empty text (text.strip() is falsy)."""
        import asyncio

        strategy = self._create_strategy()

        chunk_data = [
            json.dumps({"text": "", "chunk_id": 1}).encode(),  # Empty text
            json.dumps({"text": "   ", "chunk_id": 2}).encode(),  # Whitespace only
            json.dumps({"text": "Hello", "chunk_id": 3, "duration": 5.0}).encode(),
        ]
        mock_response = MockAsyncResponse(status=200, content_lines=chunk_data)

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock()
            mock_session.return_value.post.return_value = mock_response

            result = asyncio.run(strategy._transcribe_audio_streaming_on(False))

        assert result[0] is True


class TestAudioClientStrategyRunAudioTranscriptionBenchmark(unittest.TestCase):
    """Tests for _run_audio_transcription_benchmark method."""

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        device = MagicMock()
        return AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_run_audio_transcription_benchmark(self):
        strategy = self._create_strategy()

        with patch("asyncio.run", return_value=(True, 1.5, 0.5, 10.0, 2.0)):
            result = strategy._run_audio_transcription_benchmark(3)

        assert len(result) == 3
        assert all(isinstance(s, AudioTestStatus) for s in result)


class TestAudioClientStrategyCalculateAccuracyCheck(unittest.TestCase):
    """Tests for _calculate_accuracy_check method."""

    @patch("utils.media_clients.audio_client.AutoTokenizer.from_pretrained")
    def _create_strategy(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        model_spec = MagicMock()
        model_spec.model_name = "test"
        model_spec.hf_model_repo = "test/model"
        model_spec.model_type.name = "AUDIO"
        model_spec.cli_args = {"device": "n150"}
        device = MagicMock()
        return AudioClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.audio_client.get_performance_targets")
    def test_accuracy_check_all_pass(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(
            ttft_ms=100, tput_user=10.0, rtr=2.0, tolerance=0.05
        )

        result = strategy._calculate_accuracy_check(
            ttft_value=90, tsu_value=11.0, rtr_value=2.5
        )

        assert result == 2  # PASS

    @patch("utils.media_clients.audio_client.get_performance_targets")
    def test_accuracy_check_ttft_fail(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(
            ttft_ms=100, tput_user=None, rtr=None, tolerance=0.05
        )

        result = strategy._calculate_accuracy_check(
            ttft_value=200, tsu_value=None, rtr_value=None
        )

        assert result == 3  # FAIL

    @patch("utils.media_clients.audio_client.get_performance_targets")
    def test_accuracy_check_no_ttft_target(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(
            ttft_ms=None, tput_user=None, rtr=None, tolerance=None
        )

        result = strategy._calculate_accuracy_check(
            ttft_value=100, tsu_value=10.0, rtr_value=2.0
        )

        assert result == 0  # UNDEFINED

    @patch("utils.media_clients.audio_client.get_performance_targets")
    def test_accuracy_check_tsu_fail(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(
            ttft_ms=100, tput_user=20.0, rtr=None, tolerance=0.05
        )

        result = strategy._calculate_accuracy_check(
            ttft_value=90, tsu_value=5.0, rtr_value=None
        )

        assert result == 3  # FAIL (TSU failed)

    @patch("utils.media_clients.audio_client.get_performance_targets")
    def test_accuracy_check_rtr_fail(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(
            ttft_ms=100, tput_user=None, rtr=5.0, tolerance=0.05
        )

        result = strategy._calculate_accuracy_check(
            ttft_value=90, tsu_value=None, rtr_value=1.0
        )

        assert result == 3  # FAIL (RTR failed)

    @patch("utils.media_clients.audio_client.get_performance_targets")
    def test_accuracy_check_default_tolerance(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(
            ttft_ms=100, tput_user=None, rtr=None, tolerance=None
        )

        result = strategy._calculate_accuracy_check(
            ttft_value=90, tsu_value=None, rtr_value=None
        )

        assert result == 2  # PASS with default tolerance

    @patch("utils.media_clients.audio_client.get_performance_targets")
    def test_accuracy_check_tsu_and_rtr_pass(self, mock_targets):
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(
            ttft_ms=100, tput_user=10.0, rtr=2.0, tolerance=0.05
        )

        result = strategy._calculate_accuracy_check(
            ttft_value=90, tsu_value=15.0, rtr_value=3.0
        )

        assert result == 2  # PASS

    @patch("utils.media_clients.audio_client.get_performance_targets")
    def test_accuracy_check_tpu_target_but_tsu_none(self, mock_targets):
        """Test branch where tput_user target exists but measured tsu_value is None."""
        strategy = self._create_strategy()
        mock_targets.return_value = MagicMock(
            ttft_ms=100, tput_user=10.0, rtr=None, tolerance=0.05
        )

        result = strategy._calculate_accuracy_check(
            ttft_value=90, tsu_value=None, rtr_value=None
        )

        # Should pass because only TTFT is checked (tsu_value is None so TSU check skipped)
        assert result == 2  # PASS
