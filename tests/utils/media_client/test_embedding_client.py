# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import subprocess
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pytest

from utils.media_clients.embedding_client import (
    BENCHMARK_RESULT_END,
    BENCHMARK_RESULT_START,
    EmbeddingClientStrategy,
)


class TestEmbeddingClientStrategyInit(unittest.TestCase):
    """Tests for EmbeddingClientStrategy.__init__ method."""

    def test_init_success(self):
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        model_spec.device_model_spec.env_vars = {"MAX_MODEL_LENGTH": "2048"}
        model_spec.device_model_spec.max_concurrency = 4
        device = MagicMock()

        strategy = EmbeddingClientStrategy({}, model_spec, device, "/tmp", 8000)

        assert strategy.model == "test/model"
        assert strategy.isl == 2048
        assert strategy.num_calls == 1000
        assert strategy.dimensions == 1000
        assert strategy.concurrency == 4

    def test_init_default_isl(self):
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        model_spec.device_model_spec.env_vars = {}
        model_spec.device_model_spec.max_concurrency = 2
        device = MagicMock()

        strategy = EmbeddingClientStrategy({}, model_spec, device, "/tmp", 8000)

        assert strategy.isl == 1024  # Default value


class TestEmbeddingClientStrategyRunEval(unittest.TestCase):
    """Tests for run_eval method."""

    def _create_strategy(self):
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
        return EmbeddingClientStrategy(all_params, model_spec, device, "/tmp", 8000)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_eval_success(self, mock_mkdir, mock_file):
        strategy = self._create_strategy()

        # Mock MTEB evaluation results
        mock_task_result = MagicMock()
        mock_task_result.scores = {
            "test": {
                "pearson": 0.85,
                "spearman": 0.82,
                "main_score": 0.835,
            }
        }
        mock_results = MagicMock()
        mock_results.task_results = [mock_task_result]

        with patch.object(strategy, "get_health", return_value=(True, "tt-embedding")):
            with patch.object(
                strategy,
                "_run_embedding_transcription_eval",
                return_value={"pearson": 0.85, "spearman": 0.82, "main_score": 0.835},
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
        ]
        for key in required_keys:
            assert key in eval_result, f"Missing required key: {key}"

        # Verify metrics from parsed results
        assert eval_result["pearson"] == 0.85
        assert eval_result["spearman"] == 0.82
        assert eval_result["main_score"] == 0.835

        # Verify metadata from model_spec and all_params
        assert eval_result["model"] == "test_model"
        assert eval_result["device"] == "test_device"
        assert eval_result["task_type"] == "embedding"
        assert eval_result["task_name"] == "test_task"

    @patch.object(EmbeddingClientStrategy, "get_health", return_value=(False, None))
    def test_run_eval_health_check_failed(self, mock_health):
        strategy = self._create_strategy()

        with pytest.raises(Exception):
            strategy.run_eval()

    @patch.object(
        EmbeddingClientStrategy, "get_health", return_value=(True, "tt-embedding")
    )
    def test_run_eval_propagates_eval_exception(self, mock_health):
        strategy = self._create_strategy()

        with patch.object(
            strategy,
            "_run_embedding_transcription_eval",
            side_effect=RuntimeError("MTEB error"),
        ):
            with pytest.raises(RuntimeError):
                strategy.run_eval()

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_eval_with_list_scores(self, mock_mkdir, mock_file):
        strategy = self._create_strategy()

        with patch.object(strategy, "get_health", return_value=(True, "tt-embedding")):
            with patch.object(
                strategy,
                "_run_embedding_transcription_eval",
                return_value={"pearson": 0.85, "spearman": 0.82, "main_score": 0.835},
            ):
                strategy.run_eval()

        # Verify parsing handled list scores correctly
        write_calls = mock_file().write.call_args_list
        written_content = "".join(call[0][0] for call in write_calls)
        report_data = json.loads(written_content)
        eval_result = report_data[0]

        assert eval_result["pearson"] == 0.85
        assert eval_result["spearman"] == 0.82


class TestEmbeddingClientStrategyRunBenchmark(unittest.TestCase):
    """Tests for run_benchmark method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        model_spec.hf_model_repo = "test/model"
        model_spec.device_model_spec.env_vars = {"MAX_MODEL_LENGTH": "1024"}
        model_spec.device_model_spec.max_concurrency = 4
        device = MagicMock()
        device.name = "test_device"
        return EmbeddingClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.embedding_client.subprocess.run")
    @patch("utils.media_clients.embedding_client.VENV_CONFIGS")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_benchmark_success(
        self, mock_mkdir, mock_file, mock_venv_configs, mock_subprocess
    ):
        strategy = self._create_strategy()

        # Mock venv config
        mock_venv_config = MagicMock()
        mock_venv_config.venv_path = MagicMock()
        mock_venv_config.venv_path.__truediv__ = MagicMock(return_value=MagicMock())
        mock_venv_configs.get.return_value = mock_venv_config

        # Mock subprocess output with benchmark results
        benchmark_output = f"""Some preamble text
{BENCHMARK_RESULT_START}
Total input tokens: 50000
Benchmark duration: 10.5
Successful requests: 950
Failed requests: 50
Mean E2EL: 0.5
Request throughput: 90.5
{BENCHMARK_RESULT_END}
Some trailing text"""
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.stdout = benchmark_output
        mock_subprocess.run.return_value = mock_subprocess_result

        with patch.object(strategy, "get_health", return_value=(True, "tt-embedding")):
            with patch.object(
                strategy,
                "_run_embedding_transcription_benchmark",
                return_value={
                    "Total input tokens": "50000",
                    "Benchmark duration": "10.5",
                    "Successful requests": "950",
                    "Failed requests": "50",
                    "Mean E2EL": "0.5",
                    "Request throughput": "90.5",
                },
            ):
                strategy.run_benchmark()

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

        # Verify benchmarks structure and computed values
        benchmarks = report_data["benchmarks"]
        assert benchmarks["isl"] == 1024
        assert benchmarks["concurrency"] == 4
        assert benchmarks["num_requests"] == 1000
        # tput_prefill = 50000 / 10.5 = 4761.9
        assert abs(benchmarks["tput_prefill"] - 4761.9) < 0.1
        # tput_user = 4761.9 / 4 = 1190.5
        assert abs(benchmarks["tput_user"] - 1190.5) < 0.1
        assert benchmarks["e2el"] == 0.5
        assert benchmarks["req_tput"] == 90.5

        # Verify metadata
        assert report_data["model"] == "test_model"
        assert report_data["device"] == "test_device"
        assert report_data["task_type"] == "embedding"

    @patch.object(EmbeddingClientStrategy, "get_health", return_value=(False, None))
    def test_run_benchmark_health_check_failed(self, mock_health):
        strategy = self._create_strategy()

        with pytest.raises(Exception):
            strategy.run_benchmark()

    @patch.object(
        EmbeddingClientStrategy, "get_health", return_value=(True, "tt-embedding")
    )
    def test_run_benchmark_propagates_benchmark_exception(self, mock_health):
        strategy = self._create_strategy()

        with patch.object(
            strategy,
            "_run_embedding_transcription_benchmark",
            side_effect=subprocess.CalledProcessError(1, "cmd"),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                strategy.run_benchmark()


class TestEmbeddingClientStrategyParseBenchmarkOutput(unittest.TestCase):
    """Tests for _parse_embedding_benchmark_output method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        model_spec.device_model_spec.env_vars = {"MAX_MODEL_LENGTH": "1024"}
        model_spec.device_model_spec.max_concurrency = 4
        device = MagicMock()
        return EmbeddingClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_parse_benchmark_output_success(self):
        strategy = self._create_strategy()

        output = f"""
Some text before
{BENCHMARK_RESULT_START}
Total input tokens: 50000
Benchmark duration (seconds): 10.5
Successful requests: 950
Failed requests: 50
Mean E2EL: 0.5
Request throughput: 90.5
{BENCHMARK_RESULT_END}
Some text after
"""

        result = strategy._parse_embedding_benchmark_output(output)

        assert result["Total input tokens"] == "50000"
        assert result["Benchmark duration"] == "10.5"
        assert result["Successful requests"] == "950"
        assert result["Failed requests"] == "50"
        assert result["Mean E2EL"] == "0.5"
        assert result["Request throughput"] == "90.5"

    def test_parse_benchmark_output_no_start_marker(self):
        strategy = self._create_strategy()

        output = "Some text without benchmark markers"

        result = strategy._parse_embedding_benchmark_output(output)

        assert result == {}

    def test_parse_benchmark_output_empty_section(self):
        strategy = self._create_strategy()

        output = f"{BENCHMARK_RESULT_START}\n{BENCHMARK_RESULT_END}"

        result = strategy._parse_embedding_benchmark_output(output)

        assert result == {}

    def test_parse_benchmark_output_no_end_marker(self):
        strategy = self._create_strategy()

        output = f"""
{BENCHMARK_RESULT_START}
Total input tokens: 50000
Benchmark duration: 10.5
"""

        result = strategy._parse_embedding_benchmark_output(output)

        assert result["Total input tokens"] == "50000"
        assert result["Benchmark duration"] == "10.5"

    def test_parse_benchmark_output_strips_units(self):
        strategy = self._create_strategy()

        output = f"""
{BENCHMARK_RESULT_START}
Total input tokens (count): 50000
Benchmark duration (seconds): 10.5
{BENCHMARK_RESULT_END}
"""

        result = strategy._parse_embedding_benchmark_output(output)

        assert "Total input tokens" in result
        assert result["Total input tokens"] == "50000"
        assert "Benchmark duration" in result
        assert result["Benchmark duration"] == "10.5"

    def test_parse_benchmark_output_multiple_colons(self):
        strategy = self._create_strategy()

        output = f"""{BENCHMARK_RESULT_START}
Key with: colons: value here
Simple key: simple value
{BENCHMARK_RESULT_END}"""

        result = strategy._parse_embedding_benchmark_output(output)

        # The parsing uses split(":", 1) which splits on first colon only
        assert result["Key with"] == "colons: value here"
        assert result["Simple key"] == "simple value"


class TestEmbeddingClientStrategyGenerateBenchmarkingReport(unittest.TestCase):
    """Tests for _generate_benchmarking_report method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        model_spec.device_model_spec.env_vars = {"MAX_MODEL_LENGTH": "1024"}
        model_spec.device_model_spec.max_concurrency = 4
        device = MagicMock()
        device.name = "test_device"
        return EmbeddingClientStrategy({}, model_spec, device, "/tmp/output", 8000)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_benchmarking_report(self, mock_mkdir, mock_file):
        strategy = self._create_strategy()

        metrics = {
            "Total input tokens": "50000",
            "Benchmark duration": "10.5",
            "Successful requests": "950",
            "Failed requests": "50",
            "Mean E2EL": "0.5",
            "Request throughput": "90.5",
        }

        strategy._generate_benchmarking_report(metrics)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify file path pattern
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

        # Verify benchmarks structure
        benchmarks = report_data["benchmarks"]
        assert benchmarks["isl"] == 1024
        assert benchmarks["concurrency"] == 4
        assert benchmarks["num_requests"] == 1000
        assert benchmarks["e2el"] == 0.5
        assert benchmarks["req_tput"] == 90.5

        # Verify metadata
        assert report_data["model"] == "test_model"
        assert report_data["device"] == "test_device"
        assert report_data["task_type"] == "embedding"

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_benchmarking_report_zero_duration(self, mock_mkdir, mock_file):
        strategy = self._create_strategy()

        metrics = {
            "Total input tokens": "50000",
            "Benchmark duration": "0",
            "Successful requests": "950",
            "Failed requests": "50",
        }

        strategy._generate_benchmarking_report(metrics)

        write_calls = mock_file().write.call_args_list
        written_content = "".join(call[0][0] for call in write_calls)
        report_data = json.loads(written_content)

        # tput_prefill should be 0 when duration is 0
        assert report_data["benchmarks"]["tput_prefill"] == 0.0
        assert report_data["benchmarks"]["tput_user"] == 0.0

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_benchmarking_report_no_concurrency(self, mock_mkdir, mock_file):
        strategy = self._create_strategy()
        strategy.concurrency = None

        metrics = {
            "Total input tokens": "50000",
            "Benchmark duration": "10.5",
            "Successful requests": "950",
            "Failed requests": "50",
        }

        strategy._generate_benchmarking_report(metrics)

        write_calls = mock_file().write.call_args_list
        written_content = "".join(call[0][0] for call in write_calls)
        report_data = json.loads(written_content)

        # tput_user should be 0 when concurrency is None
        assert report_data["benchmarks"]["tput_user"] == 0.0


class TestEmbeddingClientStrategyParseEvalsOutput(unittest.TestCase):
    """Tests for _parse_embedding_evals_output method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        model_spec.device_model_spec.env_vars = {"MAX_MODEL_LENGTH": "1024"}
        model_spec.device_model_spec.max_concurrency = 4
        device = MagicMock()
        return EmbeddingClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_parse_evals_output_success(self):
        strategy = self._create_strategy()

        mock_task_result = MagicMock()
        mock_task_result.scores = {
            "test": {
                "pearson": 0.85,
                "spearman": 0.82,
                "cosine_pearson": 0.84,
                "cosine_spearman": 0.81,
                "manhattan_pearson": 0.83,
                "manhattan_spearman": 0.80,
                "euclidean_pearson": 0.86,
                "euclidean_spearman": 0.83,
                "main_score": 0.835,
                "languages": ["en"],
            }
        }
        mock_results = MagicMock()
        mock_results.task_results = [mock_task_result]

        result = strategy._parse_embedding_evals_output(mock_results)

        assert result["pearson"] == 0.85
        assert result["spearman"] == 0.82
        assert result["cosine_pearson"] == 0.84
        assert result["cosine_spearman"] == 0.81
        assert result["manhattan_pearson"] == 0.83
        assert result["manhattan_spearman"] == 0.80
        assert result["euclidean_pearson"] == 0.86
        assert result["euclidean_spearman"] == 0.83
        assert result["main_score"] == 0.835
        assert result["languages"] == ["en"]

    def test_parse_evals_output_with_list_scores(self):
        strategy = self._create_strategy()

        mock_task_result = MagicMock()
        mock_task_result.scores = {
            "test": [
                {
                    "pearson": 0.85,
                    "spearman": 0.82,
                    "main_score": 0.835,
                }
            ]
        }
        mock_results = MagicMock()
        mock_results.task_results = [mock_task_result]

        result = strategy._parse_embedding_evals_output(mock_results)

        assert result["pearson"] == 0.85
        assert result["spearman"] == 0.82
        assert result["main_score"] == 0.835

    def test_parse_evals_output_missing_keys(self):
        strategy = self._create_strategy()

        mock_task_result = MagicMock()
        mock_task_result.scores = {
            "test": {
                "pearson": 0.85,
                "spearman": 0.82,
                # Missing other keys
            }
        }
        mock_results = MagicMock()
        mock_results.task_results = [mock_task_result]

        result = strategy._parse_embedding_evals_output(mock_results)

        assert result["pearson"] == 0.85
        assert result["spearman"] == 0.82
        assert "cosine_pearson" not in result

    def test_parse_evals_output_exception(self):
        strategy = self._create_strategy()

        mock_results = MagicMock()
        mock_results.task_results = []  # Empty list will cause exception

        with pytest.raises(Exception):
            strategy._parse_embedding_evals_output(mock_results)


class TestEmbeddingClientStrategyGenerateEvalsReport(unittest.TestCase):
    """Tests for _generate_evals_report method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        model_spec.hf_model_repo = "org/model"
        device = MagicMock()
        device.name = "test_device"
        all_params = MagicMock()
        all_params.tasks = [MagicMock(task_name="test_task")]
        return EmbeddingClientStrategy(all_params, model_spec, device, "/tmp", 8000)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_evals_report(self, mock_mkdir, mock_file):
        strategy = self._create_strategy()

        metrics = {
            "pearson": 0.85,
            "spearman": 0.82,
            "main_score": 0.835,
        }

        strategy._generate_evals_report(metrics)

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
        ]
        for key in required_keys:
            assert key in eval_result, f"Missing required key: {key}"

        # Verify metrics are attached
        assert eval_result["pearson"] == 0.85
        assert eval_result["spearman"] == 0.82
        assert eval_result["main_score"] == 0.835

        # Verify metadata
        assert eval_result["model"] == "test_model"
        assert eval_result["device"] == "test_device"
        assert eval_result["task_type"] == "embedding"
        assert eval_result["task_name"] == "test_task"


class TestEmbeddingClientStrategyRunEmbeddingTranscriptionBenchmark(unittest.TestCase):
    """Tests for _run_embedding_transcription_benchmark method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.hf_model_repo = "test/model"
        model_spec.device_model_spec.env_vars = {"MAX_MODEL_LENGTH": "1024"}
        model_spec.device_model_spec.max_concurrency = 4
        device = MagicMock()
        return EmbeddingClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.embedding_client.subprocess.run")
    @patch("utils.media_clients.embedding_client.VENV_CONFIGS")
    @patch.dict("utils.media_clients.embedding_client.os.environ", {})
    def test_run_embedding_transcription_benchmark_success(
        self, mock_venv_configs, mock_subprocess
    ):
        strategy = self._create_strategy()

        # Mock venv config
        mock_venv_config = MagicMock()
        mock_venv_path = MagicMock()
        mock_venv_path.__truediv__ = MagicMock(return_value=MagicMock())
        mock_venv_config.venv_path = mock_venv_path
        mock_venv_configs.get.return_value = mock_venv_config

        # Mock subprocess output
        benchmark_output = (
            f"Some text\n{BENCHMARK_RESULT_START}\n"
            f"Total input tokens: 50000\n"
            f"Benchmark duration: 10.5\n"
            f"{BENCHMARK_RESULT_END}\n"
        )
        mock_result = MagicMock()
        mock_result.stdout = benchmark_output
        # mock_subprocess is the patched subprocess.run function, so set return_value directly
        mock_subprocess.return_value = mock_result

        result = strategy._run_embedding_transcription_benchmark()

        assert isinstance(result, dict)
        assert "Total input tokens" in result
        assert result["Total input tokens"] == "50000"
        mock_subprocess.assert_called_once()

    @patch("utils.media_clients.embedding_client.subprocess.run")
    @patch("utils.media_clients.embedding_client.VENV_CONFIGS")
    def test_run_embedding_transcription_benchmark_subprocess_error(
        self, mock_venv_configs, mock_subprocess
    ):
        strategy = self._create_strategy()

        mock_venv_config = MagicMock()
        mock_venv_path = MagicMock()
        mock_venv_path.__truediv__ = MagicMock(return_value=MagicMock())
        mock_venv_config.venv_path = mock_venv_path
        mock_venv_configs.get.return_value = mock_venv_config

        # subprocess.run with check=True raises CalledProcessError on failure
        # mock_subprocess is the patched subprocess.run function
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd")

        with pytest.raises(subprocess.CalledProcessError):
            strategy._run_embedding_transcription_benchmark()
