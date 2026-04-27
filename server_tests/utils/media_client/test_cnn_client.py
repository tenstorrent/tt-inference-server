# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pytest

from utils.media_clients.cnn_client import CnnClientStrategy
from utils.media_clients.test_status import CnnGenerationTestStatus


class TestCnnClientStrategyRunEval(unittest.TestCase):
    """Tests for run_eval method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        model_spec.hf_model_repo = "org/model"
        model_spec.device_model_spec.max_concurrency = 1
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
        return CnnClientStrategy(all_params, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.cnn_client.get_num_calls", return_value=2)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_eval_success(self, mock_mkdir, mock_file, mock_num_calls):
        strategy = self._create_strategy()
        # Multiple status entries to verify TTFT averaging
        status_list = [
            CnnGenerationTestStatus(status=True, elapsed=1.0),
            CnnGenerationTestStatus(status=True, elapsed=2.0),
        ]

        with patch.object(strategy, "get_health", return_value=(True, "tt-resnet")):
            with patch.object(
                strategy,
                "_run_image_analysis_benchmark",
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

        # Verify timestamp exists (dynamic field)
        assert "timestamp" in eval_result

        # Compare all static fields
        expected = {
            "model": "test_model",
            "device": "test_device",
            "task_type": "cnn",
            "task_name": "test_task",
            "tolerance": 0.1,
            "published_score": 0.9,
            "published_score_ref": "ref",
            "score": 1.5,  # TTFT average: (1.0 + 2.0) / 2
            # accuracy_check derived from API success rate (PASS=2 per ReportCheckTypes)
            "accuracy_check": 2,
            # tput_user = 1 / mean_ttft = 1 / 1.5
            "tput_user": pytest.approx(1.0 / 1.5),
        }
        for key, value in expected.items():
            assert eval_result[key] == value, f"Mismatch for {key}"

    @patch.object(CnnClientStrategy, "get_health", return_value=(False, None))
    def test_run_eval_health_check_failed(self, mock_health):
        strategy = self._create_strategy()

        with pytest.raises(Exception):
            strategy.run_eval()

    @patch("utils.media_clients.cnn_client.get_num_calls", return_value=2)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_eval_accuracy_check_fail_on_api_failures(
        self, mock_mkdir, mock_file, mock_num_calls
    ):
        """Non-MobileNetV2 path: accuracy_check should be FAIL (3) if any API call failed."""
        strategy = self._create_strategy()
        status_list = [
            CnnGenerationTestStatus(status=True, elapsed=1.0),
            CnnGenerationTestStatus(status=False, elapsed=2.0),
        ]

        with patch.object(strategy, "get_health", return_value=(True, "tt-resnet")):
            with patch.object(
                strategy,
                "_run_image_analysis_benchmark",
                return_value=status_list,
            ):
                strategy.run_eval()

        write_calls = mock_file().write.call_args_list
        written_content = "".join(call[0][0] for call in write_calls)
        eval_result = json.loads(written_content)[0]
        # PASS=2, FAIL=3 per ReportCheckTypes
        assert eval_result["accuracy_check"] == 3

    @patch("utils.media_clients.cnn_client.get_num_calls", return_value=1)
    def test_run_eval_propagates_benchmark_exception(self, mock_num_calls):
        strategy = self._create_strategy()

        with patch.object(strategy, "get_health", return_value=(True, "tt-resnet")):
            with patch.object(
                strategy,
                "_run_image_analysis_benchmark",
                side_effect=RuntimeError("Error"),
            ):
                with pytest.raises(RuntimeError):
                    strategy.run_eval()

    @patch("utils.media_clients.cnn_client.get_num_calls", return_value=2)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_eval_vision_eval_path_writes_both_accuracy_and_tput_user(
        self, mock_mkdir, mock_file, mock_num_calls
    ):
        """VisionEvalsTest-supported runner: eval JSON must include the
        accuracy fields from VisionEvalsTest *and* tput_user computed from
        the benchmark loop. The latter is what flips tput_check from FAIL
        to PASS in run_reports.add_target_checks_cnn_image_video."""
        strategy = self._create_strategy()
        status_list = [
            CnnGenerationTestStatus(status=True, elapsed=0.05),
            CnnGenerationTestStatus(status=True, elapsed=0.05),
        ]
        eval_result = {
            "accuracy_status": 2,
            "correct": 16,
            "total": 20,
            "mismatches_count": 4,
        }

        with patch.object(
            strategy, "get_health", return_value=(True, "tt-xla-efficientnet")
        ):
            with patch.object(
                strategy, "_run_image_analysis_benchmark", return_value=status_list
            ):
                with patch.object(
                    strategy, "_run_vision_eval", return_value=eval_result
                ) as mock_vision_eval:
                    strategy.run_eval()

        # Vision eval was invoked with the actual runner name
        mock_vision_eval.assert_called_once_with("tt-xla-efficientnet")

        write_calls = mock_file().write.call_args_list
        written_content = "".join(call[0][0] for call in write_calls)
        eval_row = json.loads(written_content)[0]

        # Accuracy fields come from VisionEvalsTest
        assert eval_row["accuracy_check"] == 2
        assert eval_row["correct"] == 16
        assert eval_row["total"] == 20
        assert eval_row["mismatches_count"] == 4
        # TTFT/tput_user come from the always-run benchmark loop
        assert eval_row["score"] == pytest.approx(0.05)
        assert eval_row["tput_user"] == pytest.approx(1.0 / 0.05)


class TestCnnClientStrategyRunBenchmark(unittest.TestCase):
    """Tests for run_benchmark method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        device = MagicMock()
        device.name = "test_device"
        return CnnClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.cnn_client.get_num_calls", return_value=2)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_benchmark_success(self, mock_mkdir, mock_file, mock_num_calls):
        strategy = self._create_strategy()
        # Multiple status entries to verify averaging
        status_list = [
            CnnGenerationTestStatus(
                status=True,
                elapsed=1.0,
                num_inference_steps=50,
                inference_steps_per_second=50.0,
            ),
            CnnGenerationTestStatus(
                status=True,
                elapsed=2.0,
                num_inference_steps=50,
                inference_steps_per_second=25.0,
            ),
        ]

        with patch.object(strategy, "get_health", return_value=(True, "tt-resnet")):
            with patch.object(
                strategy,
                "_run_image_analysis_benchmark",
                return_value=status_list,
            ):
                strategy.run_benchmark(2)

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

        # Verify timestamp exists (dynamic field)
        assert "timestamp" in report_data

        # Compare static top-level fields
        expected_metadata = {
            "model": "test_model",
            "device": "test_device",
            "task_type": "cnn",
        }
        for key, value in expected_metadata.items():
            assert report_data[key] == value

        # Compare benchmarks structure: throughput now derived as 1 / mean_ttft
        # (CNN classifiers have no inference-steps concept)
        expected_benchmarks = {
            "num_requests": 2,
            "num_inference_steps": 50,
            "ttft": 1.5,  # (1.0 + 2.0) / 2
            "tput_user": pytest.approx(1.0 / 1.5),
            "inference_steps_per_second": pytest.approx(1.0 / 1.5),
        }
        for key, value in expected_benchmarks.items():
            assert report_data["benchmarks"][key] == value

    @patch.object(CnnClientStrategy, "get_health", return_value=(False, None))
    def test_run_benchmark_health_check_failed(self, mock_health):
        strategy = self._create_strategy()

        with pytest.raises(Exception):
            strategy.run_benchmark(2)

    @patch("utils.media_clients.cnn_client.get_num_calls", return_value=1)
    def test_run_benchmark_propagates_benchmark_exception(self, mock_num_calls):
        strategy = self._create_strategy()

        with patch.object(strategy, "get_health", return_value=(True, "tt-resnet")):
            with patch.object(
                strategy,
                "_run_image_analysis_benchmark",
                side_effect=RuntimeError("Error"),
            ):
                with pytest.raises(RuntimeError):
                    strategy.run_benchmark(1)


class TestCnnClientStrategyAnalyzeImage(unittest.TestCase):
    """Tests for _analyze_image method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return CnnClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("builtins.open", mock_open(read_data="base64imagedata"))
    @patch("utils.media_clients.cnn_client.requests.post")
    def test_analyze_image_success(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        status, elapsed = strategy._analyze_image()

        assert status is True
        assert elapsed > 0

    @patch("builtins.open", mock_open(read_data="base64imagedata"))
    @patch("utils.media_clients.cnn_client.requests.post")
    def test_analyze_image_failure(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        status, elapsed = strategy._analyze_image()

        assert status is False
        assert elapsed > 0


class TestCnnClientStrategyRunImageAnalysisBenchmark(unittest.TestCase):
    """Tests for _run_image_analysis_benchmark method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return CnnClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch.object(CnnClientStrategy, "_analyze_image", return_value=(True, 1.5))
    def test_run_image_analysis_benchmark(self, mock_analyze):
        strategy = self._create_strategy()

        result = strategy._run_image_analysis_benchmark(3)

        # num_calls measurements + 1 warmup that is excluded from results
        assert len(result) == 3
        assert all(isinstance(s, CnnGenerationTestStatus) for s in result)
        assert mock_analyze.call_count == 4

    @patch.object(CnnClientStrategy, "_analyze_image", return_value=(True, 0.5))
    def test_run_image_analysis_benchmark_single_call(self, mock_analyze):
        strategy = self._create_strategy()

        result = strategy._run_image_analysis_benchmark(1)

        assert len(result) == 1
        assert result[0].elapsed == 0.5
        # 1 measurement + 1 warmup
        assert mock_analyze.call_count == 2


class TestCnnClientStrategyGenerateReport(unittest.TestCase):
    """Tests for _generate_report method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        device = MagicMock()
        device.name = "test_device"
        return CnnClientStrategy({}, model_spec, device, "/tmp/output", 8000)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_report_with_status_list(self, mock_mkdir, mock_file):
        strategy = self._create_strategy()
        status_list = [
            CnnGenerationTestStatus(
                status=True,
                elapsed=1.0,
                num_inference_steps=50,
                inference_steps_per_second=50.0,
            ),
            CnnGenerationTestStatus(
                status=True,
                elapsed=2.0,
                num_inference_steps=50,
                inference_steps_per_second=25.0,
            ),
        ]

        strategy._generate_report(status_list)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify file path pattern
        open_call_args = mock_file.call_args[0][0]
        path_str = str(open_call_args)
        assert path_str.startswith("/tmp/output/benchmark_test_id_")
        assert path_str.endswith(".json")

        # Verify JSON content
        write_calls = mock_file().write.call_args_list
        written_content = "".join(call[0][0] for call in write_calls)
        report_data = json.loads(written_content)

        # Compare expected values
        assert report_data["model"] == "test_model"
        assert report_data["task_type"] == "cnn"
        expected_benchmarks = {
            "num_requests": 2,
            "ttft": 1.5,
            "tput_user": pytest.approx(1.0 / 1.5),
            "inference_steps_per_second": pytest.approx(1.0 / 1.5),
        }
        for key, value in expected_benchmarks.items():
            assert report_data["benchmarks"][key] == value

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_report_empty_status_list(self, mock_mkdir, mock_file):
        strategy = self._create_strategy()

        strategy._generate_report([])

        # Verify JSON content handles empty list
        write_calls = mock_file().write.call_args_list
        written_content = "".join(call[0][0] for call in write_calls)
        report_data = json.loads(written_content)

        expected_benchmarks = {
            "num_requests": 0,
            "ttft": 0,
            "tput_user": 0,
            "inference_steps_per_second": 0,
        }
        for key, value in expected_benchmarks.items():
            assert report_data["benchmarks"][key] == value


class TestCnnClientStrategyCalculateTtft(unittest.TestCase):
    """Tests for _calculate_ttft_value method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        device = MagicMock()
        return CnnClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_calculate_ttft_with_status_list(self):
        strategy = self._create_strategy()
        status_list = [
            CnnGenerationTestStatus(status=True, elapsed=1.0),
            CnnGenerationTestStatus(status=True, elapsed=2.0),
            CnnGenerationTestStatus(status=True, elapsed=3.0),
        ]
        result = strategy._calculate_ttft_value(status_list)
        assert result == 2.0

    def test_calculate_ttft_empty_list(self):
        strategy = self._create_strategy()
        result = strategy._calculate_ttft_value([])
        assert result == 0

    def test_calculate_ttft_single_item(self):
        strategy = self._create_strategy()
        status_list = [CnnGenerationTestStatus(status=True, elapsed=5.0)]
        result = strategy._calculate_ttft_value(status_list)
        assert result == 5.0


# Parametrized tests
@pytest.mark.parametrize(
    "num_calls",
    [1, 2, 5, 10],
)
@patch.object(CnnClientStrategy, "_analyze_image", return_value=(True, 1.0))
def test_run_image_analysis_various_num_calls(mock_analyze, num_calls):
    """Test that benchmark runs correct number of iterations (plus one warmup)."""
    model_spec = MagicMock()
    model_spec.model_name = "test"
    device = MagicMock()
    device.name = "test"
    strategy = CnnClientStrategy({}, model_spec, device, "/tmp", 8000)

    result = strategy._run_image_analysis_benchmark(num_calls)

    # Warmup is excluded from results but counted in calls to _analyze_image
    assert len(result) == num_calls
    assert mock_analyze.call_count == num_calls + 1


@pytest.mark.parametrize(
    "status_code,expected_status",
    [
        (200, True),
        (201, False),
        (400, False),
        (404, False),
        (500, False),
        (503, False),
    ],
)
@patch("builtins.open", mock_open(read_data="base64imagedata"))
@patch("utils.media_clients.cnn_client.requests.post")
def test_analyze_image_various_status_codes(mock_post, status_code, expected_status):
    """Test _analyze_image handles various HTTP status codes correctly."""
    model_spec = MagicMock()
    model_spec.model_name = "test"
    device = MagicMock()
    device.name = "test"
    strategy = CnnClientStrategy({}, model_spec, device, "/tmp", 8000)

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_post.return_value = mock_response

    status, elapsed = strategy._analyze_image()

    assert status is expected_status
