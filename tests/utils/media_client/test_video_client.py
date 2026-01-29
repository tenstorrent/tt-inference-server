# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pytest

from utils.media_clients.test_status import VideoGenerationTestStatus
from utils.media_clients.video_client import VideoClientStrategy


class TestVideoClientStrategyRunEval(unittest.TestCase):
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
        return VideoClientStrategy(all_params, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.video_client.get_num_calls", return_value=2)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_eval_success(self, mock_mkdir, mock_file, mock_num_calls):
        strategy = self._create_strategy()
        # Multiple status entries to verify TTFT averaging
        status_list = [
            VideoGenerationTestStatus(
                status=True,
                elapsed=60.0,
                num_inference_steps=20,
                inference_steps_per_second=0.33,
                job_id="job1",
                video_path="/tmp/job1.mp4",
                prompt="Test video 1",
            ),
            VideoGenerationTestStatus(
                status=True,
                elapsed=70.0,
                num_inference_steps=20,
                inference_steps_per_second=0.29,
                job_id="job2",
                video_path="/tmp/job2.mp4",
                prompt="Test video 2",
            ),
        ]

        with patch.object(strategy, "get_health", return_value=(True, "tt-mochi")):
            with patch.object(
                strategy,
                "_run_video_generation_benchmark",
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
            "task_type": "video",
            "task_name": "test_task",
            "tolerance": 0.1,
            "published_score": 0.9,
            "published_score_ref": "ref",
            "score": 65.0,  # TTFT average: (60.0 + 70.0) / 2
        }
        for key, value in expected.items():
            assert eval_result[key] == value, f"Mismatch for {key}"

    @patch.object(VideoClientStrategy, "get_health", return_value=(False, None))
    def test_run_eval_health_check_failed(self, mock_health):
        strategy = self._create_strategy()

        with pytest.raises(Exception):
            strategy.run_eval()

    @patch("utils.media_clients.video_client.get_num_calls", return_value=1)
    def test_run_eval_propagates_benchmark_exception(self, mock_num_calls):
        strategy = self._create_strategy()

        with patch.object(strategy, "get_health", return_value=(True, "tt-mochi")):
            with patch.object(
                strategy,
                "_run_video_generation_benchmark",
                side_effect=RuntimeError("Error"),
            ):
                with pytest.raises(RuntimeError):
                    strategy.run_eval()


class TestVideoClientStrategyRunBenchmark(unittest.TestCase):
    """Tests for run_benchmark method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        device = MagicMock()
        device.name = "test_device"
        return VideoClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.video_client.get_num_calls", return_value=2)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_benchmark_success(self, mock_mkdir, mock_file, mock_num_calls):
        strategy = self._create_strategy()
        # Multiple status entries to verify averaging
        status_list = [
            VideoGenerationTestStatus(
                status=True,
                elapsed=60.0,
                num_inference_steps=20,
                inference_steps_per_second=0.33,
                job_id="job1",
                video_path="/tmp/job1.mp4",
                prompt="Test video 1",
            ),
            VideoGenerationTestStatus(
                status=True,
                elapsed=80.0,
                num_inference_steps=20,
                inference_steps_per_second=0.25,
                job_id="job2",
                video_path="/tmp/job2.mp4",
                prompt="Test video 2",
            ),
        ]

        with patch.object(strategy, "get_health", return_value=(True, "tt-mochi")):
            with patch.object(
                strategy,
                "_run_video_generation_benchmark",
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
            "task_type": "video",
        }
        for key, value in expected_metadata.items():
            assert report_data[key] == value

        # Compare benchmarks structure
        assert report_data["benchmarks"]["num_requests"] == 2
        assert report_data["benchmarks"]["num_inference_steps"] == 20
        assert report_data["benchmarks"]["ttft"] == pytest.approx(70.0)
        assert report_data["benchmarks"]["inference_steps_per_second"] == pytest.approx(
            0.29
        )

    @patch.object(VideoClientStrategy, "get_health", return_value=(False, None))
    def test_run_benchmark_health_check_failed(self, mock_health):
        strategy = self._create_strategy()

        with pytest.raises(Exception):
            strategy.run_benchmark(2)

    @patch("utils.media_clients.video_client.get_num_calls", return_value=1)
    def test_run_benchmark_propagates_benchmark_exception(self, mock_num_calls):
        strategy = self._create_strategy()

        with patch.object(strategy, "get_health", return_value=(True, "tt-mochi")):
            with patch.object(
                strategy,
                "_run_video_generation_benchmark",
                side_effect=RuntimeError("Error"),
            ):
                with pytest.raises(RuntimeError):
                    strategy.run_benchmark(1)


class TestVideoClientStrategyGenerateVideo(unittest.TestCase):
    """Tests for _generate_video method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return VideoClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.video_client.requests.post")
    def test_generate_video_success(self, mock_post):
        strategy = self._create_strategy()

        # Mock job submission response
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {"id": "job123"}
        mock_post.return_value = mock_response

        with patch.object(
            strategy, "_poll_video_completion", return_value="/tmp/video.mp4"
        ):
            status, elapsed, job_id, video_path = strategy._generate_video(
                "Test prompt"
            )

        assert status is True
        assert elapsed > 0
        assert job_id == "job123"
        assert video_path == "/tmp/video.mp4"

    @patch("utils.media_clients.video_client.requests.post")
    def test_generate_video_submission_failure(self, mock_post):
        strategy = self._create_strategy()

        # Mock failed job submission
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        status, elapsed, job_id, video_path = strategy._generate_video("Test prompt")

        assert status is False
        assert elapsed > 0
        assert job_id == ""
        assert video_path == ""

    @patch("utils.media_clients.video_client.requests.post")
    def test_generate_video_polling_failure(self, mock_post):
        strategy = self._create_strategy()

        # Mock successful job submission
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {"id": "job123"}
        mock_post.return_value = mock_response

        # Mock failed polling (returns empty path)
        with patch.object(strategy, "_poll_video_completion", return_value=""):
            status, elapsed, job_id, video_path = strategy._generate_video(
                "Test prompt"
            )

        assert status is False
        assert job_id == "job123"
        assert video_path == ""

    @patch("utils.media_clients.video_client.requests.post")
    def test_generate_video_exception(self, mock_post):
        strategy = self._create_strategy()

        # Mock exception during request
        mock_post.side_effect = Exception("Network error")

        status, elapsed, job_id, video_path = strategy._generate_video("Test prompt")

        assert status is False
        assert elapsed > 0
        assert job_id == ""
        assert video_path == ""


class TestVideoClientStrategyPollVideoCompletion(unittest.TestCase):
    """Tests for _poll_video_completion method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return VideoClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.video_client.requests.get")
    @patch("utils.media_clients.video_client.time.sleep")
    def test_poll_video_completion_success(self, mock_sleep, mock_get):
        strategy = self._create_strategy()

        # Mock completed status response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "completed"}
        mock_get.return_value = mock_response

        with patch.object(strategy, "_download_video", return_value="/tmp/video.mp4"):
            video_path = strategy._poll_video_completion(
                "job123", {}, polling_interval=1, timeout=10
            )

        assert video_path == "/tmp/video.mp4"
        # Should not sleep if completed immediately
        assert mock_sleep.call_count == 0

    @patch("utils.media_clients.video_client.requests.get")
    @patch("utils.media_clients.video_client.time.sleep")
    def test_poll_video_completion_processing_then_complete(self, mock_sleep, mock_get):
        strategy = self._create_strategy()

        # Mock processing then completed
        responses = [
            MagicMock(status_code=200, json=lambda: {"status": "processing"}),
            MagicMock(status_code=200, json=lambda: {"status": "processing"}),
            MagicMock(status_code=200, json=lambda: {"status": "completed"}),
        ]
        mock_get.side_effect = responses

        with patch.object(strategy, "_download_video", return_value="/tmp/video.mp4"):
            video_path = strategy._poll_video_completion(
                "job123", {}, polling_interval=1, timeout=10
            )

        assert video_path == "/tmp/video.mp4"
        assert mock_sleep.call_count == 2

    @patch("utils.media_clients.video_client.requests.get")
    @patch("utils.media_clients.video_client.time.sleep")
    def test_poll_video_completion_failed_status(self, mock_sleep, mock_get):
        strategy = self._create_strategy()

        # Mock failed status response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "failed"}
        mock_get.return_value = mock_response

        video_path = strategy._poll_video_completion(
            "job123", {}, polling_interval=1, timeout=10
        )

        assert video_path == ""

    @patch("utils.media_clients.video_client.requests.get")
    @patch("utils.media_clients.video_client.time.sleep")
    def test_poll_video_completion_cancelled_status(self, mock_sleep, mock_get):
        strategy = self._create_strategy()

        # Mock cancelled status response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "cancelled"}
        mock_get.return_value = mock_response

        video_path = strategy._poll_video_completion(
            "job123", {}, polling_interval=1, timeout=10
        )

        assert video_path == ""

    @patch("utils.media_clients.video_client.requests.get")
    @patch("utils.media_clients.video_client.time.sleep")
    def test_poll_video_completion_timeout(self, mock_sleep, mock_get):
        strategy = self._create_strategy()

        # Mock timeout scenario - use a class to track time
        class TimeTracker:
            def __init__(self):
                self.current = 0

            def __call__(self):
                result = self.current
                self.current += 6  # Increment by 6 seconds each call
                return result

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "processing"}
        mock_get.return_value = mock_response

        with patch("utils.media_clients.video_client.time.time", TimeTracker()):
            video_path = strategy._poll_video_completion(
                "job123", {}, polling_interval=1, timeout=10
            )

        assert video_path == ""

    @patch("utils.media_clients.video_client.requests.get")
    @patch("utils.media_clients.video_client.time.sleep")
    def test_poll_video_completion_http_error(self, mock_sleep, mock_get):
        strategy = self._create_strategy()

        # Mock HTTP error then success
        responses = [
            MagicMock(status_code=500),
            MagicMock(status_code=200, json=lambda: {"status": "completed"}),
        ]
        mock_get.side_effect = responses

        with patch.object(strategy, "_download_video", return_value="/tmp/video.mp4"):
            video_path = strategy._poll_video_completion(
                "job123", {}, polling_interval=1, timeout=10
            )

        assert video_path == "/tmp/video.mp4"
        assert mock_sleep.call_count == 1

    @patch("utils.media_clients.video_client.requests.get")
    @patch("utils.media_clients.video_client.time.sleep")
    def test_poll_video_completion_exception_during_polling(self, mock_sleep, mock_get):
        strategy = self._create_strategy()

        # Mock exception then success
        responses = [
            Exception("Network error"),
            MagicMock(status_code=200, json=lambda: {"status": "completed"}),
        ]
        mock_get.side_effect = responses

        with patch.object(strategy, "_download_video", return_value="/tmp/video.mp4"):
            video_path = strategy._poll_video_completion(
                "job123", {}, polling_interval=1, timeout=10
            )

        assert video_path == "/tmp/video.mp4"
        assert mock_sleep.call_count == 1


class TestVideoClientStrategyDownloadVideo(unittest.TestCase):
    """Tests for _download_video method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return VideoClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.video_client.requests.get")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_download_video_success(self, mock_mkdir, mock_file, mock_get):
        strategy = self._create_strategy()

        # Mock successful download
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"video", b"data"]
        mock_get.return_value = mock_response

        video_path = strategy._download_video("job123", {})

        assert video_path == "/tmp/output/videos/job123.mp4"
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify file was written
        assert mock_file().write.call_count == 2

    @patch("utils.media_clients.video_client.requests.get")
    @patch("pathlib.Path.mkdir")
    def test_download_video_http_error(self, mock_mkdir, mock_get):
        strategy = self._create_strategy()

        # Mock HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        video_path = strategy._download_video("job123", {})

        assert video_path == ""

    @patch("utils.media_clients.video_client.requests.get")
    @patch("pathlib.Path.mkdir")
    def test_download_video_exception(self, mock_mkdir, mock_get):
        strategy = self._create_strategy()

        # Mock exception during download
        mock_get.side_effect = Exception("Network error")

        video_path = strategy._download_video("job123", {})

        assert video_path == ""


class TestVideoClientStrategyRunVideoGenerationBenchmark(unittest.TestCase):
    """Tests for _run_video_generation_benchmark method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return VideoClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.video_client.INFERENCE_STEPS", {"test_model": 20})
    @patch.object(
        VideoClientStrategy,
        "_generate_video",
        return_value=(True, 60.0, "job1", "/tmp/video1.mp4"),
    )
    def test_run_video_generation_benchmark(self, mock_generate):
        strategy = self._create_strategy()

        result = strategy._run_video_generation_benchmark(3)

        assert len(result) == 3
        assert all(isinstance(s, VideoGenerationTestStatus) for s in result)
        assert mock_generate.call_count == 3

    @patch("utils.media_clients.video_client.INFERENCE_STEPS", {"test_model": 20})
    @patch.object(
        VideoClientStrategy,
        "_generate_video",
        return_value=(True, 50.0, "job1", "/tmp/video1.mp4"),
    )
    def test_run_video_generation_benchmark_single_call(self, mock_generate):
        strategy = self._create_strategy()

        result = strategy._run_video_generation_benchmark(1)

        assert len(result) == 1
        assert result[0].elapsed == 50.0
        assert result[0].job_id == "job1"
        assert result[0].video_path == "/tmp/video1.mp4"


class TestVideoClientStrategyGenerateReport(unittest.TestCase):
    """Tests for _generate_report method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        device = MagicMock()
        device.name = "test_device"
        return VideoClientStrategy({}, model_spec, device, "/tmp/output", 8000)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_report_with_status_list(self, mock_mkdir, mock_file):
        strategy = self._create_strategy()
        status_list = [
            VideoGenerationTestStatus(
                status=True,
                elapsed=60.0,
                num_inference_steps=20,
                inference_steps_per_second=0.33,
                job_id="job1",
                video_path="/tmp/job1.mp4",
                prompt="Test video 1",
            ),
            VideoGenerationTestStatus(
                status=True,
                elapsed=80.0,
                num_inference_steps=20,
                inference_steps_per_second=0.25,
                job_id="job2",
                video_path="/tmp/job2.mp4",
                prompt="Test video 2",
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
        assert report_data["task_type"] == "video"
        assert report_data["benchmarks"]["num_requests"] == 2
        assert report_data["benchmarks"]["ttft"] == pytest.approx(70.0)
        assert report_data["benchmarks"]["inference_steps_per_second"] == pytest.approx(
            0.29
        )

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
            "inference_steps_per_second": 0,
        }
        for key, value in expected_benchmarks.items():
            assert report_data["benchmarks"][key] == value


class TestVideoClientStrategyCalculateTtft(unittest.TestCase):
    """Tests for _calculate_ttft_value method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        device = MagicMock()
        return VideoClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_calculate_ttft_with_status_list(self):
        strategy = self._create_strategy()
        status_list = [
            VideoGenerationTestStatus(
                status=True,
                elapsed=60.0,
                num_inference_steps=20,
                inference_steps_per_second=0.33,
                job_id="job1",
                video_path="/tmp/job1.mp4",
                prompt="Test 1",
            ),
            VideoGenerationTestStatus(
                status=True,
                elapsed=80.0,
                num_inference_steps=20,
                inference_steps_per_second=0.25,
                job_id="job2",
                video_path="/tmp/job2.mp4",
                prompt="Test 2",
            ),
            VideoGenerationTestStatus(
                status=True,
                elapsed=70.0,
                num_inference_steps=20,
                inference_steps_per_second=0.29,
                job_id="job3",
                video_path="/tmp/job3.mp4",
                prompt="Test 3",
            ),
        ]
        result = strategy._calculate_ttft_value(status_list)
        assert result == 70.0  # (60 + 80 + 70) / 3

    def test_calculate_ttft_empty_list(self):
        strategy = self._create_strategy()
        result = strategy._calculate_ttft_value([])
        assert result == 0

    def test_calculate_ttft_single_item(self):
        strategy = self._create_strategy()
        status_list = [
            VideoGenerationTestStatus(
                status=True,
                elapsed=90.0,
                num_inference_steps=20,
                inference_steps_per_second=0.22,
                job_id="job1",
                video_path="/tmp/job1.mp4",
                prompt="Test",
            )
        ]
        result = strategy._calculate_ttft_value(status_list)
        assert result == 90.0


# Parametrized tests
@pytest.mark.parametrize(
    "num_calls",
    [1, 2, 5, 10],
)
@patch("utils.media_clients.video_client.INFERENCE_STEPS", {"test": 20})
@patch.object(
    VideoClientStrategy,
    "_generate_video",
    return_value=(True, 60.0, "job1", "/tmp/video.mp4"),
)
def test_run_video_generation_various_num_calls(mock_generate, num_calls):
    """Test that benchmark runs correct number of iterations."""
    model_spec = MagicMock()
    model_spec.model_name = "test"
    device = MagicMock()
    device.name = "test"
    strategy = VideoClientStrategy({}, model_spec, device, "/tmp", 8000)

    result = strategy._run_video_generation_benchmark(num_calls)

    assert len(result) == num_calls
    assert mock_generate.call_count == num_calls


@pytest.mark.parametrize(
    "status_code,expected_status",
    [
        (202, True),  # Video API returns 202 for job acceptance
        (200, False),
        (400, False),
        (404, False),
        (500, False),
        (503, False),
    ],
)
@patch("utils.media_clients.video_client.requests.post")
def test_generate_video_various_status_codes(mock_post, status_code, expected_status):
    """Test _generate_video handles various HTTP status codes correctly."""
    model_spec = MagicMock()
    model_spec.model_name = "test"
    device = MagicMock()
    device.name = "test"
    strategy = VideoClientStrategy({}, model_spec, device, "/tmp", 8000)

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {"id": "job123"}
    mock_post.return_value = mock_response

    if expected_status:
        with patch.object(
            strategy, "_poll_video_completion", return_value="/tmp/video.mp4"
        ):
            status, elapsed, job_id, video_path = strategy._generate_video("Test")
    else:
        status, elapsed, job_id, video_path = strategy._generate_video("Test")

    assert status is expected_status


@pytest.mark.parametrize(
    "video_status,expected_result",
    [
        ("completed", True),
        ("failed", False),
        ("cancelled", False),
        ("processing", False),  # Will timeout in test
    ],
)
@patch("utils.media_clients.video_client.requests.get")
@patch("utils.media_clients.video_client.time.sleep")
def test_poll_video_various_statuses(
    mock_sleep, mock_get, video_status, expected_result
):
    """Test _poll_video_completion handles various job statuses correctly."""
    model_spec = MagicMock()
    model_spec.model_name = "test"
    device = MagicMock()
    device.name = "test"
    strategy = VideoClientStrategy({}, model_spec, device, "/tmp", 8000)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": video_status}
    mock_get.return_value = mock_response

    # Use a class to track time that works for all status types
    class TimeTracker:
        def __init__(self, increment):
            self.current = 0
            self.increment = increment

        def __call__(self):
            result = self.current
            self.current += self.increment
            return result

    # For processing, increment enough to timeout; for others, small increment
    time_increment = 6 if video_status == "processing" else 0.1

    if expected_result:
        with patch(
            "utils.media_clients.video_client.time.time", TimeTracker(time_increment)
        ):
            with patch.object(
                strategy, "_download_video", return_value="/tmp/video.mp4"
            ):
                video_path = strategy._poll_video_completion(
                    "job123", {}, polling_interval=1, timeout=10
                )
                assert video_path == "/tmp/video.mp4"
    else:
        with patch(
            "utils.media_clients.video_client.time.time", TimeTracker(time_increment)
        ):
            video_path = strategy._poll_video_completion(
                "job123", {}, polling_interval=1, timeout=10
            )
            assert video_path == ""
