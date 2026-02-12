# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

# Mock ML dependencies before importing modules that depend on them
sys.modules["open_clip"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.model_zoo"] = MagicMock()
sys.modules["torchvision"] = MagicMock()
sys.modules["torchvision.transforms"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.linalg"] = MagicMock()

from utils.media_clients.image_client import (
    ImageClientStrategy,
)
from utils.media_clients.test_status import ImageGenerationTestStatus


class TestImageClientStrategyRunEval(unittest.TestCase):
    """Tests for run_eval method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        model_spec.hf_model_repo = "org/model"
        model_spec.device_model_spec.max_concurrency = 4
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
        return ImageClientStrategy(all_params, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.image_client.calculate_accuracy_check")
    @patch("utils.media_clients.image_client.calculate_metrics")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_eval_success(self, mock_mkdir, mock_file, mock_metrics, mock_accuracy):
        strategy = self._create_strategy()
        # fid_score, average_clip_score, deviation_clip_score
        mock_metrics.return_value = (15.5, 0.85, 0.03)
        mock_accuracy.return_value = 2  # PASS

        # Multiple status entries to verify TTFT averaging
        status_list = [
            MagicMock(elapsed=1.0),
            MagicMock(elapsed=2.0),
        ]
        total_time = 3.0

        with patch.object(strategy, "get_health", return_value=(True, "tt-sdxl-trace")):
            with patch("asyncio.run", return_value=(status_list, total_time)):
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
            "fid_score",
            "average_clip",
            "deviation_clip_score",
            "accuracy_check",
            "tput_user",
        ]
        for key in required_keys:
            assert key in eval_result, f"Missing required key: {key}"

        # Verify calculated TTFT average: (1.0 + 2.0) / 2 = 1.5
        assert eval_result["score"] == 1.5

        # Verify metrics from calculate_metrics mock
        assert eval_result["fid_score"] == 15.5
        assert eval_result["average_clip"] == 0.85
        assert eval_result["deviation_clip_score"] == 0.03
        assert eval_result["accuracy_check"] == 2

        # tput_user = 2 / (3.0 * 4) = 0.1667
        assert abs(eval_result["tput_user"] - 0.1667) < 0.001

        # Verify metadata from model_spec and all_params
        assert eval_result["model"] == "test_model"
        assert eval_result["device"] == "test_device"
        assert eval_result["task_type"] == "image"
        assert eval_result["task_name"] == "test_task"
        assert eval_result["tolerance"] == 0.1
        assert eval_result["published_score"] == 0.9
        assert eval_result["published_score_ref"] == "ref"

    @patch.object(ImageClientStrategy, "get_health", return_value=(False, None))
    def test_run_eval_health_check_failed(self, mock_health):
        strategy = self._create_strategy()

        with pytest.raises(Exception):
            strategy.run_eval()

    @patch.object(
        ImageClientStrategy, "get_health", return_value=(True, "tt-sdxl-trace")
    )
    def test_run_eval_propagates_eval_exception(self, mock_health):
        strategy = self._create_strategy()

        with patch("asyncio.run", side_effect=RuntimeError("Eval error")):
            with pytest.raises(RuntimeError):
                strategy.run_eval()

    @patch("utils.media_clients.image_client.calculate_accuracy_check")
    @patch("utils.media_clients.image_client.calculate_metrics")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_eval_routes_to_img2img(
        self, mock_mkdir, mock_file, mock_metrics, mock_accuracy
    ):
        strategy = self._create_strategy()
        mock_metrics.return_value = (10.0, 0.8, 0.05)
        mock_accuracy.return_value = True
        mock_status = MagicMock(elapsed=1.5)

        with patch.object(
            strategy, "get_health", return_value=(True, "tt-sdxl-image-to-image")
        ):
            with patch("asyncio.run", return_value=([mock_status], 2.0)):
                strategy.run_eval()

    @patch("utils.media_clients.image_client.calculate_accuracy_check")
    @patch("utils.media_clients.image_client.calculate_metrics")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_eval_routes_to_inpainting(
        self, mock_mkdir, mock_file, mock_metrics, mock_accuracy
    ):
        strategy = self._create_strategy()
        mock_metrics.return_value = (10.0, 0.8, 0.05)
        mock_accuracy.return_value = True
        mock_status = MagicMock(elapsed=1.5)

        with patch.object(strategy, "get_health", return_value=(True, "tt-sdxl-edit")):
            with patch("asyncio.run", return_value=([mock_status], 2.0)):
                strategy.run_eval()

    @patch("utils.media_clients.image_client.calculate_accuracy_check")
    @patch("utils.media_clients.image_client.calculate_metrics")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_eval_no_device_spec(
        self, mock_mkdir, mock_file, mock_metrics, mock_accuracy
    ):
        strategy = self._create_strategy()
        strategy.model_spec.device_model_spec = None
        mock_metrics.return_value = (10.0, 0.8, 0.05)
        mock_accuracy.return_value = True
        mock_status = MagicMock(elapsed=1.5)

        with patch.object(strategy, "get_health", return_value=(True, "tt-sdxl-trace")):
            with patch("asyncio.run", return_value=([mock_status], 2.0)):
                strategy.run_eval()


class TestImageClientStrategyRunBenchmark(unittest.TestCase):
    """Tests for run_benchmark method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        device = MagicMock()
        device.name = "test_device"
        return ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.image_client.get_num_calls", return_value=2)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_benchmark_success(self, mock_mkdir, mock_file, mock_num_calls):
        strategy = self._create_strategy()
        # Multiple status entries to verify averaging of TTFT and inference_steps_per_second
        status_list = [
            ImageGenerationTestStatus(
                status=True,
                elapsed=1.0,
                num_inference_steps=20,
                inference_steps_per_second=20.0,
            ),
            ImageGenerationTestStatus(
                status=True,
                elapsed=2.0,
                num_inference_steps=20,
                inference_steps_per_second=10.0,
            ),
        ]

        mock_benchmark = MagicMock(return_value=status_list)
        strategy.benchmark_methods["tt-sdxl-trace"] = mock_benchmark

        with patch.object(strategy, "get_health", return_value=(True, "tt-sdxl-trace")):
            strategy.run_benchmark()

        mock_benchmark.assert_called_once()
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

        # Verify benchmarks structure and computed averages
        benchmarks = report_data["benchmarks"]
        assert benchmarks["num_requests"] == 2
        assert benchmarks["num_inference_steps"] == 20
        # TTFT: (1.0 + 2.0) / 2 = 1.5
        assert benchmarks["ttft"] == 1.5
        # inference_steps_per_second: (20.0 + 10.0) / 2 = 15.0
        assert benchmarks["inference_steps_per_second"] == 15.0

        # Verify metadata
        assert report_data["model"] == "test_model"
        assert report_data["device"] == "test_device"
        assert report_data["task_type"] == "image"

    @patch.object(ImageClientStrategy, "get_health", return_value=(False, None))
    def test_run_benchmark_health_check_failed(self, mock_health):
        strategy = self._create_strategy()

        with pytest.raises(Exception):
            strategy.run_benchmark()

    @patch("utils.media_clients.image_client.get_num_calls", return_value=5)
    def test_run_benchmark_propagates_benchmark_exception(self, mock_num_calls):
        strategy = self._create_strategy()

        mock_benchmark = MagicMock(side_effect=RuntimeError("Benchmark error"))
        strategy.benchmark_methods["tt-sdxl-trace"] = mock_benchmark

        with patch.object(strategy, "get_health", return_value=(True, "tt-sdxl-trace")):
            with pytest.raises(RuntimeError):
                strategy.run_benchmark()

    @patch("utils.media_clients.image_client.get_num_calls", return_value=3)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_benchmark_routes_to_img2img(
        self, mock_mkdir, mock_file, mock_num_calls
    ):
        strategy = self._create_strategy()
        mock_status = ImageGenerationTestStatus(
            status=True,
            elapsed=1.5,
            num_inference_steps=30,
            inference_steps_per_second=20.0,
        )

        mock_benchmark = MagicMock(return_value=[mock_status])
        strategy.benchmark_methods["tt-sdxl-image-to-image"] = mock_benchmark

        with patch.object(
            strategy, "get_health", return_value=(True, "tt-sdxl-image-to-image")
        ):
            strategy.run_benchmark()

        mock_benchmark.assert_called_once()

    @patch("utils.media_clients.image_client.get_num_calls", return_value=3)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_benchmark_routes_to_inpainting(
        self, mock_mkdir, mock_file, mock_num_calls
    ):
        strategy = self._create_strategy()
        mock_status = ImageGenerationTestStatus(
            status=True,
            elapsed=1.5,
            num_inference_steps=20,
            inference_steps_per_second=13.3,
        )

        mock_benchmark = MagicMock(return_value=[mock_status])
        strategy.benchmark_methods["tt-sdxl-edit"] = mock_benchmark

        with patch.object(strategy, "get_health", return_value=(True, "tt-sdxl-edit")):
            strategy.run_benchmark()

        mock_benchmark.assert_called_once()

    @patch("utils.media_clients.image_client.get_num_calls", return_value=3)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_benchmark_routes_to_default(
        self, mock_mkdir, mock_file, mock_num_calls
    ):
        """Test fallback to default benchmark method for unknown runner."""
        strategy = self._create_strategy()
        mock_status = ImageGenerationTestStatus(
            status=True,
            elapsed=1.5,
            num_inference_steps=20,
            inference_steps_per_second=13.3,
        )

        with patch.object(
            strategy, "get_health", return_value=(True, "unknown-runner")
        ):
            with patch.object(
                strategy,
                "_run_image_generation_benchmark",
                return_value=[mock_status],
            ):
                strategy.run_benchmark()


class TestImageClientStrategyImageGenerationBenchmark(unittest.TestCase):
    """Tests for _run_image_generation_benchmark and _generate_image methods."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("utils.media_clients.image_client.requests.post")
    def test_generate_image_success(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        status, elapsed = strategy._generate_image()

        assert status is True
        assert elapsed > 0

    @patch("utils.media_clients.image_client.requests.post")
    def test_generate_image_failure_with_json_error(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Server error"}
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError) as exc_info:
            strategy._generate_image()

        assert "500" in str(exc_info.value)

    @patch("utils.media_clients.image_client.requests.post")
    def test_generate_image_failure_parse_error(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("Parse error")
        mock_response.text = "Raw error text"
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError):
            strategy._generate_image()

    @patch.object(ImageClientStrategy, "_generate_image", return_value=(True, 1.5))
    def test_run_image_generation_benchmark(self, mock_generate):
        strategy = self._create_strategy()

        result = strategy._run_image_generation_benchmark(3)

        assert len(result) == 3
        assert all(isinstance(s, ImageGenerationTestStatus) for s in result)
        assert mock_generate.call_count == 3

    @patch.object(ImageClientStrategy, "_generate_image", return_value=(True, 0))
    def test_run_image_generation_benchmark_zero_elapsed(self, mock_generate):
        strategy = self._create_strategy()

        result = strategy._run_image_generation_benchmark(1)

        assert result[0].inference_steps_per_second == 0


class TestImageClientStrategyImg2ImgBenchmark(unittest.TestCase):
    """Tests for img2img benchmark methods."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("builtins.open", mock_open(read_data='{"file": "base64data"}'))
    @patch("utils.media_clients.image_client.requests.post")
    def test_generate_image_img2img_success(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        status, elapsed = strategy._generate_image_img2img()

        assert status is True

    @patch("builtins.open", mock_open(read_data='{"file": "base64data"}'))
    @patch("utils.media_clients.image_client.requests.post")
    def test_generate_image_img2img_failure_with_json(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError):
            strategy._generate_image_img2img()

    @patch("builtins.open", mock_open(read_data='{"file": "base64data"}'))
    @patch("utils.media_clients.image_client.requests.post")
    def test_generate_image_img2img_failure_parse_error(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("Parse error")
        mock_response.text = "Error text"
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError):
            strategy._generate_image_img2img()

    @patch.object(
        ImageClientStrategy, "_generate_image_img2img", return_value=(True, 2.0)
    )
    def test_run_img2img_generation_benchmark(self, mock_generate):
        strategy = self._create_strategy()

        result = strategy._run_img2img_generation_benchmark(2)

        assert len(result) == 2
        assert mock_generate.call_count == 2

    @patch.object(
        ImageClientStrategy, "_generate_image_img2img", return_value=(True, 0)
    )
    def test_run_img2img_generation_benchmark_zero_elapsed(self, mock_generate):
        strategy = self._create_strategy()

        result = strategy._run_img2img_generation_benchmark(1)

        assert result[0].inference_steps_per_second == 0


class TestImageClientStrategyInpaintingBenchmark(unittest.TestCase):
    """Tests for inpainting benchmark methods."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch(
        "builtins.open",
        mock_open(read_data='{"inpaint_image": "img", "inpaint_mask": "mask"}'),
    )
    @patch("utils.media_clients.image_client.requests.post")
    def test_generate_image_inpainting_success(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        status, elapsed = strategy._generate_image_inpainting()

        assert status is True

    @patch(
        "builtins.open",
        mock_open(read_data='{"inpaint_image": "img", "inpaint_mask": "mask"}'),
    )
    @patch("utils.media_clients.image_client.requests.post")
    def test_generate_image_inpainting_failure_with_json(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError):
            strategy._generate_image_inpainting()

    @patch(
        "builtins.open",
        mock_open(read_data='{"inpaint_image": "img", "inpaint_mask": "mask"}'),
    )
    @patch("utils.media_clients.image_client.requests.post")
    def test_generate_image_inpainting_failure_parse_error(self, mock_post):
        strategy = self._create_strategy()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("Parse error")
        mock_response.text = "Error text"
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError):
            strategy._generate_image_inpainting()

    @patch.object(
        ImageClientStrategy, "_generate_image_inpainting", return_value=(True, 1.5)
    )
    def test_run_inpainting_generation_benchmark(self, mock_generate):
        strategy = self._create_strategy()

        result = strategy._run_inpainting_generation_benchmark(2)

        assert len(result) == 2
        assert mock_generate.call_count == 2

    @patch.object(
        ImageClientStrategy, "_generate_image_inpainting", return_value=(True, 0)
    )
    def test_run_inpainting_generation_benchmark_zero_elapsed(self, mock_generate):
        strategy = self._create_strategy()

        result = strategy._run_inpainting_generation_benchmark(1)

        assert result[0].inference_steps_per_second == 0


class MockAsyncResponse:
    """Mock async response for aiohttp."""

    def __init__(self, status=200, json_data=None):
        self.status = status
        self._json_data = json_data or {}

    async def json(self):
        return self._json_data

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


# Tests for async helper methods that make HTTP requests
class TestImageClientStrategyAsyncHttpMethods(unittest.TestCase):
    """Tests for async HTTP methods."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_generate_image_eval_async_success(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(
            status=200, json_data={"images": ["base64img"]}
        )
        mock_session = MockAsyncSession(mock_response)

        status, elapsed, base64img = asyncio.run(
            strategy._generate_image_eval_async(mock_session, "test prompt")
        )

        assert status is True
        assert base64img == "base64img"
        assert elapsed > 0

    def test_generate_image_eval_async_failure_status(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(status=500)
        mock_session = MockAsyncSession(mock_response)

        status, elapsed, base64img = asyncio.run(
            strategy._generate_image_eval_async(mock_session, "test prompt")
        )

        assert status is False
        assert base64img is None

    def test_generate_image_eval_async_empty_images(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(status=200, json_data={"images": []})
        mock_session = MockAsyncSession(mock_response)

        status, elapsed, base64img = asyncio.run(
            strategy._generate_image_eval_async(mock_session, "test prompt")
        )

        assert status is True
        assert base64img is None

    def test_generate_image_eval_async_exception(self):
        import asyncio

        strategy = self._create_strategy()

        class FailingSession:
            def post(self, *args, **kwargs):
                raise Exception("Connection error")

        status, elapsed, base64img = asyncio.run(
            strategy._generate_image_eval_async(FailingSession(), "test prompt")
        )

        assert status is False
        assert base64img is None

    def test_generate_image_img2img_eval_async_success(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(
            status=200, json_data={"images": ["base64img"]}
        )
        mock_session = MockAsyncSession(mock_response)

        status, elapsed, base64img = asyncio.run(
            strategy._generate_image_img2img_eval_async(
                mock_session, "test prompt", {"file": "data"}
            )
        )

        assert status is True
        assert base64img == "base64img"

    def test_generate_image_img2img_eval_async_failure(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(status=400)
        mock_session = MockAsyncSession(mock_response)

        status, elapsed, base64img = asyncio.run(
            strategy._generate_image_img2img_eval_async(
                mock_session, "test prompt", {"file": "data"}
            )
        )

        assert status is False
        assert base64img is None

    def test_generate_image_img2img_eval_async_empty_images(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(status=200, json_data={"images": []})
        mock_session = MockAsyncSession(mock_response)

        status, elapsed, base64img = asyncio.run(
            strategy._generate_image_img2img_eval_async(
                mock_session, "test prompt", {"file": "data"}
            )
        )

        assert status is True
        assert base64img is None

    def test_generate_image_img2img_eval_async_exception(self):
        import asyncio

        strategy = self._create_strategy()

        class FailingSession:
            def post(self, *args, **kwargs):
                raise Exception("Connection error")

        status, elapsed, base64img = asyncio.run(
            strategy._generate_image_img2img_eval_async(
                FailingSession(), "test prompt", {"file": "data"}
            )
        )

        assert status is False
        assert base64img is None

    def test_generate_image_inpainting_eval_async_success(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(
            status=200, json_data={"images": ["base64img"]}
        )
        mock_session = MockAsyncSession(mock_response)

        status, elapsed, base64img = asyncio.run(
            strategy._generate_image_inpainting_eval_async(
                mock_session, "test prompt", "img", "mask"
            )
        )

        assert status is True
        assert base64img == "base64img"

    def test_generate_image_inpainting_eval_async_failure(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(status=500)
        mock_session = MockAsyncSession(mock_response)

        status, elapsed, base64img = asyncio.run(
            strategy._generate_image_inpainting_eval_async(
                mock_session, "test prompt", "img", "mask"
            )
        )

        assert status is False
        assert base64img is None

    def test_generate_image_inpainting_eval_async_empty_images(self):
        import asyncio

        strategy = self._create_strategy()
        mock_response = MockAsyncResponse(status=200, json_data={"images": []})
        mock_session = MockAsyncSession(mock_response)

        status, elapsed, base64img = asyncio.run(
            strategy._generate_image_inpainting_eval_async(
                mock_session, "test prompt", "img", "mask"
            )
        )

        assert status is True
        assert base64img is None

    def test_generate_image_inpainting_eval_async_exception(self):
        import asyncio

        strategy = self._create_strategy()

        class FailingSession:
            def post(self, *args, **kwargs):
                raise Exception("Connection error")

        status, elapsed, base64img = asyncio.run(
            strategy._generate_image_inpainting_eval_async(
                FailingSession(), "test prompt", "img", "mask"
            )
        )

        assert status is False
        assert base64img is None


# Async tests for eval methods - using mocked _generate methods to avoid aiohttp complexity
class TestImageClientStrategyAsyncEvalMethods(unittest.TestCase):
    """Tests for async eval methods."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch(
        "utils.media_clients.image_client.sdxl_get_prompts",
        return_value=["prompt1", "prompt2"],
    )
    @patch(
        "utils.media_clients.image_client.is_sdxl_num_prompts_enabled", return_value=2
    )
    def test_run_image_generation_eval_success(self, mock_num_prompts, mock_prompts):
        import asyncio

        strategy = self._create_strategy()

        # Mock the individual image generation method
        with patch.object(
            strategy,
            "_generate_image_eval_async",
            new_callable=AsyncMock,
            return_value=(True, 1.5, "base64img"),
        ):
            status_list, total_time = asyncio.run(strategy._run_image_generation_eval())

        assert len(status_list) == 2
        assert total_time > 0

    @patch(
        "utils.media_clients.image_client.sdxl_get_prompts", return_value=["prompt1"]
    )
    @patch(
        "utils.media_clients.image_client.is_sdxl_num_prompts_enabled", return_value=1
    )
    def test_run_image_generation_eval_with_failures(
        self, mock_num_prompts, mock_prompts
    ):
        import asyncio

        strategy = self._create_strategy()

        with patch.object(
            strategy,
            "_generate_image_eval_async",
            new_callable=AsyncMock,
            return_value=(False, 1.5, None),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                asyncio.run(strategy._run_image_generation_eval())

        assert "failed" in str(exc_info.value).lower()

    @patch(
        "utils.media_clients.image_client.sdxl_get_prompts",
        return_value=["prompt1", "prompt2"],
    )
    @patch(
        "utils.media_clients.image_client.is_sdxl_num_prompts_enabled", return_value=2
    )
    def test_run_image_generation_eval_zero_elapsed(
        self, mock_num_prompts, mock_prompts
    ):
        import asyncio

        strategy = self._create_strategy()

        with patch.object(
            strategy,
            "_generate_image_eval_async",
            new_callable=AsyncMock,
            return_value=(True, 0, "base64img"),
        ):
            status_list, _ = asyncio.run(strategy._run_image_generation_eval())

        assert status_list[0].inference_steps_per_second == 0


class TestImageClientStrategyAsyncImg2ImgEval(unittest.TestCase):
    """Tests for async img2img eval methods."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("builtins.open", mock_open(read_data='{"file": "base64data"}'))
    def test_run_img2img_generation_eval_success(self):
        import asyncio

        strategy = self._create_strategy()

        with patch.object(
            strategy,
            "_generate_image_img2img_eval_async",
            new_callable=AsyncMock,
            return_value=(True, 1.5, "base64img"),
        ):
            status_list, total_time = asyncio.run(
                strategy._run_img2img_generation_eval()
            )

        assert len(status_list) == 1

    @patch("builtins.open", mock_open(read_data='{"file": "base64data"}'))
    def test_run_img2img_generation_eval_failure(self):
        import asyncio

        strategy = self._create_strategy()

        with patch.object(
            strategy,
            "_generate_image_img2img_eval_async",
            new_callable=AsyncMock,
            return_value=(False, 1.5, None),
        ):
            with pytest.raises(RuntimeError):
                asyncio.run(strategy._run_img2img_generation_eval())

    @patch("builtins.open", mock_open(read_data='{"file": "base64data"}'))
    def test_run_img2img_generation_eval_zero_elapsed(self):
        import asyncio

        strategy = self._create_strategy()

        with patch.object(
            strategy,
            "_generate_image_img2img_eval_async",
            new_callable=AsyncMock,
            return_value=(True, 0, "base64img"),
        ):
            status_list, _ = asyncio.run(strategy._run_img2img_generation_eval())

        assert status_list[0].inference_steps_per_second == 0


class TestImageClientStrategyAsyncInpaintingEval(unittest.TestCase):
    """Tests for async inpainting eval methods."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch(
        "builtins.open",
        mock_open(read_data='{"inpaint_image": "img", "inpaint_mask": "mask"}'),
    )
    def test_run_inpainting_generation_eval_success(self):
        import asyncio

        strategy = self._create_strategy()

        with patch.object(
            strategy,
            "_generate_image_inpainting_eval_async",
            new_callable=AsyncMock,
            return_value=(True, 1.5, "base64img"),
        ):
            status_list, total_time = asyncio.run(
                strategy._run_inpainting_generation_eval()
            )

        assert len(status_list) == 1

    @patch(
        "builtins.open",
        mock_open(read_data='{"inpaint_image": "img", "inpaint_mask": "mask"}'),
    )
    def test_run_inpainting_generation_eval_failure(self):
        import asyncio

        strategy = self._create_strategy()

        with patch.object(
            strategy,
            "_generate_image_inpainting_eval_async",
            new_callable=AsyncMock,
            return_value=(False, 1.5, None),
        ):
            with pytest.raises(RuntimeError):
                asyncio.run(strategy._run_inpainting_generation_eval())

    @patch(
        "builtins.open",
        mock_open(read_data='{"inpaint_image": "img", "inpaint_mask": "mask"}'),
    )
    def test_run_inpainting_generation_eval_zero_elapsed(self):
        import asyncio

        strategy = self._create_strategy()

        with patch.object(
            strategy,
            "_generate_image_inpainting_eval_async",
            new_callable=AsyncMock,
            return_value=(True, 0, "base64img"),
        ):
            status_list, _ = asyncio.run(strategy._run_inpainting_generation_eval())

        assert status_list[0].inference_steps_per_second == 0
