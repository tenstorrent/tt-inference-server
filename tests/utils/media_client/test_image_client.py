# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

# Mock ML dependencies before importing modules that depend on them
sys.modules["open_clip"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torchvision"] = MagicMock()
sys.modules["torchvision.transforms"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.linalg"] = MagicMock()

from utils.media_clients.image_client import (
    GUIDANCE_SCALE,
    GUIDANCE_SCALE_IMG2IMG,
    GUIDANCE_SCALE_INPAINTING,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    SDXL_IMG2IMG_INFERENCE_STEPS,
    SDXL_INPAINTING_INFERENCE_STEPS,
    SDXL_SD35_BENCHMARK_NUM_PROMPTS,
    SDXL_SD35_INFERENCE_STEPS,
    SEED_IMG2IMG,
    SEED_INPAINTING,
    STRENGTH_IMG2IMG,
    STRENGTH_INPAINTING,
    WORKFLOW_BENCHMARKS,
    WORKFLOW_EVALS,
    ImageClientStrategy,
)
from utils.media_clients.test_status import ImageGenerationTestStatus


def create_async_context_manager(response):
    """Helper to create proper async context manager for aiohttp response."""

    @asynccontextmanager
    async def mock_post(*args, **kwargs):
        yield response

    return mock_post


class TestConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_workflow_evals(self):
        assert WORKFLOW_EVALS == "evals"

    def test_workflow_benchmarks(self):
        assert WORKFLOW_BENCHMARKS == "benchmarks"

    def test_sdxl_sd35_benchmark_num_prompts(self):
        assert SDXL_SD35_BENCHMARK_NUM_PROMPTS == 20

    def test_sdxl_sd35_inference_steps(self):
        assert SDXL_SD35_INFERENCE_STEPS == 20

    def test_sdxl_inpainting_inference_steps(self):
        assert SDXL_INPAINTING_INFERENCE_STEPS == 20

    def test_negative_prompt(self):
        assert "low quality" in NEGATIVE_PROMPT

    def test_guidance_scale(self):
        assert GUIDANCE_SCALE == 8

    def test_num_inference_steps(self):
        assert NUM_INFERENCE_STEPS == 20

    def test_img2img_constants(self):
        assert SDXL_IMG2IMG_INFERENCE_STEPS == 30
        assert GUIDANCE_SCALE_IMG2IMG == 7.5
        assert SEED_IMG2IMG == 0
        assert STRENGTH_IMG2IMG == 0.6

    def test_inpainting_constants(self):
        assert GUIDANCE_SCALE_INPAINTING == 8.0
        assert SEED_INPAINTING == 0
        assert STRENGTH_INPAINTING == 0.99


class TestImageClientStrategyInit(unittest.TestCase):
    """Tests for ImageClientStrategy.__init__ method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        device = MagicMock()
        device.name = "test_device"
        return ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_init_sets_benchmark_methods(self):
        strategy = self._create_strategy()
        assert "tt-sdxl-trace" in strategy.benchmark_methods
        assert "tt-sdxl-image-to-image" in strategy.benchmark_methods
        assert "tt-sdxl-edit" in strategy.benchmark_methods
        assert "tt-sd3.5" in strategy.benchmark_methods

    def test_init_sets_eval_methods(self):
        strategy = self._create_strategy()
        assert "tt-sdxl-trace" in strategy.eval_methods
        assert "tt-sdxl-image-to-image" in strategy.eval_methods
        assert "tt-sdxl-edit" in strategy.eval_methods
        assert "tt-sd3.5" in strategy.eval_methods

    def test_init_inherits_base_attributes(self):
        model_spec = MagicMock()
        device = MagicMock()
        strategy = ImageClientStrategy(
            {"key": "value"}, model_spec, device, "/output", 9000
        )
        assert strategy.all_params == {"key": "value"}
        assert strategy.model_spec == model_spec
        assert strategy.device == device
        assert strategy.output_path == "/output"
        assert strategy.service_port == 9000
        assert strategy.base_url == "http://localhost:9000"


class TestImageClientStrategyCalculateTtft(unittest.TestCase):
    """Tests for _calculate_ttft_value method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        device = MagicMock()
        return ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    def test_calculate_ttft_with_status_list(self):
        strategy = self._create_strategy()
        status_list = [
            MagicMock(elapsed=1.0),
            MagicMock(elapsed=2.0),
            MagicMock(elapsed=3.0),
        ]
        result = strategy._calculate_ttft_value(status_list)
        assert result == 2.0

    def test_calculate_ttft_empty_list(self):
        strategy = self._create_strategy()
        result = strategy._calculate_ttft_value([])
        assert result == 0

    def test_calculate_ttft_single_item(self):
        strategy = self._create_strategy()
        status_list = [MagicMock(elapsed=5.0)]
        result = strategy._calculate_ttft_value(status_list)
        assert result == 5.0


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
    def test_run_eval_success(
        self, mock_mkdir, mock_file, mock_metrics, mock_accuracy
    ):
        strategy = self._create_strategy()
        mock_metrics.return_value = (10.0, 0.8, 0.05)
        mock_accuracy.return_value = True

        mock_status = MagicMock(elapsed=1.5)
        with patch.object(strategy, "get_health", return_value=(True, "tt-sdxl-trace")):
            with patch("asyncio.run", return_value=([mock_status], 2.0)):
                strategy.run_eval()

        mock_mkdir.assert_called()

    @patch.object(ImageClientStrategy, "get_health", return_value=(False, None))
    def test_run_eval_health_check_failed(self, mock_health):
        strategy = self._create_strategy()

        with pytest.raises(Exception):
            strategy.run_eval()

    @patch.object(
        ImageClientStrategy, "get_health", return_value=(True, "tt-sdxl-trace")
    )
    def test_run_eval_exception_during_eval(self, mock_health):
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

    @patch("utils.media_clients.image_client.get_num_calls", return_value=5)
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_run_benchmark_success(self, mock_mkdir, mock_file, mock_num_calls):
        strategy = self._create_strategy()
        mock_status = ImageGenerationTestStatus(
            status=True,
            elapsed=1.5,
            num_inference_steps=20,
            inference_steps_per_second=13.3,
        )

        mock_benchmark = MagicMock(return_value=[mock_status])
        strategy.benchmark_methods["tt-sdxl-trace"] = mock_benchmark

        with patch.object(strategy, "get_health", return_value=(True, "tt-sdxl-trace")):
            strategy.run_benchmark()

        mock_benchmark.assert_called_once()

    @patch.object(ImageClientStrategy, "get_health", return_value=(False, None))
    def test_run_benchmark_health_check_failed(self, mock_health):
        strategy = self._create_strategy()

        with pytest.raises(Exception):
            strategy.run_benchmark()

    @patch("utils.media_clients.image_client.get_num_calls", return_value=5)
    def test_run_benchmark_exception(self, mock_num_calls):
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

        with patch.object(strategy, "get_health", return_value=(True, "unknown-runner")):
            with patch.object(
                strategy,
                "_run_image_generation_benchmark",
                return_value=[mock_status],
            ):
                strategy.run_benchmark()


class TestImageClientStrategyGenerateReport(unittest.TestCase):
    """Tests for _generate_report method."""

    def _create_strategy(self):
        model_spec = MagicMock()
        model_spec.model_name = "test_model"
        model_spec.model_id = "test_id"
        device = MagicMock()
        device.name = "test_device"
        return ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_report_with_status_list(self, mock_mkdir, mock_file):
        strategy = self._create_strategy()
        status_list = [
            MagicMock(
                elapsed=1.0, num_inference_steps=20, inference_steps_per_second=20.0
            ),
            MagicMock(
                elapsed=2.0, num_inference_steps=20, inference_steps_per_second=10.0
            ),
        ]

        strategy._generate_report(status_list)

        mock_mkdir.assert_called_once()
        mock_file.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_generate_report_empty_status_list(self, mock_mkdir, mock_file):
        strategy = self._create_strategy()

        strategy._generate_report([])

        mock_file.assert_called_once()


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
    @patch("utils.media_clients.image_client.is_sdxl_num_prompts_enabled", return_value=2)
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
    @patch("utils.media_clients.image_client.is_sdxl_num_prompts_enabled", return_value=1)
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
    @patch("utils.media_clients.image_client.is_sdxl_num_prompts_enabled", return_value=2)
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


# Parametrized tests for edge cases
@pytest.mark.parametrize(
    "runner,expected_method_name",
    [
        ("tt-sdxl-trace", "_run_image_generation_benchmark"),
        ("tt-sdxl-image-to-image", "_run_img2img_generation_benchmark"),
        ("tt-sdxl-edit", "_run_inpainting_generation_benchmark"),
        ("tt-sd3.5", "_run_image_generation_benchmark"),
        ("unknown-runner", "_run_image_generation_benchmark"),
    ],
)
def test_benchmark_method_routing(runner, expected_method_name):
    """Test that benchmark routes to correct method based on runner."""
    model_spec = MagicMock()
    model_spec.model_name = "test"
    model_spec.model_id = "test_id"
    device = MagicMock()
    device.name = "test"
    strategy = ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    method = strategy.benchmark_methods.get(
        runner, strategy._run_image_generation_benchmark
    )
    assert (
        method.__name__ == expected_method_name
        or expected_method_name == "_run_image_generation_benchmark"
    )


@pytest.mark.parametrize(
    "runner,expected_method_name",
    [
        ("tt-sdxl-trace", "_run_image_generation_eval"),
        ("tt-sdxl-image-to-image", "_run_img2img_generation_eval"),
        ("tt-sdxl-edit", "_run_inpainting_generation_eval"),
        ("tt-sd3.5", "_run_image_generation_eval"),
    ],
)
def test_eval_method_routing(runner, expected_method_name):
    """Test that eval routes to correct method based on runner."""
    model_spec = MagicMock()
    model_spec.model_name = "test"
    device = MagicMock()
    device.name = "test"
    strategy = ImageClientStrategy({}, model_spec, device, "/tmp", 8000)

    method = strategy.eval_methods.get(runner)
    assert method is not None
    assert method.__name__ == expected_method_name
