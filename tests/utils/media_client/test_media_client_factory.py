# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Mock ML dependencies before importing modules that depend on them
sys.modules["open_clip"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torchvision"] = MagicMock()
sys.modules["torchvision.transforms"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.linalg"] = MagicMock()

from utils.media_clients.media_client_factory import (
    STRATEGY_MAP,
    MediaClientFactory,
    MediaTaskType,
)


def create_mock_strategy_class(name: str = "MockStrategy"):
    """Helper to create a mock strategy class with __name__ attribute."""
    mock_class = MagicMock()
    mock_class.__name__ = name
    mock_class.return_value = MagicMock()
    return mock_class


class TestMediaTaskType(unittest.TestCase):
    """Tests for MediaTaskType enum."""

    def test_evaluation_value(self):
        assert MediaTaskType.EVALUATION.value == "evaluation"

    def test_benchmark_value(self):
        assert MediaTaskType.BENCHMARK.value == "benchmark"

    def test_enum_members(self):
        assert set(MediaTaskType.__members__.keys()) == {"EVALUATION", "BENCHMARK"}


class TestStrategyMap(unittest.TestCase):
    """Tests for STRATEGY_MAP configuration."""

    def test_strategy_map_contains_cnn(self):
        assert "CNN" in STRATEGY_MAP

    def test_strategy_map_contains_image(self):
        assert "IMAGE" in STRATEGY_MAP

    def test_strategy_map_contains_audio(self):
        assert "AUDIO" in STRATEGY_MAP

    def test_strategy_map_contains_embedding(self):
        assert "EMBEDDING" in STRATEGY_MAP

    def test_strategy_map_size(self):
        assert len(STRATEGY_MAP) == 4


class TestMediaClientFactoryCreateStrategy(unittest.TestCase):
    """Tests for MediaClientFactory._create_strategy method."""

    def _create_mock_model_spec(self, model_type_name: str) -> MagicMock:
        mock_spec = MagicMock()
        mock_spec.model_type.name = model_type_name
        return mock_spec

    def test_create_strategy_cnn(self):
        mock_spec = self._create_mock_model_spec("CNN")
        mock_params = {"param": "value"}
        mock_device = MagicMock()
        output_path = "/tmp/output"
        service_port = 8000

        with patch.dict(
            "utils.media_clients.media_client_factory.STRATEGY_MAP",
            {"CNN": create_mock_strategy_class("CnnClientStrategy")},
        ):
            strategy = MediaClientFactory._create_strategy(
                mock_spec, mock_params, mock_device, output_path, service_port
            )
            assert strategy is not None

    def test_create_strategy_image(self):
        mock_spec = self._create_mock_model_spec("IMAGE")
        mock_params = {"param": "value"}
        mock_device = MagicMock()
        output_path = "/tmp/output"
        service_port = 8000

        with patch.dict(
            "utils.media_clients.media_client_factory.STRATEGY_MAP",
            {"IMAGE": create_mock_strategy_class("ImageClientStrategy")},
        ):
            strategy = MediaClientFactory._create_strategy(
                mock_spec, mock_params, mock_device, output_path, service_port
            )
            assert strategy is not None

    def test_create_strategy_audio(self):
        mock_spec = self._create_mock_model_spec("AUDIO")
        mock_params = {"param": "value"}
        mock_device = MagicMock()
        output_path = "/tmp/output"
        service_port = 8000

        with patch.dict(
            "utils.media_clients.media_client_factory.STRATEGY_MAP",
            {"AUDIO": create_mock_strategy_class("AudioClientStrategy")},
        ):
            strategy = MediaClientFactory._create_strategy(
                mock_spec, mock_params, mock_device, output_path, service_port
            )
            assert strategy is not None

    def test_create_strategy_unsupported_type_raises_value_error(self):
        mock_spec = self._create_mock_model_spec("UNSUPPORTED")
        mock_params = {}
        mock_device = MagicMock()

        with pytest.raises(ValueError):
            MediaClientFactory._create_strategy(
                mock_spec, mock_params, mock_device, "/tmp", 8000
            )

    def test_create_strategy_passes_correct_arguments(self):
        mock_spec = self._create_mock_model_spec("CNN")
        mock_params = {"key": "value"}
        mock_device = MagicMock(name="test_device")
        output_path = "/custom/output"
        service_port = 9000

        mock_strategy_class = create_mock_strategy_class("CnnClientStrategy")
        mock_strategy_instance = MagicMock()
        mock_strategy_class.return_value = mock_strategy_instance

        with patch.dict(
            "utils.media_clients.media_client_factory.STRATEGY_MAP",
            {"CNN": mock_strategy_class},
        ):
            result = MediaClientFactory._create_strategy(
                mock_spec, mock_params, mock_device, output_path, service_port
            )

            mock_strategy_class.assert_called_once_with(
                mock_params, mock_spec, mock_device, output_path, service_port
            )
            assert result == mock_strategy_instance


class TestMediaClientFactoryRunMediaTask(unittest.TestCase):
    """Tests for MediaClientFactory.run_media_task method."""

    def _create_mock_model_spec(self, model_type_name: str) -> MagicMock:
        mock_spec = MagicMock()
        mock_spec.model_type.name = model_type_name
        mock_spec.model_name = "test_model"
        return mock_spec

    def test_run_media_task_evaluation_success(self):
        mock_spec = self._create_mock_model_spec("CNN")
        mock_device = MagicMock()
        mock_device.name = "test_device"
        mock_strategy = MagicMock()

        with patch.object(
            MediaClientFactory, "_create_strategy", return_value=mock_strategy
        ):
            result = MediaClientFactory.run_media_task(
                mock_spec, {}, mock_device, "/tmp", 8000, MediaTaskType.EVALUATION
            )

            mock_strategy.run_eval.assert_called_once()
            mock_strategy.run_benchmark.assert_not_called()
            assert result == 0

    def test_run_media_task_benchmark_success(self):
        mock_spec = self._create_mock_model_spec("CNN")
        mock_device = MagicMock()
        mock_device.name = "test_device"
        mock_strategy = MagicMock()

        with patch.object(
            MediaClientFactory, "_create_strategy", return_value=mock_strategy
        ):
            result = MediaClientFactory.run_media_task(
                mock_spec, {}, mock_device, "/tmp", 8000, MediaTaskType.BENCHMARK
            )

            mock_strategy.run_benchmark.assert_called_once()
            mock_strategy.run_eval.assert_not_called()
            assert result == 0

    def test_run_media_task_returns_failure_on_exception(self):
        mock_spec = self._create_mock_model_spec("CNN")
        mock_device = MagicMock()
        mock_device.name = "test_device"

        with patch.object(
            MediaClientFactory,
            "_create_strategy",
            side_effect=Exception("Test exception"),
        ):
            result = MediaClientFactory.run_media_task(
                mock_spec, {}, mock_device, "/tmp", 8000, MediaTaskType.EVALUATION
            )

            assert result == 1

    def test_run_media_task_returns_failure_on_run_eval_exception(self):
        mock_spec = self._create_mock_model_spec("CNN")
        mock_device = MagicMock()
        mock_device.name = "test_device"
        mock_strategy = MagicMock()
        mock_strategy.run_eval.side_effect = RuntimeError("Eval failed")

        with patch.object(
            MediaClientFactory, "_create_strategy", return_value=mock_strategy
        ):
            result = MediaClientFactory.run_media_task(
                mock_spec, {}, mock_device, "/tmp", 8000, MediaTaskType.EVALUATION
            )

            assert result == 1

    def test_run_media_task_returns_failure_on_run_benchmark_exception(self):
        mock_spec = self._create_mock_model_spec("CNN")
        mock_device = MagicMock()
        mock_device.name = "test_device"
        mock_strategy = MagicMock()
        mock_strategy.run_benchmark.side_effect = RuntimeError("Benchmark failed")

        with patch.object(
            MediaClientFactory, "_create_strategy", return_value=mock_strategy
        ):
            result = MediaClientFactory.run_media_task(
                mock_spec, {}, mock_device, "/tmp", 8000, MediaTaskType.BENCHMARK
            )

            assert result == 1

    def test_run_media_task_returns_failure_on_value_error(self):
        mock_spec = self._create_mock_model_spec("UNSUPPORTED")
        mock_device = MagicMock()
        mock_device.name = "test_device"

        result = MediaClientFactory.run_media_task(
            mock_spec, {}, mock_device, "/tmp", 8000, MediaTaskType.EVALUATION
        )

        assert result == 1


# Pytest parametrized tests for better edge case coverage
@pytest.mark.parametrize(
    "model_type_name",
    ["CNN", "IMAGE", "AUDIO"],
)
def test_create_strategy_all_supported_types(model_type_name):
    """Test that all supported model types create a strategy successfully."""
    mock_spec = MagicMock()
    mock_spec.model_type.name = model_type_name
    mock_device = MagicMock()
    mock_strategy_class = create_mock_strategy_class(f"{model_type_name}Strategy")

    with patch.dict(
        "utils.media_clients.media_client_factory.STRATEGY_MAP",
        {model_type_name: mock_strategy_class},
    ):
        result = MediaClientFactory._create_strategy(
            mock_spec, {}, mock_device, "/tmp", 8000
        )
        assert result == mock_strategy_class.return_value


@pytest.mark.parametrize(
    "unsupported_type",
    ["LLM", "VIDEO", "TEXT", "UNKNOWN", "", "cnn", "image", "audio"],
)
def test_create_strategy_unsupported_types(unsupported_type):
    """Test that unsupported model types raise ValueError."""
    mock_spec = MagicMock()
    mock_spec.model_type.name = unsupported_type
    mock_device = MagicMock()

    with pytest.raises(ValueError):
        MediaClientFactory._create_strategy(mock_spec, {}, mock_device, "/tmp", 8000)


@pytest.mark.parametrize(
    "task_type,expected_method",
    [
        (MediaTaskType.EVALUATION, "run_eval"),
        (MediaTaskType.BENCHMARK, "run_benchmark"),
    ],
)
def test_run_media_task_calls_correct_method(task_type, expected_method):
    """Test that run_media_task calls the correct method based on task type."""
    mock_spec = MagicMock()
    mock_spec.model_type.name = "CNN"
    mock_spec.model_name = "test_model"
    mock_device = MagicMock()
    mock_device.name = "test_device"
    mock_strategy = MagicMock()

    with patch.object(
        MediaClientFactory, "_create_strategy", return_value=mock_strategy
    ):
        result = MediaClientFactory.run_media_task(
            mock_spec, {}, mock_device, "/tmp", 8000, task_type
        )

        getattr(mock_strategy, expected_method).assert_called_once()
        assert result == 0


@pytest.mark.parametrize(
    "exception_type",
    [ValueError, RuntimeError, TypeError, IOError, Exception],
)
def test_run_media_task_handles_various_exceptions(exception_type):
    """Test that run_media_task handles various exception types gracefully."""
    mock_spec = MagicMock()
    mock_spec.model_type.name = "CNN"
    mock_spec.model_name = "test_model"
    mock_device = MagicMock()
    mock_device.name = "test_device"

    with patch.object(
        MediaClientFactory,
        "_create_strategy",
        side_effect=exception_type("Test error"),
    ):
        result = MediaClientFactory.run_media_task(
            mock_spec, {}, mock_device, "/tmp", 8000, MediaTaskType.EVALUATION
        )

        assert result == 1
