# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio

# Add server path to sys.path for imports
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from tt_model_runners.forge_runners.forge_runner import ForgeRunner
from tt_model_runners.runner_fabric import get_device_runner
from utils.logger import TTLogger


class TestForgeRunner:
    """Test suite for ForgeRunner functionality"""

    @pytest.fixture
    def runner(self):
        """Create a ForgeRunner instance for testing"""
        with patch("tt_model_runners.forge_runners.forge_runner.ModelLoader"):
            return ForgeRunner(device_id="test_device_0")

    @pytest.mark.skip(reason="Disabling temporary for now, will re-enable after fix")
    def test_runner_initialization(self, runner):
        """Test ForgeRunner initialization"""
        assert runner.device_id == "test_device_0"
        assert runner.logger is not None
        assert isinstance(runner.logger, TTLogger)
        assert runner.compiled_model is None

    @pytest.mark.skip(reason="Disabling temporary for now, will re-enable after fix")
    def test_forge_runner_creation_via_fabric(self, monkeypatch):
        """Test that ForgeRunner can be instantiated via factory"""
        monkeypatch.setattr(settings, "model_runner", "forge")
        with patch("tt_model_runners.forge_runners.forge_runner.ModelLoader"):
            runner = get_device_runner("test_worker")
            assert "ForgeRunner" in type(runner).__name__

    @pytest.mark.skip(reason="Disabling temporary for now, will re-enable after fix")
    def test_load_model(self, runner):
        """Test model loading"""
        with patch.object(runner.loader, "load_model") as mock_load_model, patch.object(
            runner.loader, "load_inputs"
        ) as mock_load_inputs, patch("forge.compile") as mock_compile:
            mock_model = Mock()
            mock_inputs = Mock()
            mock_compiled = Mock()
            mock_output = Mock()

            mock_load_model.return_value = mock_model
            mock_load_inputs.return_value = mock_inputs
            mock_compile.return_value = mock_compiled
            mock_compiled.return_value = mock_output

            result = asyncio.run(runner.load_model())

            assert result is True
            assert runner.compiled_model == mock_compiled
            mock_load_model.assert_called_once()
            mock_load_inputs.assert_called_once()
            mock_compile.assert_called_once_with(
                mock_model, sample_inputs=[mock_inputs]
            )
            mock_compiled.assert_called_once_with(mock_inputs)

    @pytest.mark.skip(reason="Disabling temporary for now, will re-enable after fix")
    def test_close_device(self, runner):
        """Test device closing"""
        with patch("time.sleep") as mock_sleep:
            result = runner.close_device()
            assert result is True
            mock_sleep.assert_called_once_with(5)

    @pytest.mark.skip(reason="Disabling temporary for now, will re-enable after fix")
    def test_get_device(self, runner):
        """Test device retrieval"""
        device = runner.get_device(1)
        assert device == {"device_id": 1}

        # Test with None device_id
        device = runner.get_device(None)
        assert device == {"device_id": "MockDevice"}

    @pytest.mark.skip(reason="Disabling temporary for now, will re-enable after fix")
    def test_run_inference(self, runner):
        """Test inference execution"""
        with patch.object(
            runner.loader, "load_inputs"
        ) as mock_load_inputs, patch.object(
            runner.loader, "print_cls_results"
        ) as mock_print_results:
            mock_inputs = Mock()
            mock_output = Mock()
            mock_compiled = Mock()
            mock_compiled.return_value = mock_output

            mock_load_inputs.return_value = mock_inputs
            runner.compiled_model = mock_compiled

            result = runner.run_inference("test prompt")

            assert (
                result
                == "Mock inference result for prompt: test prompt on device: test_device_0"
            )
            mock_load_inputs.assert_called_once()
            mock_compiled.assert_called_once_with(mock_inputs)
            mock_print_results.assert_called_once_with(mock_output)

    @pytest.mark.skip(reason="Disabling temporary for now, will re-enable after fix")
    def test_run_inference_default_steps(self, runner):
        """Test inference with default parameters"""
        with patch.object(
            runner.loader, "load_inputs"
        ) as mock_load_inputs, patch.object(
            runner.loader, "print_cls_results"
        ) as mock_print_results:
            mock_inputs = Mock()
            mock_output = Mock()
            mock_compiled = Mock()
            mock_compiled.return_value = mock_output

            mock_load_inputs.return_value = mock_inputs
            runner.compiled_model = mock_compiled

            result = runner.run_inference("test prompt")

            assert (
                result
                == "Mock inference result for prompt: test prompt on device: test_device_0"
            )
            mock_load_inputs.assert_called_once()
            mock_compiled.assert_called_once_with(mock_inputs)
            mock_print_results.assert_called_once_with(mock_output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
