# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import sys
from unittest.mock import Mock, patch

import pytest

# Mock external dependencies before importing
sys.modules["ttnn"] = Mock()

# Mock config settings
mock_settings = Mock()
mock_settings.default_throttle_level = "5"
mock_settings.enable_telemetry = False
mock_settings.is_galaxy = False
mock_settings.device_mesh_shape = (1, 1)

mock_settings_module = Mock()
mock_settings_module.settings = mock_settings
sys.modules["config.settings"] = mock_settings_module

# Mock telemetry
sys.modules["telemetry.telemetry_client"] = Mock()
sys.modules["telemetry.telemetry_client"].get_telemetry_client = Mock()

# Mock torch utils
mock_set_torch_thread_limits = Mock()
sys.modules["utils.torch_utils"] = Mock()
sys.modules["utils.torch_utils"].set_torch_thread_limits = mock_set_torch_thread_limits

# Mock logger
mock_logger = Mock()
sys.modules["utils.logger"] = Mock()
sys.modules["utils.logger"].TTLogger = Mock(return_value=mock_logger)

# Mock device runner
sys.modules["tt_model_runners.base_device_runner"] = Mock()
sys.modules["tt_model_runners.runner_fabric"] = Mock()

# Now import the module under test
from device_workers.worker_utils import (
    _setup_galaxy_mesh_config,
    initialize_device_worker,
    setup_cpu_threading_limits,
    setup_worker_environment,
)


class TestSetupCPUThreadingLimits:
    """Test cases for setup_cpu_threading_limits function"""

    def test_sets_environment_variables(self):
        """Test that CPU threading limits are set correctly"""
        cpu_threads = "4"
        num_threads = 2

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "device_workers.worker_utils.set_torch_thread_limits"
            ) as mock_set:
                setup_cpu_threading_limits(cpu_threads, num_threads)

                assert os.environ["OMP_NUM_THREADS"] == "4"
                assert os.environ["MKL_NUM_THREADS"] == "4"
                assert os.environ["TORCH_NUM_THREADS"] == "4"
                mock_set.assert_called_once_with(num_threads=2)

    def test_sets_throttle_level_when_configured(self):
        """Test that throttle level is set when configured in settings"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("device_workers.worker_utils.set_torch_thread_limits"):
                # Create a mock settings with throttle level
                mock_settings_with_throttle = Mock()
                mock_settings_with_throttle.default_throttle_level = "3"

                with patch(
                    "device_workers.worker_utils.settings", mock_settings_with_throttle
                ):
                    setup_cpu_threading_limits("2", 1)

                    assert os.environ["TT_MM_THROTTLE_PERF"] == "3"

    def test_skips_throttle_level_when_not_configured(self):
        """Test that throttle level is not set when not configured"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("device_workers.worker_utils.set_torch_thread_limits"):
                # Create a mock settings without throttle level
                mock_settings_no_throttle = Mock()
                mock_settings_no_throttle.default_throttle_level = None

                with patch(
                    "device_workers.worker_utils.settings", mock_settings_no_throttle
                ):
                    setup_cpu_threading_limits("2", 1)

                    assert "TT_MM_THROTTLE_PERF" not in os.environ

    def test_default_num_threads(self):
        """Test default num_threads parameter"""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "device_workers.worker_utils.set_torch_thread_limits"
            ) as mock_set:
                setup_cpu_threading_limits("2")

                mock_set.assert_called_with(num_threads=1)


class TestSetupWorkerEnvironment:
    """Test cases for setup_worker_environment function"""

    def test_sets_device_visibility(self):
        """Test that device visibility environment variables are set"""
        worker_id = "0"

        with patch.dict(os.environ, {}, clear=True):
            with patch("device_workers.worker_utils.get_telemetry_client"):
                setup_worker_environment(worker_id)

                assert os.environ["TT_VISIBLE_DEVICES"] == "0"
                assert os.environ["TT_METAL_VISIBLE_DEVICES"] == "0"

    def test_sets_metal_cache_path(self):
        """Test that TT_METAL_CACHE is set correctly"""
        worker_id = "0"

        with patch.dict(os.environ, {"TT_METAL_HOME": "/opt/tt-metal"}, clear=True):
            with patch("device_workers.worker_utils.get_telemetry_client"):
                setup_worker_environment(worker_id)

                assert os.environ["TT_METAL_CACHE"] == "/opt/tt-metal/built/0"

    def test_handles_comma_separated_worker_id(self):
        """Test that comma in worker_id is replaced with underscore"""
        worker_id = "0,1"

        with patch.dict(os.environ, {"TT_METAL_HOME": "/opt/tt-metal"}, clear=True):
            with patch("device_workers.worker_utils.get_telemetry_client"):
                setup_worker_environment(worker_id)

                assert os.environ["TT_METAL_CACHE"] == "/opt/tt-metal/built/0_1"

    def test_initializes_telemetry_when_enabled(self):
        """Test that telemetry is initialized when enabled"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("device_workers.worker_utils.set_torch_thread_limits"):
                with patch(
                    "device_workers.worker_utils.get_telemetry_client"
                ) as mock_get_telemetry:
                    # Create settings with telemetry enabled
                    mock_settings_telemetry = Mock()
                    mock_settings_telemetry.enable_telemetry = True
                    mock_settings_telemetry.is_galaxy = False
                    mock_settings_telemetry.default_throttle_level = None

                    with patch(
                        "device_workers.worker_utils.settings", mock_settings_telemetry
                    ):
                        setup_worker_environment("0")

                        mock_get_telemetry.assert_called_once()

    def test_skips_telemetry_when_disabled(self):
        """Test that telemetry is not initialized when disabled"""
        mock_settings.enable_telemetry = False
        mock_get_telemetry = Mock()

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "device_workers.worker_utils.get_telemetry_client", mock_get_telemetry
            ):
                setup_worker_environment("0")

                mock_get_telemetry.assert_not_called()

    def test_calls_galaxy_setup_when_enabled(self):
        """Test that galaxy mesh config is set up when is_galaxy is True"""
        with patch.dict(os.environ, {"TT_METAL_HOME": "/opt/tt-metal"}, clear=True):
            with patch("device_workers.worker_utils.set_torch_thread_limits"):
                with patch("device_workers.worker_utils.get_telemetry_client"):
                    with patch(
                        "device_workers.worker_utils._setup_galaxy_mesh_config"
                    ) as mock_galaxy:
                        # Create settings with galaxy enabled
                        mock_settings_galaxy = Mock()
                        mock_settings_galaxy.enable_telemetry = False
                        mock_settings_galaxy.is_galaxy = True
                        mock_settings_galaxy.default_throttle_level = None

                        with patch(
                            "device_workers.worker_utils.settings", mock_settings_galaxy
                        ):
                            setup_worker_environment("0")

                            mock_galaxy.assert_called_once_with("/opt/tt-metal")

    def test_skips_galaxy_setup_when_disabled(self):
        """Test that galaxy mesh config is not set up when is_galaxy is False"""
        mock_settings.is_galaxy = False

        with patch.dict(os.environ, {}, clear=True):
            with patch("device_workers.worker_utils.get_telemetry_client"):
                with patch(
                    "device_workers.worker_utils._setup_galaxy_mesh_config"
                ) as mock_galaxy:
                    setup_worker_environment("0")

                    mock_galaxy.assert_not_called()

    def test_custom_cpu_threads(self):
        """Test custom cpu_threads parameter"""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "device_workers.worker_utils.set_torch_thread_limits"
            ) as mock_set:
                with patch("device_workers.worker_utils.get_telemetry_client"):
                    setup_worker_environment("0", cpu_threads="8", num_threads=4)

                    assert os.environ["OMP_NUM_THREADS"] == "8"
                    mock_set.assert_called_with(num_threads=4)


class TestSetupGalaxyMeshConfig:
    """Test cases for _setup_galaxy_mesh_config function"""

    def test_sets_core_grid_override(self):
        """Test that core grid override is set"""
        with patch.dict(os.environ, {}, clear=True):
            _setup_galaxy_mesh_config("/opt/tt-metal")

            assert os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] == "7,7"

    def test_sets_n150_mesh_descriptor(self):
        """Test mesh descriptor for (1, 1) device shape"""
        mock_settings.device_mesh_shape = (1, 1)

        with patch.dict(os.environ, {}, clear=True):
            _setup_galaxy_mesh_config("/opt/tt-metal")

            expected_path = (
                "/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/"
                "n150_mesh_graph_descriptor.textproto"
            )
            assert os.environ["TT_MESH_GRAPH_DESC_PATH"] == expected_path

    def test_sets_n300_mesh_descriptor(self):
        """Test mesh descriptor for (2, 1) device shape"""
        with patch.dict(os.environ, {}, clear=True):
            # Create settings with n300 shape
            mock_settings_n300 = Mock()
            mock_settings_n300.device_mesh_shape = (2, 1)

            with patch("device_workers.worker_utils.settings", mock_settings_n300):
                _setup_galaxy_mesh_config("/opt/tt-metal")

                expected_path = (
                    "/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/"
                    "n300_mesh_graph_descriptor.textproto"
                )
                assert os.environ["TT_MESH_GRAPH_DESC_PATH"] == expected_path

    def test_sets_t3k_mesh_descriptor(self):
        """Test mesh descriptor for (2, 4) device shape"""
        with patch.dict(os.environ, {}, clear=True):
            # Create settings with t3k shape
            mock_settings_t3k = Mock()
            mock_settings_t3k.device_mesh_shape = (2, 4)

            with patch("device_workers.worker_utils.settings", mock_settings_t3k):
                _setup_galaxy_mesh_config("/opt/tt-metal")

                expected_path = (
                    "/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/"
                    "t3k_mesh_graph_descriptor.textproto"
                )
                assert os.environ["TT_MESH_GRAPH_DESC_PATH"] == expected_path

    def test_skips_mesh_descriptor_for_unknown_shape(self):
        """Test that mesh descriptor is not set for unknown device shape"""
        with patch.dict(os.environ, {}, clear=True):
            # Create settings with unknown shape
            mock_settings_unknown = Mock()
            mock_settings_unknown.device_mesh_shape = (3, 3)  # Unknown shape

            with patch("device_workers.worker_utils.settings", mock_settings_unknown):
                _setup_galaxy_mesh_config("/opt/tt-metal")

                assert "TT_MESH_GRAPH_DESC_PATH" not in os.environ


class TestInitializeDeviceWorker:
    """Test cases for initialize_device_worker function"""

    def test_creates_and_sets_event_loop(self):
        """Test that a new event loop is created and set"""
        mock_get_device_runner = Mock()
        mock_device_runner = Mock()
        mock_device_runner.set_device = Mock()
        mock_device_runner.warmup = Mock(return_value=asyncio.Future())
        mock_device_runner.warmup.return_value.set_result(None)
        mock_get_device_runner.return_value = mock_device_runner

        mock_loop = Mock()
        mock_loop.run_until_complete = Mock()

        with patch(
            "device_workers.worker_utils.get_device_runner", mock_get_device_runner
        ):
            with patch("asyncio.new_event_loop", return_value=mock_loop) as mock_new:
                with patch("asyncio.set_event_loop") as mock_set:
                    device_runner, loop = initialize_device_worker("0", mock_logger)

                    mock_new.assert_called_once()
                    mock_set.assert_called_once_with(mock_loop)
                    assert loop is mock_loop

    def test_calls_get_device_runner_with_correct_params(self):
        """Test that get_device_runner is called with correct parameters"""
        mock_get_device_runner = Mock()
        mock_device_runner = Mock()
        mock_device_runner.set_device = Mock()
        mock_device_runner.warmup = Mock(return_value=asyncio.Future())
        mock_device_runner.warmup.return_value.set_result(None)
        mock_get_device_runner.return_value = mock_device_runner

        mock_loop = Mock()
        mock_loop.run_until_complete = Mock()

        with patch(
            "device_workers.worker_utils.get_device_runner", mock_get_device_runner
        ):
            with patch("asyncio.new_event_loop", return_value=mock_loop):
                with patch("asyncio.set_event_loop"):
                    initialize_device_worker("0", mock_logger, num_torch_threads=4)

                    mock_get_device_runner.assert_called_once_with("0", 4)

    def test_calls_set_device_and_warmup(self):
        """Test that set_device and warmup are called"""
        mock_get_device_runner = Mock()
        mock_device_runner = Mock()
        mock_device_runner.set_device = Mock()
        mock_device_runner.warmup = Mock(return_value=asyncio.Future())
        mock_device_runner.warmup.return_value.set_result(None)
        mock_get_device_runner.return_value = mock_device_runner

        mock_loop = Mock()
        mock_loop.run_until_complete = Mock()

        with patch(
            "device_workers.worker_utils.get_device_runner", mock_get_device_runner
        ):
            with patch("asyncio.new_event_loop", return_value=mock_loop):
                with patch("asyncio.set_event_loop"):
                    device_runner, loop = initialize_device_worker("0", mock_logger)

                    mock_device_runner.set_device.assert_called_once()
                    mock_loop.run_until_complete.assert_called_once()

    def test_returns_device_runner_and_loop(self):
        """Test that device_runner and loop are returned on success"""
        mock_get_device_runner = Mock()
        mock_device_runner = Mock()
        mock_device_runner.set_device = Mock()
        mock_device_runner.warmup = Mock(return_value=asyncio.Future())
        mock_device_runner.warmup.return_value.set_result(None)
        mock_get_device_runner.return_value = mock_device_runner

        mock_loop = Mock()
        mock_loop.run_until_complete = Mock()

        with patch(
            "device_workers.worker_utils.get_device_runner", mock_get_device_runner
        ):
            with patch("asyncio.new_event_loop", return_value=mock_loop):
                with patch("asyncio.set_event_loop"):
                    device_runner, loop = initialize_device_worker("0", mock_logger)

                    assert device_runner is mock_device_runner
                    assert loop is mock_loop

    def test_handles_keyboard_interrupt_during_warmup(self):
        """Test that KeyboardInterrupt during warmup is handled gracefully"""
        mock_get_device_runner = Mock()
        mock_device_runner = Mock()
        mock_device_runner.set_device = Mock()
        mock_get_device_runner.return_value = mock_device_runner

        mock_loop = Mock()
        mock_loop.run_until_complete = Mock(side_effect=KeyboardInterrupt())
        mock_loop.close = Mock()

        with patch(
            "device_workers.worker_utils.get_device_runner", mock_get_device_runner
        ):
            with patch("asyncio.new_event_loop", return_value=mock_loop):
                with patch("asyncio.set_event_loop"):
                    device_runner, loop = initialize_device_worker("0", mock_logger)

                    assert device_runner is None
                    assert loop is None
                    mock_loop.close.assert_called_once()
                    mock_logger.warning.assert_called_once()

    def test_handles_exception_during_initialization(self):
        """Test that exceptions during initialization are handled"""
        mock_get_device_runner = Mock(
            side_effect=Exception("Device initialization failed")
        )

        mock_loop = Mock()
        mock_loop.close = Mock()

        with patch(
            "device_workers.worker_utils.get_device_runner", mock_get_device_runner
        ):
            with patch("asyncio.new_event_loop", return_value=mock_loop):
                with patch("asyncio.set_event_loop"):
                    with pytest.raises(Exception, match="Device initialization failed"):
                        initialize_device_worker("0", mock_logger)

                    mock_loop.close.assert_called_once()
                    mock_logger.error.assert_called_once()

    def test_closes_device_on_exception(self):
        """Test that device is closed if exception occurs after device_runner is created"""
        mock_device_runner = Mock()
        mock_device_runner.set_device = Mock(side_effect=Exception("Set device failed"))
        mock_device_runner.close_device = Mock()

        mock_get_device_runner = Mock(return_value=mock_device_runner)

        mock_loop = Mock()
        mock_loop.close = Mock()

        with patch(
            "device_workers.worker_utils.get_device_runner", mock_get_device_runner
        ):
            with patch("asyncio.new_event_loop", return_value=mock_loop):
                with patch("asyncio.set_event_loop"):
                    with pytest.raises(Exception, match="Set device failed"):
                        initialize_device_worker("0", mock_logger)

                    mock_device_runner.close_device.assert_called_once()
                    mock_loop.close.assert_called_once()


# Pytest fixtures for module-level setup
@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test"""
    mock_logger.reset_mock()
    mock_set_torch_thread_limits.reset_mock()
    # Reset mock_settings to default values
    mock_settings.default_throttle_level = "5"
    mock_settings.enable_telemetry = False
    mock_settings.is_galaxy = False
    mock_settings.device_mesh_shape = (1, 1)
    mock_settings.default_throttle_level = "5"
    mock_settings.enable_telemetry = False
    mock_settings.is_galaxy = False
    mock_settings.device_mesh_shape = (1, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
