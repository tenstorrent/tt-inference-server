# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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
mock_settings.warmup_timeout_seconds = 1800

mock_settings_module = Mock()
mock_settings_module.settings = mock_settings
sys.modules["config.settings"] = mock_settings_module


def _consume_coro(coro):
    """Close any coroutine passed to a mocked ``run_until_complete`` to avoid
    ``RuntimeWarning: coroutine '...' was never awaited`` from ``asyncio.wait_for``.
    """
    if asyncio.iscoroutine(coro):
        coro.close()
    return None


def _consume_coro_raise(exc):
    """Return a side_effect that closes the coroutine then raises ``exc``."""

    def _inner(coro):
        if asyncio.iscoroutine(coro):
            coro.close()
        raise exc

    return _inner


# Mock telemetry
sys.modules["telemetry.telemetry_client"] = Mock()
sys.modules["telemetry.telemetry_client"].get_telemetry_client = Mock()

# Mock vllm_settings (needed by config.constants at import time)
sys.modules["config.vllm_settings"] = Mock()

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

# Now import the modules under test
from device_workers.worker_utils import initialize_device_worker
from utils.runner_utils import (
    _setup_blackhole_mesh_config,
    _setup_galaxy_mesh_config,
    setup_cpu_threading_limits,
    setup_runner_environment,
)


class TestSetupCPUThreadingLimits:
    """Test cases for setup_cpu_threading_limits function"""

    def test_sets_environment_variables(self):
        """Test that CPU threading limits are set correctly"""
        cpu_threads = "4"
        num_threads = 2

        with patch.dict(os.environ, {}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits") as mock_set:
                setup_cpu_threading_limits(cpu_threads, num_threads)

                assert os.environ["OMP_NUM_THREADS"] == "4"
                assert os.environ["MKL_NUM_THREADS"] == "4"
                assert os.environ["TORCH_NUM_THREADS"] == "2"
                mock_set.assert_called_once_with(num_threads=2)

    def test_sets_throttle_level_when_configured(self):
        """Test that throttle level is set when configured in settings"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits"):
                # Create a mock settings with throttle level
                mock_settings_with_throttle = Mock()
                mock_settings_with_throttle.default_throttle_level = "3"

                with patch("utils.runner_utils.settings", mock_settings_with_throttle):
                    setup_cpu_threading_limits("2", 1)

                    assert os.environ["TT_MM_THROTTLE_PERF"] == "3"

    def test_skips_throttle_level_when_not_configured(self):
        """Test that throttle level is not set when not configured"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits"):
                # Create a mock settings without throttle level
                mock_settings_no_throttle = Mock()
                mock_settings_no_throttle.default_throttle_level = None

                with patch("utils.runner_utils.settings", mock_settings_no_throttle):
                    setup_cpu_threading_limits("2", 1)

                    assert "TT_MM_THROTTLE_PERF" not in os.environ

    def test_default_num_threads(self):
        """Test default num_threads parameter"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits") as mock_set:
                setup_cpu_threading_limits("2")

                mock_set.assert_called_with(num_threads=1)


class TestSetupRunnerEnvironment:
    """Test cases for setup_runner_environment function"""

    def test_sets_device_visibility(self):
        """Test that device visibility environment variables are set"""
        worker_id = "0"

        with patch.dict(os.environ, {}, clear=True):
            with patch("utils.runner_utils.get_telemetry_client"):
                setup_runner_environment(worker_id)

                assert os.environ["TT_VISIBLE_DEVICES"] == "0"

    def test_sets_metal_cache_path(self):
        """Test that TT_METAL_CACHE is set correctly"""
        worker_id = "0"

        with patch.dict(os.environ, {"TT_METAL_HOME": "/opt/tt-metal"}, clear=True):
            with patch("utils.runner_utils.get_telemetry_client"):
                setup_runner_environment(worker_id)

                assert os.environ["TT_METAL_CACHE"] == "/opt/tt-metal/built/0"

    def test_handles_comma_separated_worker_id(self):
        """Test that comma in worker_id is replaced with underscore"""
        worker_id = "0,1"

        with patch.dict(os.environ, {"TT_METAL_HOME": "/opt/tt-metal"}, clear=True):
            with patch("utils.runner_utils.get_telemetry_client"):
                setup_runner_environment(worker_id)

                assert os.environ["TT_METAL_CACHE"] == "/opt/tt-metal/built/0_1"

    def test_initializes_telemetry_when_enabled(self):
        """Test that telemetry is initialized when enabled"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits"):
                with patch(
                    "utils.runner_utils.get_telemetry_client"
                ) as mock_get_telemetry:
                    # Create settings with telemetry enabled
                    mock_settings_telemetry = Mock()
                    mock_settings_telemetry.enable_telemetry = True
                    mock_settings_telemetry.is_galaxy = False
                    mock_settings_telemetry.default_throttle_level = None

                    with patch("utils.runner_utils.settings", mock_settings_telemetry):
                        setup_runner_environment("0")

                        mock_get_telemetry.assert_called_once()

    def test_calls_galaxy_setup_when_enabled_for_whisper(self):
        """Test that galaxy mesh config is set up when is_galaxy is True for qualifying runner"""
        with patch.dict(os.environ, {"TT_METAL_HOME": "/opt/tt-metal"}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits"):
                with patch("utils.runner_utils.get_telemetry_client"):
                    with patch(
                        "utils.runner_utils._setup_galaxy_mesh_config"
                    ) as mock_galaxy:
                        mock_settings_galaxy = Mock()
                        mock_settings_galaxy.enable_telemetry = False
                        mock_settings_galaxy.is_galaxy = True
                        mock_settings_galaxy.model_runner = "tt-whisper"
                        mock_settings_galaxy.default_throttle_level = None

                        with patch("utils.runner_utils.settings", mock_settings_galaxy):
                            setup_runner_environment("0")

                            mock_galaxy.assert_called_once_with("/opt/tt-metal")

    def test_skips_galaxy_setup_for_non_qualifying_runner(self):
        """Test that galaxy mesh config is NOT set up for runners like tt-sdxl-trace"""
        with patch.dict(os.environ, {"TT_METAL_HOME": "/opt/tt-metal"}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits"):
                with patch("utils.runner_utils.get_telemetry_client"):
                    with patch(
                        "utils.runner_utils._setup_galaxy_mesh_config"
                    ) as mock_galaxy:
                        mock_settings_galaxy = Mock()
                        mock_settings_galaxy.enable_telemetry = False
                        mock_settings_galaxy.is_galaxy = True
                        mock_settings_galaxy.model_runner = "tt-sdxl-trace"
                        mock_settings_galaxy.default_throttle_level = None

                        with patch("utils.runner_utils.settings", mock_settings_galaxy):
                            setup_runner_environment("0")

                            mock_galaxy.assert_not_called()

    def test_custom_cpu_threads(self):
        """Test custom cpu_threads parameter"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits") as mock_set:
                with patch("utils.runner_utils.get_telemetry_client"):
                    setup_runner_environment("0", cpu_threads="8", num_torch_threads=4)

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

            with patch("utils.runner_utils.settings", mock_settings_n300):
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

            with patch("utils.runner_utils.settings", mock_settings_t3k):
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

            with patch("utils.runner_utils.settings", mock_settings_unknown):
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
        mock_loop.run_until_complete = Mock(side_effect=_consume_coro)

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
        mock_loop.run_until_complete = Mock(side_effect=_consume_coro)

        with patch(
            "device_workers.worker_utils.get_device_runner", mock_get_device_runner
        ):
            with patch("asyncio.new_event_loop", return_value=mock_loop):
                with patch("asyncio.set_event_loop"):
                    initialize_device_worker("0", mock_logger)

                    mock_get_device_runner.assert_called_once_with("0")

    def test_calls_set_device_and_warmup(self):
        """Test that set_device and warmup are called"""
        mock_get_device_runner = Mock()
        mock_device_runner = Mock()
        mock_device_runner.set_device = Mock()
        mock_device_runner.warmup = Mock(return_value=asyncio.Future())
        mock_device_runner.warmup.return_value.set_result(None)
        mock_get_device_runner.return_value = mock_device_runner

        mock_loop = Mock()
        mock_loop.run_until_complete = Mock(side_effect=_consume_coro)

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
        mock_loop.run_until_complete = Mock(side_effect=_consume_coro)

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
        mock_loop.run_until_complete = Mock(
            side_effect=_consume_coro_raise(KeyboardInterrupt())
        )
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
                    assert mock_logger.error.call_count == 1

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


class TestSetupBlackholeMeshConfig:
    """Test cases for _setup_blackhole_mesh_config function"""

    def test_sets_p150_mesh_descriptor(self):
        """Test mesh descriptor is set for p150 device"""
        mock_settings_bh = Mock()
        mock_settings_bh.device = "p150"

        with patch.dict(os.environ, {}, clear=True):
            with patch("utils.runner_utils.settings", mock_settings_bh):
                _setup_blackhole_mesh_config("/opt/tt-metal")

                expected_path = (
                    "/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/"
                    "p150_mesh_graph_descriptor.textproto"
                )
                assert os.environ["TT_MESH_GRAPH_DESC_PATH"] == expected_path

    def test_sets_p300_mesh_descriptor(self):
        """Test mesh descriptor is set for p300 device"""
        mock_settings_bh = Mock()
        mock_settings_bh.device = "p300"

        with patch.dict(os.environ, {}, clear=True):
            with patch("utils.runner_utils.settings", mock_settings_bh):
                _setup_blackhole_mesh_config("/opt/tt-metal")

                expected_path = (
                    "/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/"
                    "p300_mesh_graph_descriptor.textproto"
                )
                assert os.environ["TT_MESH_GRAPH_DESC_PATH"] == expected_path

    def test_skips_descriptor_for_unknown_device(self):
        """Test that TT_MESH_GRAPH_DESC_PATH is not set for unknown BH device"""
        mock_settings_bh = Mock()
        mock_settings_bh.device = "unknown_device"

        with patch.dict(os.environ, {}, clear=True):
            with patch("utils.runner_utils.settings", mock_settings_bh):
                _setup_blackhole_mesh_config("/opt/tt-metal")

                assert "TT_MESH_GRAPH_DESC_PATH" not in os.environ


class TestSetupRunnerEnvironmentBlackhole:
    """Test BH device branch in setup_runner_environment"""

    def test_calls_blackhole_setup_for_bh_device_with_qualifying_runner(self):
        """Test that _setup_blackhole_mesh_config is called for a BH device with whisper"""
        with patch.dict(os.environ, {"TT_METAL_HOME": "/opt/tt-metal"}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits"):
                with patch("utils.runner_utils.get_telemetry_client"):
                    with patch(
                        "utils.runner_utils._setup_blackhole_mesh_config"
                    ) as mock_bh:
                        mock_settings_bh = Mock()
                        mock_settings_bh.enable_telemetry = False
                        mock_settings_bh.is_galaxy = False
                        mock_settings_bh.device = "p150"
                        mock_settings_bh.model_runner = "tt-whisper"
                        mock_settings_bh.default_throttle_level = None

                        with patch("utils.runner_utils.settings", mock_settings_bh):
                            setup_runner_environment("0")

                            mock_bh.assert_called_once_with("/opt/tt-metal")

    def test_calls_blackhole_setup_for_speecht5(self):
        """Test that _setup_blackhole_mesh_config is called for speecht5 on BH"""
        with patch.dict(os.environ, {"TT_METAL_HOME": "/opt/tt-metal"}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits"):
                with patch("utils.runner_utils.get_telemetry_client"):
                    with patch(
                        "utils.runner_utils._setup_blackhole_mesh_config"
                    ) as mock_bh:
                        mock_settings_bh = Mock()
                        mock_settings_bh.enable_telemetry = False
                        mock_settings_bh.is_galaxy = False
                        mock_settings_bh.device = "p300"
                        mock_settings_bh.model_runner = "tt-speecht5-tts"
                        mock_settings_bh.default_throttle_level = None

                        with patch("utils.runner_utils.settings", mock_settings_bh):
                            setup_runner_environment("0")

                            mock_bh.assert_called_once_with("/opt/tt-metal")

    def test_skips_blackhole_setup_for_sdxl_on_bh_device(self):
        """Test that _setup_blackhole_mesh_config is NOT called for tt-sdxl-trace on BH"""
        with patch.dict(os.environ, {"TT_METAL_HOME": "/opt/tt-metal"}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits"):
                with patch("utils.runner_utils.get_telemetry_client"):
                    with patch(
                        "utils.runner_utils._setup_blackhole_mesh_config"
                    ) as mock_bh:
                        mock_settings_bh = Mock()
                        mock_settings_bh.enable_telemetry = False
                        mock_settings_bh.is_galaxy = False
                        mock_settings_bh.device = "p150x8"
                        mock_settings_bh.model_runner = "tt-sdxl-trace"
                        mock_settings_bh.default_throttle_level = None

                        with patch("utils.runner_utils.settings", mock_settings_bh):
                            setup_runner_environment("0")

                            mock_bh.assert_not_called()

    def test_skips_blackhole_setup_for_flux_on_bh_device(self):
        """Test that _setup_blackhole_mesh_config is NOT called for flux on BH"""
        with patch.dict(os.environ, {"TT_METAL_HOME": "/opt/tt-metal"}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits"):
                with patch("utils.runner_utils.get_telemetry_client"):
                    with patch(
                        "utils.runner_utils._setup_blackhole_mesh_config"
                    ) as mock_bh:
                        mock_settings_bh = Mock()
                        mock_settings_bh.enable_telemetry = False
                        mock_settings_bh.is_galaxy = False
                        mock_settings_bh.device = "p150x8"
                        mock_settings_bh.model_runner = "tt-flux.1-dev"
                        mock_settings_bh.default_throttle_level = None

                        with patch("utils.runner_utils.settings", mock_settings_bh):
                            setup_runner_environment("0")

                            mock_bh.assert_not_called()

    def test_does_not_call_blackhole_setup_for_non_bh_device(self):
        """Test that _setup_blackhole_mesh_config is not called for non-BH device"""
        with patch.dict(os.environ, {"TT_METAL_HOME": "/opt/tt-metal"}, clear=True):
            with patch("utils.runner_utils.set_torch_thread_limits"):
                with patch("utils.runner_utils.get_telemetry_client"):
                    with patch(
                        "utils.runner_utils._setup_blackhole_mesh_config"
                    ) as mock_bh:
                        mock_settings_non_bh = Mock()
                        mock_settings_non_bh.enable_telemetry = False
                        mock_settings_non_bh.is_galaxy = False
                        mock_settings_non_bh.device = "n300"
                        mock_settings_non_bh.model_runner = "tt-whisper"
                        mock_settings_non_bh.default_throttle_level = None

                        with patch("utils.runner_utils.settings", mock_settings_non_bh):
                            setup_runner_environment("0")

                            mock_bh.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
