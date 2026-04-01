# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import pickle
import sys
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

sys.modules["ttnn"] = Mock()

mock_settings = Mock()
mock_settings.enable_telemetry = False
mock_settings.model_runner = "sp_runner"
mock_settings.use_dynamic_batcher = False
mock_settings.is_galaxy = False
mock_settings.device_mesh_shape = (1, 1)
mock_settings.default_throttle_level = ""
mock_settings_module = Mock()
mock_settings_module.settings = mock_settings
mock_settings_module.get_settings = Mock(return_value=mock_settings)
sys.modules["config.settings"] = mock_settings_module
sys.modules["telemetry.telemetry_client"] = Mock()
sys.modules["utils.logger"] = Mock()
sys.modules["utils.logger"].TTLogger = Mock(return_value=Mock())

from ipc.video_shm import VideoStatus

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tt_model_runners.video_runner import (
    _is_shutdown,
    _rank,
    _write_response_to_shm,
    _write_error_to_shm,
    _create_dit_runner,
)


class TestRank:
    def test_defaults_to_zero(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OMPI_COMM_WORLD_RANK", None)
            os.environ.pop("RANK", None)
            assert _rank() == 0

    def test_reads_rank_env(self):
        with patch.dict(os.environ, {"RANK": "2"}):
            assert _rank() == 2

    def test_ompi_rank_takes_precedence(self):
        with patch.dict(os.environ, {"OMPI_COMM_WORLD_RANK": "3", "RANK": "1"}):
            assert _rank() == 3


class TestIsShutdown:
    def test_initially_false(self):
        import tt_model_runners.video_runner as vr

        original = vr._shutdown
        vr._shutdown = False
        assert _is_shutdown() is False
        vr._shutdown = original


class TestWriteResponseToShm:
    def test_writes_pickled_video(self):
        mock_shm = MagicMock()
        frames = np.random.randint(0, 256, (1, 3, 4, 6, 3), dtype=np.uint8)

        _write_response_to_shm(mock_shm, "task-1", frames)

        mock_shm.write_response.assert_called_once()
        resp = mock_shm.write_response.call_args[0][0]
        assert resp.status == VideoStatus.SUCCESS
        assert resp.task_id == "task-1"
        recovered = pickle.loads(resp.frame_data)
        np.testing.assert_array_equal(recovered, frames)

    def test_handles_4d_input(self):
        mock_shm = MagicMock()
        frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)

        _write_response_to_shm(mock_shm, "task-2", frames)

        mock_shm.write_response.assert_called_once()
        resp = mock_shm.write_response.call_args[0][0]
        recovered = pickle.loads(resp.frame_data)
        np.testing.assert_array_equal(recovered, frames)

    def test_preserves_float_data(self):
        mock_shm = MagicMock()
        frames = np.ones((1, 1, 2, 2, 3), dtype=np.float32) * 0.5

        _write_response_to_shm(mock_shm, "task-3", frames)

        resp = mock_shm.write_response.call_args[0][0]
        recovered = pickle.loads(resp.frame_data)
        np.testing.assert_array_almost_equal(recovered, frames)

    def test_preserves_extreme_float_values(self):
        mock_shm = MagicMock()
        frames = np.array([[[[[2.0, -1.0, 0.5]]]]]).astype(np.float32)

        _write_response_to_shm(mock_shm, "task-4", frames)

        resp = mock_shm.write_response.call_args[0][0]
        recovered = pickle.loads(resp.frame_data)
        np.testing.assert_array_almost_equal(recovered, frames)


class TestWriteErrorToShm:
    def test_writes_error_response(self):
        mock_shm = MagicMock()
        _write_error_to_shm(mock_shm, "task-err", "boom")

        mock_shm.write_response.assert_called_once()
        resp = mock_shm.write_response.call_args[0][0]
        assert resp.status == VideoStatus.ERROR
        assert resp.task_id == "task-err"
        assert resp.frame_data == b""
        assert resp.error_message == "boom"


class TestCreateDitRunner:
    @staticmethod
    def _make_dit_module(mock_mochi, mock_wan):
        mock_mochi.__name__ = "TTMochi1Runner"
        mock_wan.__name__ = "TTWan22Runner"
        mod = Mock()
        mod.TTMochi1Runner = mock_mochi
        mod.TTWan22Runner = mock_wan
        return {"tt_model_runners.dit_runners": mod}

    def test_creates_mochi_runner(self):
        mock_mochi = Mock()
        mock_wan = Mock()
        with patch.dict(sys.modules, self._make_dit_module(mock_mochi, mock_wan)):
            _create_dit_runner("tt-mochi-1", 0)
            mock_mochi.assert_called_once_with("")

    def test_creates_wan_runner(self):
        mock_mochi = Mock()
        mock_wan = Mock()
        with patch.dict(sys.modules, self._make_dit_module(mock_mochi, mock_wan)):
            _create_dit_runner("tt-wan2.2", 1)
            mock_wan.assert_called_once_with("")

    def test_raises_on_unsupported_runner(self):
        mock_mochi = Mock()
        mock_wan = Mock()
        with patch.dict(sys.modules, self._make_dit_module(mock_mochi, mock_wan)):
            with pytest.raises(ValueError, match="Unsupported MODEL_RUNNER"):
                _create_dit_runner("invalid_runner", 0)


class TestHandleSigterm:
    def test_sets_shutdown_flag(self):
        import tt_model_runners.video_runner as vr

        original = vr._shutdown
        vr._shutdown = False
        vr._handle_sigterm(None, None)
        assert vr._shutdown is True
        vr._shutdown = original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
