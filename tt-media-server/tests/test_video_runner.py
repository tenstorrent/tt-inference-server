# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import pickle
import sys
from unittest.mock import MagicMock, Mock, patch

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

from ipc.video_shm import VideoRequest, VideoStatus

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tt_model_runners.video_runner import (
    _attach_mpi_comm,
    _broadcast_request,
    _create_dit_runner,
    _is_shutdown,
    _rank,
    _run_inference_loop,
    _write_error_to_shm,
    _write_response_to_shm,
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


class TestAttachMpiComm:
    def test_success_returns_comm_world(self):
        mock_mpi4py = Mock()
        with patch.dict(sys.modules, {"mpi4py": mock_mpi4py}):
            result = _attach_mpi_comm()
            assert mock_mpi4py.rc.initialize is False
            assert mock_mpi4py.rc.finalize is False
            assert result is mock_mpi4py.MPI.COMM_WORLD

    def test_raises_runtime_error_when_mpi4py_unavailable(self):
        with patch.dict(sys.modules, {"mpi4py": None}):
            with pytest.raises(RuntimeError, match="mpi4py is required"):
                _attach_mpi_comm()


class TestBroadcastRequest:
    def test_broadcasts_request_from_root(self):
        req = VideoRequest(
            task_id="t1",
            prompt="p",
            negative_prompt="",
            num_inference_steps=1,
            seed=0,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=1.0,
            guidance_scale_2=1.0,
        )
        comm = Mock()
        comm.bcast.return_value = req
        result = _broadcast_request(comm, req)
        comm.bcast.assert_called_once_with(req, root=0)
        assert result is req

    def test_returns_none_for_shutdown(self):
        comm = Mock()
        comm.bcast.return_value = None
        result = _broadcast_request(comm, None)
        assert result is None


class TestRunInferenceLoop:
    @staticmethod
    def _make_request(**overrides):
        defaults = dict(
            task_id="task-loop",
            prompt="test prompt for inference loop",
            negative_prompt="neg",
            num_inference_steps=5,
            seed=7,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=3.0,
            guidance_scale_2=4.0,
        )
        defaults.update(overrides)
        return VideoRequest(**defaults)

    def test_rank0_processes_request_and_writes_response(self):
        import tt_model_runners.video_runner as vr

        original = vr._shutdown
        vr._shutdown = False

        req = self._make_request()
        comm = Mock()
        comm.Get_rank.return_value = 0
        comm.bcast = Mock(side_effect=[req, None])

        input_shm = MagicMock()
        input_shm.read_request = Mock(side_effect=[req, None])
        output_shm = MagicMock()

        runner = Mock()
        runner.run.return_value = np.zeros((1, 2, 2, 3), dtype=np.uint8)

        mock_domain = Mock()
        with patch.dict(
            sys.modules,
            {
                "domain": Mock(),
                "domain.video_generate_request": mock_domain,
            },
        ):
            _run_inference_loop(comm, runner, input_shm, output_shm)

        runner.run.assert_called_once()
        output_shm.write_response.assert_called_once()
        resp = output_shm.write_response.call_args[0][0]
        assert resp.status == VideoStatus.SUCCESS
        assert resp.task_id == "task-loop"
        vr._shutdown = original

    def test_rank0_writes_error_on_inference_failure(self):
        import tt_model_runners.video_runner as vr

        original = vr._shutdown
        vr._shutdown = False

        req = self._make_request()
        comm = Mock()
        comm.Get_rank.return_value = 0
        comm.bcast = Mock(side_effect=[req, None])

        input_shm = MagicMock()
        input_shm.read_request = Mock(side_effect=[req, None])
        output_shm = MagicMock()

        runner = Mock()
        runner.run.side_effect = RuntimeError("inference failed")

        mock_domain = Mock()
        with patch.dict(
            sys.modules,
            {
                "domain": Mock(),
                "domain.video_generate_request": mock_domain,
            },
        ):
            _run_inference_loop(comm, runner, input_shm, output_shm)

        output_shm.write_response.assert_called_once()
        resp = output_shm.write_response.call_args[0][0]
        assert resp.status == VideoStatus.ERROR
        assert "inference failed" in resp.error_message
        vr._shutdown = original

    def test_breaks_on_none_request(self):
        import tt_model_runners.video_runner as vr

        original = vr._shutdown
        vr._shutdown = False

        comm = Mock()
        comm.Get_rank.return_value = 0
        comm.bcast.return_value = None

        input_shm = MagicMock()
        input_shm.read_request.return_value = None
        output_shm = MagicMock()

        runner = Mock()
        _run_inference_loop(comm, runner, input_shm, output_shm)

        runner.run.assert_not_called()
        vr._shutdown = original

    def test_non_rank0_runs_inference_without_shm_io(self):
        import tt_model_runners.video_runner as vr

        original = vr._shutdown
        vr._shutdown = False

        req = self._make_request()
        comm = Mock()
        comm.Get_rank.return_value = 1
        comm.bcast = Mock(side_effect=[req, None])

        runner = Mock()
        runner.run.return_value = np.zeros((1, 2, 2, 3), dtype=np.uint8)

        mock_domain = Mock()
        with patch.dict(
            sys.modules,
            {
                "domain": Mock(),
                "domain.video_generate_request": mock_domain,
            },
        ):
            _run_inference_loop(comm, runner, None, None)

        runner.run.assert_called_once()
        vr._shutdown = original


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
