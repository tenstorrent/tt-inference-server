# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
    run_all_ranks,
    video_request_to_generate_request,
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
    def test_writes_mp4_path_to_shm(self, tmp_path):
        mock_shm = MagicMock()
        mp4_path = str(tmp_path / "out.mp4")
        open(mp4_path, "wb").close()

        _write_response_to_shm(mock_shm, "task-1", mp4_path)

        mock_shm.write_response.assert_called_once()
        resp = mock_shm.write_response.call_args[0][0]
        assert resp.status == VideoStatus.SUCCESS
        assert resp.task_id == "task-1"
        assert resp.file_path == mp4_path
        assert resp.error_message == ""

    def test_accepts_any_string_path(self):
        mock_shm = MagicMock()
        path = "/tmp/videos/coordinator-output.mp4"

        _write_response_to_shm(mock_shm, "task-2", path)

        resp = mock_shm.write_response.call_args[0][0]
        assert resp.file_path == path


class TestWriteErrorToShm:
    def test_writes_error_response(self):
        mock_shm = MagicMock()
        _write_error_to_shm(mock_shm, "task-err", "boom")

        mock_shm.write_response.assert_called_once()
        resp = mock_shm.write_response.call_args[0][0]
        assert resp.status == VideoStatus.ERROR
        assert resp.task_id == "task-err"
        assert resp.file_path == ""
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


class TestVideoRequestToGenerateRequest:
    def test_maps_only_fields_shared_with_video_generate_request(self):
        req = VideoRequest(
            task_id="t1",
            prompt="hello",
            negative_prompt="blurry",
            num_inference_steps=20,
            seed=42,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=3.0,
            guidance_scale_2=4.0,
        )
        gen = video_request_to_generate_request(req)
        assert gen.prompt == "hello"
        assert gen.negative_prompt == "blurry"
        assert gen.num_inference_steps == 20
        assert gen.seed == 42


class TestHandleSigterm:
    def test_sets_shutdown_flag(self):
        import tt_model_runners.video_runner as vr

        original = vr._shutdown
        vr._shutdown = False
        vr._handle_sigterm(None, None)
        assert vr._shutdown is True
        vr._shutdown = original


class TestMPIHelpers:
    def test_broadcast_request_calls_bcast(self):
        mock_comm = MagicMock()
        req = _make_request()
        mock_comm.bcast.return_value = req
        result = _broadcast_request(mock_comm, req)
        mock_comm.bcast.assert_called_once_with(req, root=0)
        assert result.task_id == "t1"

    def test_broadcast_none_signals_shutdown(self):
        mock_comm = MagicMock()
        mock_comm.bcast.return_value = None
        result = _broadcast_request(mock_comm, None)
        assert result is None


class TestAttachMpiComm:
    def test_success_attaches_comm_world(self):
        mock_mpi4py = MagicMock()
        with patch.dict(sys.modules, {"mpi4py": mock_mpi4py}):
            comm = _attach_mpi_comm()
        assert comm is mock_mpi4py.MPI.COMM_WORLD
        assert mock_mpi4py.rc.initialize is False
        assert mock_mpi4py.rc.finalize is False

    def test_raises_when_mpi4py_missing(self):
        real_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def _block_mpi4py(name, *args, **kwargs):
            if name == "mpi4py":
                raise ImportError("no mpi4py")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_mpi4py):
            with pytest.raises(RuntimeError, match="mpi4py is required"):
                _attach_mpi_comm()


class TestRunInferenceLoop:
    def test_rank0_successful_inference(self):
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        req = _make_request()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.bcast.side_effect = [req, None]

        mock_runner = MagicMock()
        mock_frames = MagicMock()
        mock_runner.run.return_value = mock_frames

        mock_input_shm = MagicMock()
        mock_input_shm.read_request.return_value = req
        mock_output_shm = MagicMock()

        mock_vm = MagicMock()
        mock_vm.export_to_mp4.return_value = "/tmp/videos/out.mp4"

        with patch("utils.video_manager.VideoManager", return_value=mock_vm):
            _run_inference_loop(mock_comm, mock_runner, mock_input_shm, mock_output_shm)

        mock_runner.run.assert_called_once()
        mock_vm.export_to_mp4.assert_called_once_with(mock_frames)
        mock_output_shm.write_response.assert_called_once()
        resp = mock_output_shm.write_response.call_args[0][0]
        assert resp.status == VideoStatus.SUCCESS
        assert resp.file_path == "/tmp/videos/out.mp4"

    def test_rank0_inference_error_writes_error(self):
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        req = _make_request()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.bcast.side_effect = [req, None]

        mock_runner = MagicMock()
        mock_runner.run.side_effect = RuntimeError("inference exploded")

        mock_input_shm = MagicMock()
        mock_input_shm.read_request.return_value = req
        mock_output_shm = MagicMock()

        _run_inference_loop(mock_comm, mock_runner, mock_input_shm, mock_output_shm)

        mock_output_shm.write_response.assert_called_once()
        resp = mock_output_shm.write_response.call_args[0][0]
        assert resp.status == VideoStatus.ERROR
        assert "inference exploded" in resp.error_message

    def test_none_request_breaks_loop(self):
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.bcast.return_value = None

        mock_runner = MagicMock()
        mock_input_shm = MagicMock()
        mock_input_shm.read_request.return_value = None
        mock_output_shm = MagicMock()

        _run_inference_loop(mock_comm, mock_runner, mock_input_shm, mock_output_shm)

        mock_runner.run.assert_not_called()

    def test_nonrank0_participates_in_inference(self):
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        req = _make_request()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.bcast.side_effect = [req, None]

        mock_runner = MagicMock()

        _run_inference_loop(mock_comm, mock_runner, None, None)

        mock_runner.run.assert_called_once()


class TestRunAllRanks:
    def test_exits_when_model_runner_not_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODEL_RUNNER", None)
            os.environ.pop("OMPI_COMM_WORLD_RANK", None)
            with pytest.raises(SystemExit):
                run_all_ranks()

    def test_rank0_full_lifecycle(self):
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        mock_runner = MagicMock()
        mock_runner.warmup = AsyncMock()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4
        mock_comm.bcast.return_value = None

        mock_input_shm = MagicMock()
        mock_input_shm.read_request.return_value = None
        mock_output_shm = MagicMock()

        env = {"MODEL_RUNNER": "tt-wan2.2", "OMPI_COMM_WORLD_RANK": "0"}
        with patch.dict(os.environ, env, clear=False), patch(
            "tt_model_runners.video_runner._create_dit_runner", return_value=mock_runner
        ), patch(
            "tt_model_runners.video_runner._attach_mpi_comm", return_value=mock_comm
        ), patch(
            "tt_model_runners.video_runner.VideoShm",
            side_effect=[mock_input_shm, mock_output_shm],
        ):
            run_all_ranks()

        mock_runner.set_device.assert_called_once()
        mock_runner.load_weights.assert_called_once()
        mock_runner.warmup.assert_called_once()
        mock_input_shm.open.assert_called_once_with(create=True)
        mock_output_shm.open.assert_called_once_with(create=True)
        mock_input_shm.close.assert_called_once()
        mock_input_shm.unlink.assert_called_once()
        mock_output_shm.close.assert_called_once()
        mock_output_shm.unlink.assert_called_once()
        mock_runner.close_device.assert_called_once()

    def test_rank0_cleanup_on_keyboard_interrupt(self):
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        mock_runner = MagicMock()
        mock_runner.warmup = AsyncMock()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4

        mock_input_shm = MagicMock()
        mock_output_shm = MagicMock()

        def _raise_interrupt(comm, runner, ishm, oshm):
            raise KeyboardInterrupt()

        env = {"MODEL_RUNNER": "tt-wan2.2", "OMPI_COMM_WORLD_RANK": "0"}
        with patch.dict(os.environ, env, clear=False), patch(
            "tt_model_runners.video_runner._create_dit_runner", return_value=mock_runner
        ), patch(
            "tt_model_runners.video_runner._attach_mpi_comm", return_value=mock_comm
        ), patch(
            "tt_model_runners.video_runner.VideoShm",
            side_effect=[mock_input_shm, mock_output_shm],
        ), patch(
            "tt_model_runners.video_runner._run_inference_loop",
            side_effect=_raise_interrupt,
        ):
            run_all_ranks()

        mock_input_shm.close.assert_called_once()
        mock_input_shm.unlink.assert_called_once()
        mock_output_shm.close.assert_called_once()
        mock_output_shm.unlink.assert_called_once()
        mock_runner.close_device.assert_called_once()

    def test_nonrank0_skips_shm(self):
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        mock_runner = MagicMock()
        mock_runner.warmup = AsyncMock()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.Get_size.return_value = 4
        mock_comm.bcast.return_value = None

        env = {"MODEL_RUNNER": "tt-wan2.2", "OMPI_COMM_WORLD_RANK": "1"}
        with patch.dict(os.environ, env, clear=False), patch(
            "tt_model_runners.video_runner._create_dit_runner", return_value=mock_runner
        ), patch(
            "tt_model_runners.video_runner._attach_mpi_comm", return_value=mock_comm
        ):
            run_all_ranks()

        mock_runner.set_device.assert_called_once()
        mock_runner.close_device.assert_called_once()


class TestMainEntryPoint:
    def test_main_calls_run_all_ranks(self):
        import tt_model_runners.video_runner as vr

        with patch.object(vr, "run_all_ranks") as mock_run:
            vr.main()
            mock_run.assert_called_once()


def _make_request(**overrides):
    defaults = dict(
        task_id="t1",
        prompt="hello world test prompt",
        negative_prompt="",
        num_inference_steps=20,
        seed=42,
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=3.0,
        guidance_scale_2=4.0,
    )
    defaults.update(overrides)
    return VideoRequest(**defaults)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
