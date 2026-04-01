# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import sys
from unittest.mock import MagicMock, Mock, patch

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
    _create_dit_runner,
    _is_shutdown,
    _rank,
    _write_error_to_shm,
    _write_response_to_shm,
    video_request_to_generate_request,
)
from ipc.video_shm import VideoRequest


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
        from tt_model_runners.video_runner import _broadcast_request

        mock_comm = MagicMock()
        req = VideoRequest(
            task_id="t1",
            prompt="hello",
            negative_prompt="",
            num_inference_steps=20,
            seed=42,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=3.0,
            guidance_scale_2=4.0,
        )
        mock_comm.bcast.return_value = req
        result = _broadcast_request(mock_comm, req)
        mock_comm.bcast.assert_called_once_with(req, root=0)
        assert result.task_id == "t1"

    def test_broadcast_none_signals_shutdown(self):
        from tt_model_runners.video_runner import _broadcast_request

        mock_comm = MagicMock()
        mock_comm.bcast.return_value = None
        result = _broadcast_request(mock_comm, None)
        assert result is None


class TestMainEntryPoint:
    def test_main_calls_run_all_ranks(self):
        import tt_model_runners.video_runner as vr

        with patch.object(vr, "run_all_ranks") as mock_run:
            vr.main()
            mock_run.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
