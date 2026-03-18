# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import socket
import sys
import threading
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

sys.modules["ttnn"] = Mock()

mock_settings = Mock()
mock_settings.enable_telemetry = False
mock_settings.model_runner = "sp_runner"
mock_settings_module = Mock()
mock_settings_module.settings = mock_settings
mock_settings_module.get_settings = Mock(return_value=mock_settings)
sys.modules["config.settings"] = mock_settings_module
sys.modules["telemetry.telemetry_client"] = Mock()
sys.modules["utils.logger"] = Mock()
sys.modules["utils.logger"].TTLogger = Mock(return_value=Mock())

from ipc.video_shm import FrameStatus, VideoRequest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tt_model_runners.video_runner import (
    _is_shutdown,
    _rank,
    _send_via_socket,
    _recv_via_socket,
    _write_frames_to_shm,
    _write_error_to_shm,
    create_dit_runner,
    RANK_CONFIG,
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


class TestRankConfig:
    def test_has_four_ranks(self):
        assert len(RANK_CONFIG) == 4
        for r in [0, 1, 2, 3]:
            assert r in RANK_CONFIG
            assert "ip" in RANK_CONFIG[r]
            assert "port" in RANK_CONFIG[r]


class TestSocketHelpers:
    def test_send_recv_roundtrip(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        port = server_sock.getsockname()[1]
        server_sock.listen(1)

        req = VideoRequest(
            task_id="test-task-id",
            prompt="hello world",
            negative_prompt="bad",
            num_inference_steps=20,
            seed=42,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=3.0,
            guidance_scale_2=4.0,
        )

        received = []

        def server_fn():
            conn, _ = server_sock.accept()
            got = _recv_via_socket(conn)
            received.append(got)
            conn.close()

        t = threading.Thread(target=server_fn)
        t.start()

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        _send_via_socket(client, req)
        client.close()
        t.join(timeout=5.0)
        server_sock.close()

        assert len(received) == 1
        got = received[0]
        assert got.task_id == req.task_id
        assert got.prompt == req.prompt
        assert got.num_inference_steps == 20
        assert got.seed == 42

    def test_recv_returns_none_on_closed_connection(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        port = server_sock.getsockname()[1]
        server_sock.listen(1)

        received = []

        def server_fn():
            conn, _ = server_sock.accept()
            got = _recv_via_socket(conn)
            received.append(got)
            conn.close()

        t = threading.Thread(target=server_fn)
        t.start()

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        client.close()
        t.join(timeout=5.0)
        server_sock.close()

        assert received == [None]


class TestWriteFramesToShm:
    def test_writes_frames_and_done(self):
        mock_shm = MagicMock()
        frames = np.random.randint(0, 256, (1, 3, 4, 6, 3), dtype=np.uint8)

        _write_frames_to_shm(mock_shm, "task-1", frames)

        assert mock_shm.write_frame.call_count == 4  # 3 FRAME + 1 DONE
        calls = mock_shm.write_frame.call_args_list

        for i in range(3):
            fr = calls[i][0][0]
            assert fr.status == FrameStatus.FRAME
            assert fr.frame_index == i
            assert fr.total_frames == 3
            assert fr.task_id == "task-1"
            assert fr.height == 4
            assert fr.width == 6
            assert fr.channels == 3

        done = calls[3][0][0]
        assert done.status == FrameStatus.DONE
        assert done.frame_data == b""

    def test_handles_4d_input(self):
        mock_shm = MagicMock()
        frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)

        _write_frames_to_shm(mock_shm, "task-2", frames)

        assert mock_shm.write_frame.call_count == 3  # 2 FRAME + 1 DONE

    def test_converts_float_to_uint8(self):
        mock_shm = MagicMock()
        frames = np.ones((1, 1, 2, 2, 3), dtype=np.float32) * 0.5

        _write_frames_to_shm(mock_shm, "task-3", frames)

        fr = mock_shm.write_frame.call_args_list[0][0][0]
        expected_val = int(0.5 * 255)
        assert fr.frame_data[0] == expected_val

    def test_clips_float_values(self):
        mock_shm = MagicMock()
        frames = np.array([[[[[2.0, -1.0, 0.5]]]]]).astype(np.float32)

        _write_frames_to_shm(mock_shm, "task-4", frames)

        fr = mock_shm.write_frame.call_args_list[0][0][0]
        assert fr.frame_data[0] == 255
        assert fr.frame_data[1] == 0
        assert fr.frame_data[2] == 127


class TestWriteErrorToShm:
    def test_writes_error_frame(self):
        mock_shm = MagicMock()
        _write_error_to_shm(mock_shm, "task-err")

        mock_shm.write_frame.assert_called_once()
        fr = mock_shm.write_frame.call_args[0][0]
        assert fr.status == FrameStatus.ERROR
        assert fr.task_id == "task-err"
        assert fr.frame_data == b""


class TestCreateDitRunner:
    def test_creates_mochi_runner(self):
        mock_mochi = Mock()
        mock_wan = Mock()
        with patch(
            "tt_model_runners.dit_runners.TTMochi1Runner", mock_mochi, create=True
        ), patch("tt_model_runners.dit_runners.TTWan22Runner", mock_wan, create=True):
            create_dit_runner("tt_mochi_1", "0")
            mock_mochi.assert_called_once_with("0")

    def test_creates_wan_runner(self):
        mock_mochi = Mock()
        mock_wan = Mock()
        with patch(
            "tt_model_runners.dit_runners.TTMochi1Runner", mock_mochi, create=True
        ), patch("tt_model_runners.dit_runners.TTWan22Runner", mock_wan, create=True):
            create_dit_runner("tt_wan_2_2", "1")
            mock_wan.assert_called_once_with("1")

    def test_raises_on_unsupported_runner(self):
        mock_mochi = Mock()
        mock_wan = Mock()
        with patch(
            "tt_model_runners.dit_runners.TTMochi1Runner", mock_mochi, create=True
        ), patch("tt_model_runners.dit_runners.TTWan22Runner", mock_wan, create=True):
            with pytest.raises(ValueError, match="Unsupported MODEL_RUNNER"):
                create_dit_runner("invalid_runner", "0")


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
