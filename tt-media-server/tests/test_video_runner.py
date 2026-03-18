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

from ipc.video_shm import VideoRequest, VideoStatus

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tt_model_runners.video_runner import (
    _is_shutdown,
    _rank,
    _send_via_socket,
    _recv_via_socket,
    _write_response_to_shm,
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


class TestWriteResponseToShm:
    def test_writes_single_blob(self):
        mock_shm = MagicMock()
        frames = np.random.randint(0, 256, (1, 3, 4, 6, 3), dtype=np.uint8)

        _write_response_to_shm(mock_shm, "task-1", frames)

        mock_shm.write_response.assert_called_once()
        resp = mock_shm.write_response.call_args[0][0]
        assert resp.status == VideoStatus.SUCCESS
        assert resp.task_id == "task-1"
        assert resp.num_frames == 3
        assert resp.height == 4
        assert resp.width == 6
        assert resp.channels == 3
        assert len(resp.frame_data) == 3 * 4 * 6 * 3

    def test_handles_4d_input(self):
        mock_shm = MagicMock()
        frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)

        _write_response_to_shm(mock_shm, "task-2", frames)

        mock_shm.write_response.assert_called_once()
        resp = mock_shm.write_response.call_args[0][0]
        assert resp.num_frames == 2

    def test_converts_float_to_uint8(self):
        mock_shm = MagicMock()
        frames = np.ones((1, 1, 2, 2, 3), dtype=np.float32) * 0.5

        _write_response_to_shm(mock_shm, "task-3", frames)

        resp = mock_shm.write_response.call_args[0][0]
        expected_val = int(0.5 * 255)
        assert resp.frame_data[0] == expected_val

    def test_clips_float_values(self):
        mock_shm = MagicMock()
        frames = np.array([[[[[2.0, -1.0, 0.5]]]]]).astype(np.float32)

        _write_response_to_shm(mock_shm, "task-4", frames)

        resp = mock_shm.write_response.call_args[0][0]
        assert resp.frame_data[0] == 255
        assert resp.frame_data[1] == 0
        assert resp.frame_data[2] == 127


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
