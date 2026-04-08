# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import socket
import sys
import threading
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
    RANK_CONFIG,
    _is_shutdown,
    _rank,
    _recv_via_socket,
    _send_via_socket,
    _write_error_to_shm,
    _write_response_to_shm,
    create_dit_runner,
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

    def test_recv_via_socket_reads_payload_in_multiple_chunks(self):
        """Covers the recv loop in _recv_via_socket (multi-chunk body)."""
        import pickle
        import struct

        import tt_model_runners.video_runner as vr

        req = VideoRequest(
            task_id="abcdef12-3456-7890-abcd-ef1234567890",
            prompt="hello",
            negative_prompt="",
            num_inference_steps=20,
            seed=1,
            height=480,
            width=832,
            num_frames=4,
            guidance_scale=3.0,
            guidance_scale_2=4.0,
        )
        payload = pickle.dumps(req)
        length_bytes = struct.pack("<I", len(payload))
        chunk_a = payload[: max(1, len(payload) // 2)]
        chunk_b = payload[len(chunk_a) :]

        conn = MagicMock()
        conn.recv = Mock(side_effect=[length_bytes, chunk_a, chunk_b])
        got = vr._recv_via_socket(conn)
        assert got is not None
        assert got.task_id == req.task_id


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
            create_dit_runner("tt-mochi-1", "0")
            mock_mochi.assert_called_once_with("0")

    def test_creates_wan_runner(self):
        mock_mochi = Mock()
        mock_wan = Mock()
        with patch.dict(sys.modules, self._make_dit_module(mock_mochi, mock_wan)):
            create_dit_runner("tt-wan2.2", "1")
            mock_wan.assert_called_once_with("1")

    def test_raises_on_unsupported_runner(self):
        mock_mochi = Mock()
        mock_wan = Mock()
        with patch.dict(sys.modules, self._make_dit_module(mock_mochi, mock_wan)):
            with pytest.raises(ValueError, match="Unsupported MODEL_RUNNER"):
                create_dit_runner("invalid_runner", "0")


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


class TestVideoRunnerCoverage:
    """Extra branch coverage for main(), SHM loop, sockets, bootstrap."""

    def test_recv_via_socket_returns_none_on_exception(self):
        import tt_model_runners.video_runner as vr

        bad = MagicMock()
        bad.recv.side_effect = ConnectionError("closed")
        with patch.object(vr._log, "warning") as w:
            assert vr._recv_via_socket(bad) is None
        w.assert_called_once()

    def test_connect_to_workers_all_fail(self):
        import tt_model_runners.video_runner as vr

        def fail_socket(*a, **k):
            s = MagicMock()
            s.connect.side_effect = ConnectionRefusedError()
            return s

        with patch.object(vr.socket, "socket", side_effect=fail_socket):
            out = vr._connect_to_workers()
        assert out == {}

    def test_connect_to_workers_partial_success(self):
        """At least one worker connects; failed connects still close the socket."""
        import tt_model_runners.video_runner as vr

        n = [0]

        def socket_factory(*a, **k):
            s = MagicMock()
            n[0] += 1
            if n[0] == 1:
                s.connect.return_value = None
            else:
                s.connect.side_effect = ConnectionRefusedError()
            return s

        with patch.object(vr.socket, "socket", side_effect=socket_factory):
            out = vr._connect_to_workers()
        assert len(out) >= 1
        assert 1 in out

    def test_bootstrap_exits_when_model_runner_missing(self):
        import tt_model_runners.video_runner as vr

        with patch.dict(os.environ, {"MODEL_RUNNER": ""}, clear=False):
            with patch.object(vr.sys, "exit", side_effect=RuntimeError("exit")):
                with pytest.raises(RuntimeError, match="exit"):
                    vr._bootstrap_dit_runner(0, "")

    def test_bootstrap_dit_runner_runs_full_setup(self):
        """Covers create_dit_runner, set_device, load_weights, asyncio.run(warmup)."""
        import tt_model_runners.video_runner as vr

        mock_runner = MagicMock()
        mock_runner.warmup = AsyncMock(return_value=True)
        with patch.dict(
            os.environ,
            {"MODEL_RUNNER": "tt-mochi-1", "TT_VISIBLE_DEVICES": "0"},
            clear=False,
        ):
            with patch(
                "tt_model_runners.video_runner.create_dit_runner",
                return_value=mock_runner,
            ):
                result = vr._bootstrap_dit_runner(0, "0")
        assert result is mock_runner
        mock_runner.set_device.assert_called_once()
        mock_runner.load_weights.assert_called_once()
        mock_runner.warmup.assert_called_once()

    @patch("tt_model_runners.video_runner.cleanup_orphaned_video_files", return_value=0)
    @patch("tt_model_runners.video_runner._connect_to_workers", return_value={})
    @patch("tt_model_runners.video_runner.time.sleep")
    @patch("tt_model_runners.video_runner.VideoShm")
    @patch("tt_model_runners.video_runner._bootstrap_dit_runner")
    def test_run_rank0_exits_when_read_returns_none(
        self,
        mock_boot,
        mock_vshm,
        mock_sleep,
        mock_conn,
        mock_clean,
    ):
        import tt_model_runners.video_runner as vr

        mock_runner = MagicMock()
        mock_boot.return_value = mock_runner
        mock_in = MagicMock()
        mock_out = MagicMock()
        mock_in.read_request = Mock(return_value=None)
        mock_vshm.side_effect = [mock_in, mock_out]
        vr.run_rank0_coordinator()
        mock_runner.close_device.assert_called()
        mock_in.close.assert_called()
        mock_out.close.assert_called()

    @patch("tt_model_runners.video_runner.cleanup_orphaned_video_files", return_value=0)
    @patch("tt_model_runners.video_runner._connect_to_workers", return_value={})
    @patch("tt_model_runners.video_runner.time.sleep")
    @patch("tt_model_runners.video_runner.VideoShm")
    @patch("tt_model_runners.video_runner._bootstrap_dit_runner")
    def test_run_rank0_inference_error_writes_error_shm(
        self,
        mock_boot,
        mock_vshm,
        mock_sleep,
        mock_conn,
        mock_clean,
    ):
        import tt_model_runners.video_runner as vr

        req = VideoRequest(
            task_id="abcdef12-3456-7890-abcd-ef1234567890",
            prompt="p",
            negative_prompt="",
            num_inference_steps=12,
            seed=0,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=3.0,
            guidance_scale_2=4.0,
        )
        mock_runner = MagicMock()
        mock_runner.run.side_effect = RuntimeError("inference failed")
        mock_boot.return_value = mock_runner
        mock_in = MagicMock()
        mock_out = MagicMock()
        mock_in.read_request = Mock(side_effect=[req, None])
        mock_vshm.side_effect = [mock_in, mock_out]
        vr.run_rank0_coordinator()
        err_calls = [
            c
            for c in mock_out.write_response.call_args_list
            if c[0][0].status == VideoStatus.ERROR
        ]
        assert len(err_calls) == 1
        mock_runner.close_device.assert_called()

    @patch("tt_model_runners.video_runner.cleanup_orphaned_video_files", return_value=0)
    @patch("tt_model_runners.video_runner._connect_to_workers", return_value={})
    @patch("tt_model_runners.video_runner.time.sleep")
    @patch("tt_model_runners.video_runner.VideoShm")
    @patch("tt_model_runners.video_runner._bootstrap_dit_runner")
    def test_run_rank0_success_writes_success_response(
        self,
        mock_boot,
        mock_vshm,
        mock_sleep,
        mock_conn,
        mock_clean,
    ):
        import tt_model_runners.video_runner as vr

        req = VideoRequest(
            task_id="abcdef12-3456-7890-abcd-ef1234567890",
            prompt="p",
            negative_prompt="",
            num_inference_steps=12,
            seed=0,
            height=480,
            width=832,
            num_frames=4,
            guidance_scale=3.0,
            guidance_scale_2=4.0,
        )
        mock_runner = MagicMock()
        mock_runner.run.return_value = [MagicMock()]
        mock_boot.return_value = mock_runner
        mock_in = MagicMock()
        mock_out = MagicMock()
        mock_in.read_request = Mock(side_effect=[req, None])
        mock_vshm.side_effect = [mock_in, mock_out]
        with patch("utils.video_manager.VideoManager") as VM:
            VM.return_value.export_to_mp4.return_value = "/tmp/coord_ok.mp4"
            vr.run_rank0_coordinator()
        success = [
            c[0][0]
            for c in mock_out.write_response.call_args_list
            if c[0][0].status == VideoStatus.SUCCESS
        ]
        assert len(success) == 1
        assert success[0].file_path == "/tmp/coord_ok.mp4"

    @patch("tt_model_runners.video_runner.cleanup_orphaned_video_files", return_value=2)
    @patch("tt_model_runners.video_runner._connect_to_workers", return_value={})
    @patch("tt_model_runners.video_runner.time.sleep")
    @patch("tt_model_runners.video_runner.VideoShm")
    @patch("tt_model_runners.video_runner._bootstrap_dit_runner")
    def test_run_rank0_logs_cleanup_when_orphans_removed(
        self,
        mock_boot,
        mock_vshm,
        mock_sleep,
        mock_conn,
        mock_clean,
    ):
        import tt_model_runners.video_runner as vr

        mock_boot.return_value = MagicMock()
        mock_in = MagicMock()
        mock_out = MagicMock()
        mock_in.read_request = Mock(return_value=None)
        mock_vshm.side_effect = [mock_in, mock_out]
        with patch.object(vr._log, "info") as log_info:
            vr.run_rank0_coordinator()
        assert any("2" in str(c) for c in log_info.call_args_list)

    @patch("tt_model_runners.video_runner.cleanup_orphaned_video_files", return_value=0)
    @patch("tt_model_runners.video_runner._connect_to_workers")
    @patch("tt_model_runners.video_runner.time.sleep")
    @patch("tt_model_runners.video_runner.VideoShm")
    @patch("tt_model_runners.video_runner._bootstrap_dit_runner")
    def test_run_rank0_warns_when_worker_send_fails(
        self,
        mock_boot,
        mock_vshm,
        mock_sleep,
        mock_conn_workers,
        mock_clean,
    ):
        import tt_model_runners.video_runner as vr

        bad = MagicMock()
        bad.sendall.side_effect = OSError("broken pipe")
        mock_conn_workers.return_value = {1: bad}
        req = VideoRequest(
            task_id="abcdef12-3456-7890-abcd-ef1234567890",
            prompt="p",
            negative_prompt="",
            num_inference_steps=12,
            seed=0,
            height=480,
            width=832,
            num_frames=4,
            guidance_scale=3.0,
            guidance_scale_2=4.0,
        )
        mock_boot.return_value = MagicMock()
        mock_in = MagicMock()
        mock_out = MagicMock()
        mock_in.read_request = Mock(side_effect=[req, None])
        mock_vshm.side_effect = [mock_in, mock_out]
        with patch("utils.video_manager.VideoManager") as VM:
            VM.return_value.export_to_mp4.return_value = "/tmp/out.mp4"
            with patch.object(vr._log, "warning") as w:
                vr.run_rank0_coordinator()
        w.assert_called()

    @patch("tt_model_runners.video_runner.cleanup_orphaned_video_files", return_value=0)
    @patch("tt_model_runners.video_runner._connect_to_workers", return_value={})
    @patch("tt_model_runners.video_runner.time.sleep")
    @patch("tt_model_runners.video_runner.VideoShm")
    @patch("tt_model_runners.video_runner._bootstrap_dit_runner")
    def test_run_rank0_keyboard_interrupt(
        self,
        mock_boot,
        mock_vshm,
        mock_sleep,
        mock_conn,
        mock_clean,
    ):
        import tt_model_runners.video_runner as vr

        mock_boot.return_value = MagicMock()
        mock_in = MagicMock()
        mock_out = MagicMock()
        mock_in.read_request = Mock(side_effect=KeyboardInterrupt())
        mock_vshm.side_effect = [mock_in, mock_out]
        vr.run_rank0_coordinator()

    @patch("tt_model_runners.video_runner._bootstrap_dit_runner")
    @patch("tt_model_runners.video_runner.socket.socket")
    def test_run_worker_rank_recv_none_closes_conn(self, mock_sock_cls, mock_boot):
        import tt_model_runners.video_runner as vr

        mock_runner = MagicMock()
        mock_boot.return_value = mock_runner
        mock_listen = MagicMock()
        mock_conn = MagicMock()
        mock_listen.accept = MagicMock(return_value=(mock_conn, ("127.0.0.1", 12345)))
        mock_sock_cls.return_value = mock_listen
        with patch.object(vr, "_recv_via_socket", return_value=None):
            vr.run_worker_rank(1)
        mock_conn.close.assert_called()
        mock_runner.close_device.assert_called()

    @patch("tt_model_runners.video_runner._bootstrap_dit_runner")
    @patch("tt_model_runners.video_runner.socket.socket")
    def test_run_worker_rank_runs_inference_then_stops(self, mock_sock_cls, mock_boot):
        import tt_model_runners.video_runner as vr

        req = VideoRequest(
            task_id="abcdef12-3456-7890-abcd-ef1234567890",
            prompt="p",
            negative_prompt="",
            num_inference_steps=12,
            seed=0,
            height=480,
            width=832,
            num_frames=4,
            guidance_scale=3.0,
            guidance_scale_2=4.0,
        )
        mock_runner = MagicMock()
        mock_boot.return_value = mock_runner
        mock_listen = MagicMock()
        mock_conn = MagicMock()
        mock_listen.accept = MagicMock(return_value=(mock_conn, ("127.0.0.1", 12345)))
        mock_sock_cls.return_value = mock_listen
        with patch.object(vr, "_recv_via_socket", side_effect=[req, None]):
            vr.run_worker_rank(2)
        mock_runner.run.assert_called_once()

    @patch("tt_model_runners.video_runner._bootstrap_dit_runner")
    @patch("tt_model_runners.video_runner.socket.socket")
    def test_run_worker_rank_inference_error_logs(self, mock_sock_cls, mock_boot):
        import tt_model_runners.video_runner as vr

        req = VideoRequest(
            task_id="abcdef12-3456-7890-abcd-ef1234567890",
            prompt="p",
            negative_prompt="",
            num_inference_steps=12,
            seed=0,
            height=480,
            width=832,
            num_frames=4,
            guidance_scale=3.0,
            guidance_scale_2=4.0,
        )
        mock_runner = MagicMock()
        mock_runner.run.side_effect = RuntimeError("worker infer failed")
        mock_boot.return_value = mock_runner
        mock_listen = MagicMock()
        mock_conn = MagicMock()
        mock_listen.accept = MagicMock(return_value=(mock_conn, ("127.0.0.1", 12345)))
        mock_sock_cls.return_value = mock_listen
        with patch.object(vr, "_recv_via_socket", side_effect=[req, None]):
            with patch.object(vr._log, "error") as err:
                vr.run_worker_rank(1)
        err.assert_called()

    @patch("tt_model_runners.video_runner._bootstrap_dit_runner")
    @patch("tt_model_runners.video_runner.socket.socket")
    def test_run_worker_rank_keyboard_interrupt(self, mock_sock_cls, mock_boot):
        import tt_model_runners.video_runner as vr

        mock_listen = MagicMock()
        mock_conn = MagicMock()
        mock_listen.accept = MagicMock(return_value=(mock_conn, ("127.0.0.1", 12345)))
        mock_sock_cls.return_value = mock_listen
        with patch.object(vr, "_recv_via_socket", side_effect=KeyboardInterrupt()):
            with patch.object(vr._log, "info") as log_info:
                vr.run_worker_rank(1)
        assert any("Interrupt" in str(c) for c in log_info.call_args_list)

    @patch("tt_model_runners.video_runner._bootstrap_dit_runner")
    @patch("tt_model_runners.video_runner.socket.socket")
    def test_run_worker_rank_outer_exception(self, mock_sock_cls, mock_boot):
        import tt_model_runners.video_runner as vr

        mock_boot.return_value = MagicMock()
        mock_listen = MagicMock()
        mock_listen.accept.side_effect = RuntimeError("bind failed")
        mock_sock_cls.return_value = mock_listen
        vr.run_worker_rank(1)

    def test_main_invalid_rank_exits(self):
        import tt_model_runners.video_runner as vr

        with patch.object(vr, "_rank", return_value=99):
            with patch.object(vr.sys, "exit") as ex:
                vr.main()
                ex.assert_called_once_with(1)

    def test_main_dispatches_rank0(self):
        import tt_model_runners.video_runner as vr

        with patch.object(vr, "_rank", return_value=0):
            with patch.object(vr, "run_rank0_coordinator") as r0:
                vr.main()
                r0.assert_called_once()

    def test_main_dispatches_worker(self):
        import tt_model_runners.video_runner as vr

        with patch.object(vr, "_rank", return_value=2):
            with patch.object(vr, "run_worker_rank") as rw:
                vr.main()
                rw.assert_called_once_with(2)

    def test_recv_via_socket_unpickling_error_returns_none(self):
        """Invalid pickle body hits except Exception in _recv_via_socket (lines 153–155)."""
        import struct

        import tt_model_runners.video_runner as vr

        conn = MagicMock()
        payload = b"\xffnot-valid-pickle"
        conn.recv = Mock(side_effect=[struct.pack("<I", len(payload)), payload])
        with patch.object(vr._log, "warning") as w:
            assert vr._recv_via_socket(conn) is None
        w.assert_called_once()

    def test_bootstrap_dit_runner_logs_model_and_visible_devices(self):
        """Covers Rank N: model=..., device=... (lines 190–191)."""
        import tt_model_runners.video_runner as vr

        mock_runner = MagicMock()
        mock_runner.warmup = AsyncMock(return_value=True)
        with patch.dict(
            os.environ,
            {
                "MODEL_RUNNER": "tt-mochi-1",
                "TT_VISIBLE_DEVICES": "0,1",
            },
            clear=False,
        ):
            with patch(
                "tt_model_runners.video_runner.create_dit_runner",
                return_value=mock_runner,
            ):
                with patch.object(vr._log, "info") as log_info:
                    vr._bootstrap_dit_runner(2, "dev-x")
        texts = [str(c) for c in log_info.call_args_list]
        assert any("model=tt-mochi-1" in t and "device=0,1" in t for t in texts)

    def test_run_worker_rank_outer_exception_on_accept(self):
        """accept() raises → outer except logs (lines 375–376)."""
        import tt_model_runners.video_runner as vr

        mock_listen = MagicMock()
        mock_listen.accept.side_effect = RuntimeError("accept failed")
        mock_listen.bind = MagicMock()
        mock_listen.listen = MagicMock()
        mock_listen.setsockopt = MagicMock()
        mock_listen.close = MagicMock()

        with patch.object(vr.socket, "socket", return_value=mock_listen):
            with patch.object(vr, "_bootstrap_dit_runner", return_value=MagicMock()):
                with patch.object(vr._log, "error") as err:
                    vr.run_worker_rank(1)
        err.assert_called()
        assert any("accept failed" in str(c) for c in err.call_args_list)

    def test_main_logs_starting_rank(self):
        """Covers Starting video runner with rank=... (line 391)."""
        import tt_model_runners.video_runner as vr

        with patch.object(vr, "_rank", return_value=0):
            with patch.object(vr, "run_rank0_coordinator"):
                with patch.object(vr._log, "info") as log_info:
                    vr.main()
        assert any(
            "Starting video runner" in str(c) and "rank=" in str(c)
            for c in log_info.call_args_list
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
