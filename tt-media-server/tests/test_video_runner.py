# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
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
from ipc.video_shm import VideoRequest, VideoStatus

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tt_model_runners.video_runner import (
    _attach_mpi_comm,
    _broadcast_request,
    _create_dit_runner,
    _is_shutdown,
    _rank,
    _rank0_load_image_prompts,
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
    def _make_dit_module(
        mock_mochi, mock_wan, mock_wan_i2v=None, mock_wan_i2v_prodia=None
    ):
        mock_mochi.__name__ = "TTMochi1Runner"
        mock_wan.__name__ = "TTWan22Runner"
        mod = Mock()
        mod.TTMochi1Runner = mock_mochi
        mod.TTWan22Runner = mock_wan
        if mock_wan_i2v is not None:
            mock_wan_i2v.__name__ = "TTWan22I2VRunner"
            mod.TTWan22I2VRunner = mock_wan_i2v
        if mock_wan_i2v_prodia is not None:
            mock_wan_i2v_prodia.__name__ = "TTWan22I2VProdiaRunner"
            mod.TTWan22I2VProdiaRunner = mock_wan_i2v_prodia
        return {"tt_model_runners.dit_runners": mod}

    def test_creates_mochi_runner(self):
        mock_mochi = Mock()
        mock_wan = Mock()
        mock_wan_i2v = Mock()
        with patch.dict(
            sys.modules, self._make_dit_module(mock_mochi, mock_wan, mock_wan_i2v)
        ):
            _create_dit_runner("tt-mochi-1", 0)
            mock_mochi.assert_called_once_with("")

    def test_creates_wan_runner(self):
        mock_mochi = Mock()
        mock_wan = Mock()
        mock_wan_i2v = Mock()
        with patch.dict(
            sys.modules, self._make_dit_module(mock_mochi, mock_wan, mock_wan_i2v)
        ):
            _create_dit_runner("tt-wan2.2", 1)
            mock_wan.assert_called_once_with("")

    def test_creates_wan_i2v_runner(self):
        """``tt-wan2.2-i2v`` must resolve to ``TTWan22I2VRunner`` so the
        multi-host MPI entrypoint can serve I2V requests at all."""
        mock_mochi = Mock()
        mock_wan = Mock()
        mock_wan_i2v = Mock()
        with patch.dict(
            sys.modules, self._make_dit_module(mock_mochi, mock_wan, mock_wan_i2v)
        ):
            _create_dit_runner("tt-wan2.2-i2v", 0)
            mock_wan_i2v.assert_called_once_with("")
            mock_wan.assert_not_called()
            mock_mochi.assert_not_called()

    def test_creates_wan_i2v_prodia_runner(self):
        """``tt-wan2.2-i2v-prodia`` must resolve to ``TTWan22I2VProdiaRunner``
        — the distilled 3-step variant. Both I2V variants must be selectable
        via MODEL_RUNNER so operators can swap quality vs. latency without
        rebuilding."""
        mock_mochi = Mock()
        mock_wan = Mock()
        mock_wan_i2v = Mock()
        mock_wan_i2v_prodia = Mock()
        with patch.dict(
            sys.modules,
            self._make_dit_module(
                mock_mochi, mock_wan, mock_wan_i2v, mock_wan_i2v_prodia
            ),
        ):
            _create_dit_runner("tt-wan2.2-i2v-prodia", 0)
            mock_wan_i2v_prodia.assert_called_once_with("")
            mock_wan_i2v.assert_not_called()
            mock_wan.assert_not_called()
            mock_mochi.assert_not_called()

    def test_raises_on_unsupported_runner(self):
        mock_mochi = Mock()
        mock_wan = Mock()
        mock_wan_i2v = Mock()
        with patch.dict(
            sys.modules, self._make_dit_module(mock_mochi, mock_wan, mock_wan_i2v)
        ):
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

    def test_returns_t2v_when_image_prompts_empty(self):
        """Empty / None ``image_prompts`` falls through to the T2V path so
        single-host T2V behaviour is byte-identical to before the change."""
        from domain.video_generate_request import VideoGenerateRequest
        from domain.video_i2v_generate_request import VideoI2VGenerateRequest

        req = _make_request()
        gen = video_request_to_generate_request(req, image_prompts=None)
        assert isinstance(gen, VideoGenerateRequest)
        assert not isinstance(gen, VideoI2VGenerateRequest)

        gen_empty = video_request_to_generate_request(req, image_prompts=[])
        assert isinstance(gen_empty, VideoGenerateRequest)
        assert not isinstance(gen_empty, VideoI2VGenerateRequest)

    def test_returns_i2v_when_image_prompts_non_empty(self):
        """Non-empty ``image_prompts`` triggers the I2V request type with
        the entries unpacked into ``ImagePromptEntry`` objects."""
        from domain.video_i2v_generate_request import (
            ImagePromptEntry,
            VideoI2VGenerateRequest,
        )

        req = _make_request()
        image_prompts = [
            {"image": _tiny_png_b64(), "frame_pos": 0},
            {"image": _tiny_png_b64(), "frame_pos": 40},
        ]
        gen = video_request_to_generate_request(req, image_prompts=image_prompts)

        assert isinstance(gen, VideoI2VGenerateRequest)
        assert len(gen.image_prompts) == 2
        assert all(isinstance(p, ImagePromptEntry) for p in gen.image_prompts)
        assert gen.image_prompts[0].frame_pos == 0
        assert gen.image_prompts[1].frame_pos == 40


class TestHandleSigterm:
    def test_sets_shutdown_flag(self):
        import tt_model_runners.video_runner as vr

        original = vr._shutdown
        vr._shutdown = False
        vr._handle_sigterm(None, None)
        assert vr._shutdown is True
        vr._shutdown = original


# NOTE: standalone happy-path / shutdown coverage for ``_broadcast_request``
# now lives in ``TestI2VBroadcast`` below — the helper's contract changed
# from "broadcast a single VideoRequest" to "broadcast a (request,
# image_prompts) tuple" so I2V conditioning images reach ranks 1..N. Keeping
# the old single-value assertions would be a contradiction with the new
# contract, so they have been replaced rather than amended.


class TestI2VBroadcast:
    """Multi-host I2V requires the conditioning image_prompts to reach all
    ranks alongside the text request. ``_broadcast_request`` is the chokepoint
    that bundles ``(req, image_prompts, skip)`` into a single MPI ``bcast``
    call. ``skip`` is the lockstep-skip signal for rank-0-detected errors
    (e.g. unreadable side-file); see ``_run_inference_loop``.
    """

    def test_broadcasts_request_and_image_prompts_tuple(self):
        mock_comm = MagicMock()
        req = _make_request()
        image_prompts = [{"image": _tiny_png_b64(), "frame_pos": 0}]
        mock_comm.bcast.return_value = (req, image_prompts, False)

        got_req, got_imgs, got_skip = _broadcast_request(
            mock_comm, req, image_prompts=image_prompts
        )

        mock_comm.bcast.assert_called_once_with((req, image_prompts, False), root=0)
        assert got_req is req
        assert got_imgs == image_prompts
        assert got_skip is False

    def test_broadcasts_request_with_no_image_prompts(self):
        """T2V backward compat: when ``image_prompts`` is omitted, the
        broadcast payload is ``(req, None, False)`` so non-rank-0 receivers
        still unpack a uniform tuple shape."""
        mock_comm = MagicMock()
        req = _make_request()
        mock_comm.bcast.return_value = (req, None, False)

        got_req, got_imgs, got_skip = _broadcast_request(mock_comm, req)

        mock_comm.bcast.assert_called_once_with((req, None, False), root=0)
        assert got_req is req
        assert got_imgs is None
        assert got_skip is False

    def test_broadcasts_none_pair_for_shutdown(self):
        """Shutdown signal: rank-0 ``read_request`` returned ``None``; the
        broadcast unwraps to ``(None, None, False)`` so all ranks break out
        of the inference loop in lock-step."""
        mock_comm = MagicMock()
        mock_comm.bcast.return_value = (None, None, False)

        got_req, got_imgs, got_skip = _broadcast_request(mock_comm, None)

        assert got_req is None
        assert got_imgs is None
        assert got_skip is False

    def test_broadcasts_skip_flag_when_set(self):
        """Skip flag must propagate through the broadcast so ranks 1..N
        can no-op the iteration in lockstep with rank 0 — otherwise they
        would block on collective ops with no peer."""
        mock_comm = MagicMock()
        req = _make_request()
        mock_comm.bcast.return_value = (req, [], True)

        got_req, got_imgs, got_skip = _broadcast_request(
            mock_comm, req, image_prompts=[], skip=True
        )

        mock_comm.bcast.assert_called_once_with((req, [], True), root=0)
        assert got_skip is True


class TestRank0LoadImagePrompts:
    """Direct coverage of the rank-0 side-file resolver. The end-to-end
    behaviour is also verified via ``TestI2VInferenceLoopSideFile`` below,
    but pinning the three branches at the helper level keeps the contract
    explicit and survives future refactors of the loop body."""

    def test_returns_none_false_for_t2v_request(self):
        import queue as _queue

        req = _make_request(image_path="")
        encode_queue: _queue.Queue = _queue.Queue()

        prompts, skip = _rank0_load_image_prompts(req, encode_queue)

        assert prompts is None
        assert skip is False
        assert encode_queue.qsize() == 0

    def test_returns_none_false_when_raw_req_is_none(self):
        """Shutdown iteration: rank 0 read None from input ring. The helper
        must not touch the queue or attempt any I/O."""
        import queue as _queue

        encode_queue: _queue.Queue = _queue.Queue()

        prompts, skip = _rank0_load_image_prompts(None, encode_queue)

        assert prompts is None
        assert skip is False
        assert encode_queue.qsize() == 0

    def test_returns_prompts_false_for_readable_side_file(self, tmp_path):
        import queue as _queue

        side_path = tmp_path / "tt_img_ok.json"
        payload = [{"image": _tiny_png_b64(), "frame_pos": 0}]
        side_path.write_text(json.dumps(payload))
        req = _make_request(image_path=str(side_path))
        encode_queue: _queue.Queue = _queue.Queue()

        prompts, skip = _rank0_load_image_prompts(req, encode_queue)

        assert prompts == payload
        assert skip is False
        assert encode_queue.qsize() == 0

    def test_returns_empty_true_and_enqueues_error_when_unreadable(self, tmp_path):
        """The fail-fast invariant: a missing/corrupt side-file must enqueue
        an error job AND return ``skip=True`` so the loop body broadcasts
        that flag and ranks 1..N no-op in lockstep."""
        import queue as _queue

        missing = tmp_path / "tt_img_missing.json"
        req = _make_request(image_path=str(missing))
        encode_queue: _queue.Queue = _queue.Queue()

        prompts, skip = _rank0_load_image_prompts(req, encode_queue)

        assert prompts == []
        assert skip is True
        assert encode_queue.qsize() == 1
        job = encode_queue.get_nowait()
        assert job.task_id == req.task_id
        assert job.error is not None
        assert "side-file unreadable" in job.error
        assert str(missing) in job.error

    def test_returns_empty_true_and_enqueues_error_for_empty_list_side_file(
        self, tmp_path
    ):
        """A side-file containing valid JSON ``[]`` is the only remaining
        silent-T2V degradation path after ``bea8dad``: the parse succeeds,
        ``prompts is not None``, but ``video_request_to_generate_request``'s
        ``if image_prompts:`` is falsy for empty lists and would build a
        T2V request for an I2V model. The helper must reject it the same
        way it rejects unreadable files."""
        import queue as _queue

        side_path = tmp_path / "tt_img_empty.json"
        side_path.write_text("[]")
        req = _make_request(image_path=str(side_path))
        encode_queue: _queue.Queue = _queue.Queue()

        prompts, skip = _rank0_load_image_prompts(req, encode_queue)

        assert prompts == []
        assert skip is True
        assert encode_queue.qsize() == 1
        job = encode_queue.get_nowait()
        assert job.task_id == req.task_id
        assert job.error is not None
        assert "empty list" in job.error
        assert str(side_path) in job.error


class TestI2VInferenceLoopSideFile:
    """End-to-end behaviour of ``_run_inference_loop`` when an I2V request
    references a side-file containing the JSON-serialised image_prompts."""

    def _run_loop_once(self, mock_comm, mock_runner, mock_input_shm):
        """Run the loop with a single iteration + shutdown sentinel."""
        import queue as _queue

        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        encode_queue: _queue.Queue = _queue.Queue()
        _run_inference_loop(mock_comm, mock_runner, mock_input_shm, encode_queue)
        return encode_queue

    def test_rank0_reads_side_file_and_broadcasts_tuple(self, tmp_path):
        """Rank 0 deserialises the side-file before broadcasting so all
        ranks receive identical conditioning images for collective inference.
        """
        side_path = tmp_path / "tt_img_task1.json"
        image_prompts = [
            {"image": _tiny_png_b64(), "frame_pos": 0},
            {"image": _tiny_png_b64(), "frame_pos": 40},
        ]
        side_path.write_text(json.dumps(image_prompts))

        req = _make_request(image_path=str(side_path))
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        # Iteration 1: real payload. Iteration 2: shutdown sentinel.
        mock_comm.bcast.side_effect = [
            (req, image_prompts, False),
            (None, None, False),
        ]

        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock()
        mock_input_shm = MagicMock()
        mock_input_shm.read_request.side_effect = [req, None]

        self._run_loop_once(mock_comm, mock_runner, mock_input_shm)

        first_call = mock_comm.bcast.call_args_list[0]
        payload = first_call.args[0] if first_call.args else first_call[0][0]
        assert payload[0].task_id == req.task_id
        assert payload[1] == image_prompts
        assert payload[2] is False

    def test_rank0_non_list_side_file_enqueues_error_and_skips(self, tmp_path):
        """Corrupt side-file (parses, but top-level shape isn't a list) must
        produce an explicit error response AND set ``skip=True`` on the
        broadcast so all ranks no-op this iteration. Falling back to ``[]``
        as a "clean T2V degrade" is wrong: ``TTWan22I2VRunner.run`` would
        then raise ``AttributeError`` on ``request.image_prompts`` at a
        downstream layer where the API has no useful context for the
        failure, while a fail-fast at the side-file boundary surfaces the
        actual cause (unreadable file)."""
        side_path = tmp_path / "tt_img_corrupt.json"
        side_path.write_text(json.dumps({"image": "b64", "frame_pos": 0}))

        req = _make_request(image_path=str(side_path))
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        # Iteration 1: skip=True (rank-0 error path). Iteration 2: shutdown.
        mock_comm.bcast.side_effect = [(req, [], True), (None, None, False)]

        mock_runner = MagicMock()
        mock_input_shm = MagicMock()
        mock_input_shm.read_request.side_effect = [req, None]

        encode_queue = self._run_loop_once(mock_comm, mock_runner, mock_input_shm)

        # Inference must NOT have run on rank 0 (or any rank) for this iter.
        mock_runner.run.assert_not_called()
        # And the broadcast carried skip=True so ranks 1..N also no-op.
        first_call = mock_comm.bcast.call_args_list[0]
        payload = first_call.args[0] if first_call.args else first_call[0][0]
        assert payload[2] is True
        # Encoder thread received an error job for this task id.
        assert encode_queue.qsize() == 1
        job = encode_queue.get_nowait()
        assert job.task_id == req.task_id
        assert job.error is not None
        assert "side-file unreadable" in job.error

    def test_rank0_missing_side_file_enqueues_error_and_skips(self, tmp_path):
        """Missing side-file (e.g. tmpfs reaped between SHM hand-off and
        runner read) takes the same fail-fast path as a corrupt one — the
        API must see a clear error, not a silent T2V output."""
        missing = tmp_path / "tt_img_does_not_exist.json"
        req = _make_request(image_path=str(missing))

        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.bcast.side_effect = [(req, [], True), (None, None, False)]

        mock_runner = MagicMock()
        mock_input_shm = MagicMock()
        mock_input_shm.read_request.side_effect = [req, None]

        encode_queue = self._run_loop_once(mock_comm, mock_runner, mock_input_shm)

        mock_runner.run.assert_not_called()
        first_call = mock_comm.bcast.call_args_list[0]
        payload = first_call.args[0] if first_call.args else first_call[0][0]
        assert payload[2] is True
        assert encode_queue.qsize() == 1
        job = encode_queue.get_nowait()
        assert job.task_id == req.task_id
        assert "side-file unreadable" in job.error

    def test_nonrank0_skips_inference_when_skip_flag_set(self, tmp_path):
        """Ranks 1..N must skip ``runner.run`` whenever the broadcast
        carries ``skip=True``, otherwise they'd block forever on collective
        ops with no rank-0 peer (rank 0 already exited the iteration via
        the encode_queue error path).

        Production passes ``encode_queue=None`` for non-rank-0 ranks (only
        rank 0 owns the queue); the skip path must not touch the queue on
        these ranks, so passing ``None`` here pins that contract."""
        req = _make_request(image_path="")
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.bcast.side_effect = [(req, [], True), (None, None, False)]

        mock_runner = MagicMock()
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        _run_inference_loop(mock_comm, mock_runner, None, None)

        mock_runner.run.assert_not_called()

    def test_rank0_skips_side_file_read_for_t2v(self, tmp_path, monkeypatch):
        """T2V request (``image_path == ""``) must not attempt any file I/O
        at all — keeps the T2V hot path identical to pre-I2V behaviour."""
        opens_seen = []
        real_open = open

        def tracking_open(path, *args, **kwargs):
            opens_seen.append(str(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", tracking_open)

        req = _make_request(image_path="")
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.bcast.side_effect = [(req, None, False), (None, None, False)]
        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock()
        mock_input_shm = MagicMock()
        mock_input_shm.read_request.side_effect = [req, None]

        self._run_loop_once(mock_comm, mock_runner, mock_input_shm)

        # The runner code itself must not have opened any tt_img_*.json file.
        assert not any("tt_img_" in p for p in opens_seen)


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
    def test_rank0_successful_inference_enqueues_for_encoder(self):
        """Successful inference on rank 0 hands frames off to the encoder
        queue. The inference loop itself must never write to ``output_shm``
        or call ``export_to_mp4`` — the encoder thread is the sole writer.

        Broadcast payload is the new ``(req, image_prompts)`` tuple even on
        the T2V hot path (image_prompts is None) so all ranks unpack a
        uniformly-shaped value from ``comm.bcast``.
        """
        import queue as _queue

        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        req = _make_request()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.bcast.side_effect = [(req, None, False), (None, None, False)]

        mock_runner = MagicMock()
        mock_frames = MagicMock()
        mock_runner.run.return_value = mock_frames

        mock_input_shm = MagicMock()
        mock_input_shm.read_request.return_value = req

        encode_queue: _queue.Queue = _queue.Queue()

        _run_inference_loop(mock_comm, mock_runner, mock_input_shm, encode_queue)

        mock_runner.run.assert_called_once()

        assert encode_queue.qsize() == 1
        job = encode_queue.get_nowait()
        assert job.task_id == req.task_id
        assert job.frames is mock_frames
        assert job.error is None

    def test_rank0_inference_error_enqueues_error_job(self):
        """Inference failure on rank 0 enqueues an error job; the encoder
        thread is the single writer of ``output_shm`` for both success and
        error paths, which preserves FIFO ordering under all failure modes.
        """
        import queue as _queue

        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        req = _make_request()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.bcast.side_effect = [(req, None, False), (None, None, False)]

        mock_runner = MagicMock()
        mock_runner.run.side_effect = RuntimeError("inference exploded")

        mock_input_shm = MagicMock()
        mock_input_shm.read_request.return_value = req

        encode_queue: _queue.Queue = _queue.Queue()

        _run_inference_loop(mock_comm, mock_runner, mock_input_shm, encode_queue)

        assert encode_queue.qsize() == 1
        job = encode_queue.get_nowait()
        assert job.task_id == req.task_id
        assert job.frames is None
        assert "inference exploded" in job.error

    def test_none_request_breaks_loop(self):
        """Shutdown sentinel: rank 0's ``read_request`` returned None and the
        broadcast unwraps to ``(None, None)`` — all ranks break out of the
        loop in lock-step without calling the runner."""
        import queue as _queue

        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.bcast.return_value = (None, None, False)

        mock_runner = MagicMock()
        mock_input_shm = MagicMock()
        mock_input_shm.read_request.return_value = None

        _run_inference_loop(mock_comm, mock_runner, mock_input_shm, _queue.Queue())

        mock_runner.run.assert_not_called()

    def test_nonrank0_participates_in_inference(self):
        """Non-rank-0 ranks contribute to distributed inference but never
        touch SHM or the encoder queue."""
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        req = _make_request()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.bcast.side_effect = [(req, None, False), (None, None, False)]

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
        # Shutdown payload is a (None, None, False) tuple — see _broadcast_request.
        mock_comm.bcast.return_value = (None, None, False)

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
        # Under the create-or-attach ownership model, open() takes no args and
        # the runner never unlinks — segments are owned by the operator via
        # `python -m ipc.video_shm_bootstrap down`.
        mock_input_shm.open.assert_called_once_with()
        mock_output_shm.open.assert_called_once_with()
        mock_input_shm.close.assert_called_once()
        mock_input_shm.unlink.assert_not_called()
        mock_output_shm.close.assert_called_once()
        mock_output_shm.unlink.assert_not_called()
        mock_runner.close_device.assert_called_once()

    def test_disables_export_in_runner_when_attribute_present(self):
        """When a runner opts into in-worker MP4 export (``export_in_runner``
        attribute), the MPI launcher must override it back to False so the
        encoder thread stays the sole writer to ``output_shm``. Otherwise the
        runner returns a path-string and the encoder receives garbage on the
        bcast hop."""
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        mock_runner = MagicMock()
        mock_runner.export_in_runner = True
        mock_runner.warmup = AsyncMock()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4
        mock_comm.bcast.return_value = (None, None, False)

        mock_input_shm = MagicMock()
        mock_input_shm.read_request.return_value = None
        mock_output_shm = MagicMock()

        env = {
            "MODEL_RUNNER": "tt-wan2.2-i2v-prodia",
            "OMPI_COMM_WORLD_RANK": "0",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "tt_model_runners.video_runner._create_dit_runner",
            return_value=mock_runner,
        ), patch(
            "tt_model_runners.video_runner._attach_mpi_comm",
            return_value=mock_comm,
        ), patch(
            "tt_model_runners.video_runner.VideoShm",
            side_effect=[mock_input_shm, mock_output_shm],
        ), patch(
            "tt_model_runners.video_runner.cleanup_orphaned_video_files",
            return_value=0,
        ):
            run_all_ranks()

        assert mock_runner.export_in_runner is False

    def test_export_in_runner_override_is_safe_for_runners_without_attribute(self):
        """Runners that never opted into in-worker export (e.g. plain T2V)
        must not gain an ``export_in_runner`` attribute via the override —
        ``hasattr`` keeps the override a no-op there."""
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        # Use spec= to make hasattr return False for export_in_runner.
        mock_runner = MagicMock(
            spec=["warmup", "set_device", "close_device", "run", "load_weights"]
        )
        mock_runner.warmup = AsyncMock()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4
        mock_comm.bcast.return_value = (None, None, False)

        mock_input_shm = MagicMock()
        mock_input_shm.read_request.return_value = None
        mock_output_shm = MagicMock()

        env = {"MODEL_RUNNER": "tt-wan2.2", "OMPI_COMM_WORLD_RANK": "0"}
        with patch.dict(os.environ, env, clear=False), patch(
            "tt_model_runners.video_runner._create_dit_runner",
            return_value=mock_runner,
        ), patch(
            "tt_model_runners.video_runner._attach_mpi_comm",
            return_value=mock_comm,
        ), patch(
            "tt_model_runners.video_runner.VideoShm",
            side_effect=[mock_input_shm, mock_output_shm],
        ), patch(
            "tt_model_runners.video_runner.cleanup_orphaned_video_files",
            return_value=0,
        ):
            run_all_ranks()

        assert not hasattr(mock_runner, "export_in_runner")

    def test_rank0_sweeps_orphaned_video_files_on_shutdown(self):
        """Mirrors mock_video_runner_base + sp_runner: rank 0 must sweep
        leftover mp4s on tmpfs at shutdown so a crash mid-encode doesn't
        leak files until the next SP_RUNNER restart."""
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        mock_runner = MagicMock()
        mock_runner.warmup = AsyncMock()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4
        mock_comm.bcast.return_value = (None, None, False)

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
        ), patch(
            "tt_model_runners.video_runner.cleanup_orphaned_video_files",
            return_value=3,
        ) as mock_cleanup:
            run_all_ranks()

        mock_cleanup.assert_called_once_with()

    def test_nonrank0_does_not_sweep_orphaned_video_files(self):
        """Sweep is rank-0-only; runner peers on other hosts share no tmpfs
        with the encoder thread, so calling cleanup there would either no-op
        or (worse) race-delete files belonging to a different rank-0."""
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        mock_runner = MagicMock()
        mock_runner.warmup = AsyncMock()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.Get_size.return_value = 4
        mock_comm.bcast.return_value = (None, None, False)

        env = {"MODEL_RUNNER": "tt-wan2.2", "OMPI_COMM_WORLD_RANK": "1"}
        with patch.dict(os.environ, env, clear=False), patch(
            "tt_model_runners.video_runner._create_dit_runner", return_value=mock_runner
        ), patch(
            "tt_model_runners.video_runner._attach_mpi_comm", return_value=mock_comm
        ), patch(
            "tt_model_runners.video_runner.cleanup_orphaned_video_files",
            return_value=0,
        ) as mock_cleanup:
            run_all_ranks()

        mock_cleanup.assert_not_called()

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

        def _raise_interrupt(comm, runner, ishm, encode_queue):
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

        # Under the create-or-attach ownership model, the runner only closes
        # its own fd on shutdown; unlinking is operator-driven.
        mock_input_shm.close.assert_called_once()
        mock_input_shm.unlink.assert_not_called()
        mock_output_shm.close.assert_called_once()
        mock_output_shm.unlink.assert_not_called()
        mock_runner.close_device.assert_called_once()

    def test_nonrank0_skips_shm(self):
        import tt_model_runners.video_runner as vr

        vr._shutdown = False
        mock_runner = MagicMock()
        mock_runner.warmup = AsyncMock()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.Get_size.return_value = 4
        # Shutdown payload is a (None, None, False) tuple — see _broadcast_request.
        mock_comm.bcast.return_value = (None, None, False)

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


_TINY_PNG_B64_CACHE: str = ""


def _tiny_png_b64() -> str:
    """Smallest valid PNG, base64-encoded — used as a stand-in for I2V
    conditioning images in tests that don't actually decode them."""
    global _TINY_PNG_B64_CACHE
    if _TINY_PNG_B64_CACHE:
        return _TINY_PNG_B64_CACHE

    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=0).save(buf, format="PNG")
    _TINY_PNG_B64_CACHE = base64.b64encode(buf.getvalue()).decode("ascii")
    return _TINY_PNG_B64_CACHE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
