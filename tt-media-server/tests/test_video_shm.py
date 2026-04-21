# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os
import pickle
import struct
import sys
import threading
import time
import uuid
from unittest.mock import Mock, patch

import pytest

sys.modules["ttnn"] = Mock()

mock_settings = Mock()
mock_settings.enable_telemetry = False
mock_settings.model_runner = "sp_runner"
mock_settings.use_dynamic_batcher = False
mock_settings.is_galaxy = False
mock_settings.device_mesh_shape = (1, 1)
# Match test_video_runner: unset Mock is truthy and breaks runner_utils env setup.
mock_settings.default_throttle_level = ""
mock_settings_module = Mock()
mock_settings_module.settings = mock_settings
mock_settings_module.get_settings = Mock(return_value=mock_settings)
sys.modules["config.settings"] = mock_settings_module
sys.modules["telemetry.telemetry_client"] = Mock()

from ipc.video_shm import (
    VideoRequest,
    VideoResponse,
    VideoShm,
    VideoStatus,
    cleanup_orphaned_video_files,
    video_result_path,
)

# ── Helpers ──


def _unique_name(prefix: str = "test_vshm") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _force_cleanup_shm(name: str) -> None:
    """Best-effort removal of a SHM region (and its ``<name>_state`` sibling)
    from ``/dev/shm/``. Every ``VideoShm`` now owns two segments under the
    create-or-attach model, so both must be unlinked between tests."""
    from multiprocessing import shared_memory

    for target in (name, f"{name}{VideoShm._STATE_SUFFIX}"):
        try:
            s = shared_memory.SharedMemory(name=target, create=False)
            s.close()
            s.unlink()
        except FileNotFoundError:
            pass


def _make_request(**overrides) -> VideoRequest:
    defaults = dict(
        task_id="abcdef12-3456-7890-abcd-ef1234567890",
        prompt="A beautiful sunset over the ocean",
        negative_prompt="blurry, low quality",
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


def _make_response(
    task_id: str = "abcdef12-3456-7890-abcd-ef1234567890",
    status: VideoStatus = VideoStatus.SUCCESS,
    file_path: str = "/dev/shm/tt_video_test.pkl",
    error_message: str = "",
) -> VideoResponse:
    return VideoResponse(
        task_id=task_id,
        status=status,
        file_path=file_path,
        error_message=error_message,
    )


# ── Fixtures ──


@pytest.fixture
def input_shm():
    name = _unique_name("in")
    shm = VideoShm(name, mode="input")
    shm.open()
    yield shm
    shm.close()
    _force_cleanup_shm(name)


@pytest.fixture
def output_shm():
    name = _unique_name("out")
    shm = VideoShm(name, mode="output")
    shm.open()
    yield shm
    shm.close()
    _force_cleanup_shm(name)


@pytest.fixture
def input_pair(input_shm):
    """Writer + reader attached to the same input SHM region."""
    writer = VideoShm(input_shm.name, mode="input")
    writer.open(create=False)
    reader = VideoShm(input_shm.name, mode="input")
    reader.open(create=False)
    yield writer, reader
    writer.close()
    reader.close()


@pytest.fixture
def output_pair(output_shm):
    """Writer + reader attached to the same output SHM region."""
    writer = VideoShm(output_shm.name, mode="output")
    writer.open(create=False)
    reader = VideoShm(output_shm.name, mode="output")
    reader.open(create=False)
    yield writer, reader
    writer.close()
    reader.close()


# ── Init ──


class TestVideoShmInit:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            VideoShm("test", mode="bad")

    def test_input_mode_slot_size(self):
        shm = VideoShm("test", mode="input")
        assert shm._slot_size == VideoShm.INPUT_SLOT_SIZE

    def test_output_mode_slot_size(self):
        shm = VideoShm("test", mode="output")
        assert shm._slot_size == VideoShm.OUTPUT_SLOT_SIZE

    def test_name_strips_leading_slash(self):
        shm = VideoShm("/myname", mode="input")
        assert shm.name == "myname"


# ── Lifecycle ──


class TestVideoShmLifecycle:
    def test_context_manager(self):
        name = _unique_name("ctx")
        with VideoShm(name, mode="input") as shm:
            assert shm._shm is not None
            assert shm._buf is not None
        assert shm._shm is None
        assert shm._buf is None
        _force_cleanup_shm(name)

    def test_open_reattach_on_existing(self):
        name = _unique_name("reattach")
        shm1 = VideoShm(name, mode="input")
        shm1.open()
        shm2 = VideoShm(name, mode="input")
        shm2.open(create=False)
        assert shm2._shm is not None
        shm1.close()
        shm2.close()
        _force_cleanup_shm(name)

    def test_close_idempotent(self):
        name = _unique_name("close")
        shm = VideoShm(name, mode="input")
        shm.open()
        shm.close()
        shm.close()
        _force_cleanup_shm(name)

    def test_open_create_false_attaches(self):
        name = _unique_name("attach")
        creator = VideoShm(name, mode="input")
        creator.open(create=True)

        attacher = VideoShm(name, mode="input")
        attacher.open(create=False)
        assert attacher._buf is not None

        # Writes through one are visible from the other
        struct.pack_into("<i", creator._buf, 0, 0xDEAD)
        assert struct.unpack_from("<i", attacher._buf, 0)[0] == 0xDEAD

        attacher.close()
        creator.close()
        _force_cleanup_shm(name)

    def test_unlink_removes_shm(self):
        name = _unique_name("unlink")
        shm = VideoShm(name, mode="input")
        shm.open()
        shm.unlink()
        shm.close()

        from multiprocessing import shared_memory

        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)

    def test_open_is_idempotent_when_segment_exists(self):
        """Under the create-or-attach model, a second open() on an existing
        segment must succeed (attach), not raise. The `create` kwarg is
        accepted for backward-compat but ignored — behaviour is always
        create-if-missing-else-attach."""
        name = _unique_name("reopen")
        shm1 = VideoShm(name, mode="input")
        shm1.open()

        shm2 = VideoShm(name, mode="input")
        shm2.open()  # attaches; must not unlink or recreate
        assert shm2._shm is not None
        assert shm2._buf is not None

        # Cross-handle visibility: a write through shm1 is seen by shm2.
        struct.pack_into("<i", shm1._buf, 0, 0xBEEF)
        assert struct.unpack_from("<i", shm2._buf, 0)[0] == 0xBEEF

        shm1.close()
        shm2.close()
        _force_cleanup_shm(name)

    def test_unlink_idempotent(self):
        """Double unlink must not raise."""
        name = _unique_name("dbl_unlink")
        shm = VideoShm(name, mode="input")
        shm.open()
        shm.unlink()
        shm.unlink()
        shm.close()


# ── Request roundtrip ──


class TestRequestRoundtrip:
    def test_basic_roundtrip(self, input_pair):
        writer, reader = input_pair
        req = _make_request()
        writer.write_request(req)
        got = reader.read_request()

        assert got is not None
        assert got.task_id == req.task_id
        assert got.prompt == req.prompt
        assert got.negative_prompt == req.negative_prompt
        assert got.num_inference_steps == req.num_inference_steps
        assert got.seed == req.seed
        assert got.height == req.height
        assert got.width == req.width
        assert got.num_frames == req.num_frames
        assert abs(got.guidance_scale - req.guidance_scale) < 1e-5
        assert abs(got.guidance_scale_2 - req.guidance_scale_2) < 1e-5

    def test_empty_strings(self, input_pair):
        writer, reader = input_pair
        req = _make_request(prompt="", negative_prompt="")
        writer.write_request(req)
        got = reader.read_request()
        assert got.prompt == ""
        assert got.negative_prompt == ""

    def test_long_prompt_truncated(self, input_pair):
        writer, reader = input_pair
        req = _make_request(prompt="x" * 5000)
        writer.write_request(req)
        got = reader.read_request()
        assert len(got.prompt) == VideoShm.MAX_PROMPT_LEN

    def test_unicode_prompt(self, input_pair):
        writer, reader = input_pair
        req = _make_request(prompt="日本語テスト 🎬")
        writer.write_request(req)
        got = reader.read_request()
        assert got.prompt == req.prompt

    def test_utf8_multibyte_truncation_boundary(self, input_pair):
        """Truncation at max_len must not split multi-byte UTF-8 characters."""
        writer, reader = input_pair
        # "日" is 3 bytes in UTF-8. 683 * 3 = 2049, one byte over MAX_PROMPT_LEN (2048).
        # Naive slice at 2048 would cut the 683rd character mid-byte.
        char_3byte = "日"
        prompt = char_3byte * 683
        req = _make_request(prompt=prompt)
        writer.write_request(req)
        got = reader.read_request()

        raw_bytes = got.prompt.encode("utf-8")
        assert len(raw_bytes) <= VideoShm.MAX_PROMPT_LEN
        assert len(got.prompt) == 682

    def test_utf8_4byte_truncation_boundary(self, input_pair):
        """4-byte emoji characters must not be split at the boundary."""
        writer, reader = input_pair
        emoji = "\U0001f3ac"  # 🎬 = 4 bytes
        # 512 * 4 = 2048 exactly; 513 * 4 = 2052 > 2048
        prompt = emoji * 513
        req = _make_request(prompt=prompt)
        writer.write_request(req)
        got = reader.read_request()

        raw_bytes = got.prompt.encode("utf-8")
        assert len(raw_bytes) <= VideoShm.MAX_PROMPT_LEN
        assert len(got.prompt) == 512

    def test_long_negative_prompt_truncated(self, input_pair):
        writer, reader = input_pair
        req = _make_request(negative_prompt="x" * 2000)
        writer.write_request(req)
        got = reader.read_request()
        assert len(got.negative_prompt) == VideoShm.MAX_NEG_PROMPT_LEN

    def test_multiple_requests_sequential(self, input_pair):
        writer, reader = input_pair
        for i in range(5):
            req = _make_request(
                task_id=f"task-{i:030d}",
                prompt=f"prompt {i}",
                seed=i * 10,
            )
            writer.write_request(req)
            got = reader.read_request()
            assert got.task_id == req.task_id
            assert got.prompt == req.prompt
            assert got.seed == i * 10

    def test_guidance_scale_precision(self, input_pair):
        writer, reader = input_pair
        req = _make_request(guidance_scale=7.5, guidance_scale_2=1.25)
        writer.write_request(req)
        got = reader.read_request()
        assert abs(got.guidance_scale - 7.5) < 1e-5
        assert abs(got.guidance_scale_2 - 1.25) < 1e-5

    def test_zero_seed_and_steps(self, input_pair):
        writer, reader = input_pair
        req = _make_request(seed=0, num_inference_steps=0)
        writer.write_request(req)
        got = reader.read_request()
        assert got.seed == 0
        assert got.num_inference_steps == 0


# ── Response roundtrip ──


class TestResponseRoundtrip:
    def test_basic_response(self, output_pair):
        writer, reader = output_pair
        resp = _make_response()
        writer.write_response(resp)
        got = reader.read_response()

        assert got is not None
        assert got.task_id == resp.task_id
        assert got.status == VideoStatus.SUCCESS
        assert got.file_path == resp.file_path
        assert got.error_message == ""

    def test_error_response(self, output_pair):
        writer, reader = output_pair
        resp = _make_response(
            status=VideoStatus.ERROR,
            file_path="",
            error_message="pipeline exploded",
        )
        writer.write_response(resp)
        got = reader.read_response()
        assert got.status == VideoStatus.ERROR
        assert got.error_message == "pipeline exploded"
        assert got.file_path == ""

    def test_long_file_path_truncated(self, output_pair):
        from ipc.video_shm import MAX_FILE_PATH_LEN

        writer, reader = output_pair
        long_path = "/dev/shm/" + "x" * 300
        resp = _make_response(file_path=long_path)
        writer.write_response(resp)
        got = reader.read_response()
        assert len(got.file_path) == MAX_FILE_PATH_LEN

    def test_multiple_responses_sequential(self, output_pair):
        writer, reader = output_pair
        for i in range(4):
            resp = _make_response(file_path=f"/dev/shm/tt_video_task_{i}.pkl")
            writer.write_response(resp)
            got = reader.read_response()
            assert got.file_path == f"/dev/shm/tt_video_task_{i}.pkl"

    def test_empty_file_path_on_error(self, output_pair):
        writer, reader = output_pair
        resp = _make_response(
            status=VideoStatus.ERROR,
            file_path="",
            error_message="err",
        )
        writer.write_response(resp)
        got = reader.read_response()
        assert got.file_path == ""
        assert got.error_message == "err"

    def test_response_preserves_all_fields(self, output_pair):
        writer, reader = output_pair
        resp = VideoResponse(
            task_id="abcdef12-3456-7890-abcd-ef1234567890",
            status=VideoStatus.SUCCESS,
            file_path="/dev/shm/tt_video_my_task.pkl",
            error_message="",
        )
        writer.write_response(resp)
        got = reader.read_response()
        assert got.task_id == resp.task_id
        assert got.status == VideoStatus.SUCCESS
        assert got.file_path == "/dev/shm/tt_video_my_task.pkl"
        assert got.error_message == ""


# ── Slot states ──


class TestSlotStateTransitions:
    def test_slot_starts_empty(self, input_shm):
        state = struct.unpack_from("<i", input_shm._buf, 0)[0]
        assert state == VideoShm._EMPTY

    def test_write_sets_filled(self, input_pair):
        writer, _ = input_pair
        writer.write_request(_make_request())
        state = struct.unpack_from("<i", writer._buf, 0)[0]
        assert state == VideoShm._FILLED

    def test_read_sets_empty(self, input_pair):
        writer, reader = input_pair
        writer.write_request(_make_request())
        reader.read_request()
        state = struct.unpack_from("<i", reader._buf, 0)[0]
        assert state == VideoShm._EMPTY


# ── Ring buffer wrapping ──


class TestRingBufferWrapping:
    def test_position_wraps_after_slots(self, input_pair):
        writer, reader = input_pair
        req = _make_request()
        total = VideoShm.INPUT_SLOTS + 2
        for _ in range(total):
            writer.write_request(req)
            got = reader.read_request()
            assert got is not None
            assert got.task_id == req.task_id
        # Indices live in the <name>_state SHM segment and are monotonic u64
        # counters (not mod-slots) — the slot index is derived as (idx % _slots).
        assert writer._get_writer_index() == total
        assert reader._get_reader_index() == total
        assert writer._get_writer_index() % VideoShm.INPUT_SLOTS == 2
        assert reader._get_reader_index() % VideoShm.INPUT_SLOTS == 2

    def test_responses_wrap(self, output_pair):
        writer, reader = output_pair
        total = VideoShm.OUTPUT_SLOTS + 1
        for i in range(total):
            resp = _make_response(file_path=f"/dev/shm/tt_video_wrap_{i}.pkl")
            writer.write_response(resp)
            got = reader.read_response()
            assert got.file_path == f"/dev/shm/tt_video_wrap_{i}.pkl"
        assert writer._get_writer_index() == total
        assert reader._get_reader_index() == total
        assert writer._get_writer_index() % VideoShm.OUTPUT_SLOTS == 1
        assert reader._get_reader_index() % VideoShm.OUTPUT_SLOTS == 1


# ── Shutdown signaling ──


class TestShutdownSignaling:
    def test_read_request_returns_none_on_shutdown(self):
        name = _unique_name("sd_req")
        shutdown = False

        shm = VideoShm(name, mode="input", is_shutdown=lambda: shutdown)
        shm.open()

        def trigger_shutdown():
            nonlocal shutdown
            time.sleep(0.05)
            shutdown = True

        t = threading.Thread(target=trigger_shutdown)
        t.start()
        result = shm.read_request()
        t.join()
        assert result is None

        shm.close()
        _force_cleanup_shm(name)

    def test_read_response_returns_none_on_shutdown(self):
        name = _unique_name("sd_resp")
        shutdown = False

        shm = VideoShm(name, mode="output", is_shutdown=lambda: shutdown)
        shm.open()

        def trigger_shutdown():
            nonlocal shutdown
            time.sleep(0.05)
            shutdown = True

        t = threading.Thread(target=trigger_shutdown)
        t.start()
        result = shm.read_response()
        t.join()
        assert result is None

        shm.close()
        _force_cleanup_shm(name)

    def test_write_request_returns_on_shutdown(self):
        name = _unique_name("sd_wr")
        shutdown = False

        shm = VideoShm(name, mode="input", is_shutdown=lambda: shutdown)
        shm.open()

        for i in range(VideoShm.INPUT_SLOTS):
            off = i * shm._slot_size
            struct.pack_into("<i", shm._buf, off, VideoShm._FILLED)

        def trigger_shutdown():
            nonlocal shutdown
            time.sleep(0.05)
            shutdown = True

        t = threading.Thread(target=trigger_shutdown)
        t.start()
        shm.write_request(_make_request())
        t.join()

        shm.close()
        _force_cleanup_shm(name)

    def test_write_response_returns_on_shutdown(self):
        name = _unique_name("sd_wr_resp")
        shutdown = False

        shm = VideoShm(name, mode="output", is_shutdown=lambda: shutdown)
        shm.open()

        for i in range(VideoShm.OUTPUT_SLOTS):
            off = i * shm._slot_size
            struct.pack_into("<i", shm._buf, off, VideoShm._FILLED)

        def trigger_shutdown():
            nonlocal shutdown
            time.sleep(0.05)
            shutdown = True

        t = threading.Thread(target=trigger_shutdown)
        t.start()
        shm.write_response(_make_response())
        t.join()

        shm.close()
        _force_cleanup_shm(name)


# ── Cross-thread ──


class TestCrossThreadRoundtrip:
    def test_threaded_request_roundtrip(self):
        name = _unique_name("thr")
        shutdown = False

        writer_shm = VideoShm(name, mode="input", is_shutdown=lambda: shutdown)
        writer_shm.open()

        reader_shm = VideoShm(name, mode="input", is_shutdown=lambda: shutdown)
        reader_shm.open(create=False)

        req = _make_request(task_id="thread-test-task-id-1234567890ab")
        received = []

        def reader_fn():
            got = reader_shm.read_request()
            if got:
                received.append(got)

        t = threading.Thread(target=reader_fn)
        t.start()

        time.sleep(0.02)
        writer_shm.write_request(req)
        t.join(timeout=2.0)

        assert len(received) == 1
        assert received[0].task_id == req.task_id
        assert received[0].prompt == req.prompt

        shutdown = True
        reader_shm.close()
        writer_shm.close()
        _force_cleanup_shm(name)


# ── Timeout ──


class TestTimeout:
    def test_read_response_timeout(self):
        """read_response(timeout_s) returns None after the deadline."""
        name = _unique_name("tmout_resp")
        shm = VideoShm(name, mode="output")
        shm.open()

        start = time.monotonic()
        result = shm.read_response(timeout_s=0.15)
        elapsed = time.monotonic() - start

        assert result is None
        assert elapsed >= 0.14
        assert elapsed < 1.0

        shm.close()
        _force_cleanup_shm(name)

    def test_read_request_timeout(self):
        """read_request(timeout_s) returns None after the deadline."""
        name = _unique_name("tmout_r")
        shm = VideoShm(name, mode="input")
        shm.open()

        start = time.monotonic()
        result = shm.read_request(timeout_s=0.15)
        elapsed = time.monotonic() - start

        assert result is None
        assert elapsed >= 0.14
        assert elapsed < 1.0

        shm.close()
        _force_cleanup_shm(name)

    def test_read_response_returns_before_timeout_when_data_available(
        self, output_pair
    ):
        """Timeout should not fire when data is available promptly."""
        writer, reader = output_pair
        resp = _make_response()

        def delayed_write():
            time.sleep(0.05)
            writer.write_response(resp)

        t = threading.Thread(target=delayed_write)
        t.start()

        result = reader.read_response(timeout_s=5.0)
        t.join()
        assert result is not None
        assert result.file_path == resp.file_path


# ── Error response protocol (end-to-end) ──


class TestErrorResponseProtocol:
    def test_runner_error_response_received_by_server(self):
        """Simulate runner writing an ERROR response after a pipeline exception."""
        in_name = _unique_name("eproto_in")
        out_name = _unique_name("eproto_out")

        in_shm_creator = VideoShm(in_name, mode="input")
        in_shm_creator.open()
        out_shm_creator = VideoShm(out_name, mode="output")
        out_shm_creator.open()

        server_in = VideoShm(in_name, mode="input")
        server_in.open(create=False)
        server_in.write_request(_make_request(num_frames=3, height=2, width=3))

        runner_done = threading.Event()

        def runner_side():
            runner_in = VideoShm(in_name, mode="input")
            runner_in.open(create=False)
            runner_out = VideoShm(out_name, mode="output")
            runner_out.open(create=False)

            req = runner_in.read_request(timeout_s=2.0)
            assert req is not None

            runner_out.write_response(
                VideoResponse(
                    task_id=req.task_id,
                    status=VideoStatus.ERROR,
                    file_path="",
                    error_message="pipeline crashed",
                )
            )

            runner_in.close()
            runner_out.close()
            runner_done.set()

        t = threading.Thread(target=runner_side)
        t.start()
        runner_done.wait(timeout=5.0)
        t.join(timeout=1.0)

        server_out = VideoShm(out_name, mode="output")
        server_out.open(create=False)
        result = server_out.read_response(timeout_s=2.0)

        assert result is not None
        assert result.status == VideoStatus.ERROR
        assert result.error_message == "pipeline crashed"

        server_in.close()
        server_out.close()
        in_shm_creator.close()
        out_shm_creator.close()
        _force_cleanup_shm(in_name)
        _force_cleanup_shm(out_name)


# ── File-based IPC helpers ──


class TestVideoResultPath:
    def test_returns_expected_path(self):
        task_id = "abc-123"
        path = video_result_path(task_id)
        from ipc.video_shm import VIDEO_FILE_DIR

        assert path == os.path.join(VIDEO_FILE_DIR, f"tt_video_{task_id}.pkl")

    def test_unique_per_task_id(self):
        assert video_result_path("a") != video_result_path("b")


class TestCleanupOrphanedVideoFiles:
    """Verify :func:`cleanup_orphaned_video_files` removes matching files."""

    @pytest.fixture(autouse=True)
    def _use_tmpdir(self, tmp_path, monkeypatch):
        """Redirect VIDEO_FILE_DIR to a temp directory for isolation."""
        monkeypatch.setattr("ipc.video_shm.VIDEO_FILE_DIR", str(tmp_path))
        self._tmp = tmp_path

    def test_removes_matching_files(self):
        for i in range(3):
            (self._tmp / f"tt_video_task{i}.pkl").write_bytes(b"x")
        removed = cleanup_orphaned_video_files()
        assert removed == 3
        assert list(self._tmp.glob("tt_video_*.pkl")) == []

    def test_ignores_non_matching_files(self):
        (self._tmp / "other_file.txt").write_bytes(b"keep")
        (self._tmp / "tt_video_only.pkl").write_bytes(b"x")
        removed = cleanup_orphaned_video_files()
        assert removed == 1
        assert (self._tmp / "other_file.txt").exists()

    def test_returns_zero_when_no_files(self):
        assert cleanup_orphaned_video_files() == 0

    def test_oserror_on_unlink_is_ignored(self, monkeypatch):
        """Per-file unlink failures are swallowed (lines 85–86 in video_shm)."""
        import ipc.video_shm as video_shm_mod

        (self._tmp / "tt_video_stale.pkl").write_bytes(b"x")

        def unlink_raises(_path):
            raise OSError("permission denied")

        # Patch the same os object cleanup_orphaned_video_files uses (covers except OSError).
        monkeypatch.setattr(video_shm_mod.os, "unlink", unlink_raises)
        assert cleanup_orphaned_video_files() == 0
        assert (self._tmp / "tt_video_stale.pkl").exists()

    def test_oserror_on_unlink_via_patch(self):
        """Same as test_oserror_on_unlink_is_ignored using patch (diff-cover reliability)."""
        import ipc.video_shm as video_shm_mod

        p = self._tmp / "tt_video_patch.pkl"
        p.write_bytes(b"x")
        with patch.object(video_shm_mod.os, "unlink", side_effect=OSError("busy")):
            assert cleanup_orphaned_video_files() == 0
        assert p.exists()


# ── End-to-end file-based IPC roundtrip ──


class TestFileBasedRoundtrip:
    """Simulate the full file-based IPC cycle: write video → SHM → read → load → cleanup."""

    @pytest.fixture(autouse=True)
    def _use_tmpdir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ipc.video_shm.VIDEO_FILE_DIR", str(tmp_path))
        self._tmp = tmp_path

    def test_write_read_cleanup_cycle(self, output_pair):
        writer, reader = output_pair
        task_id = "e2e-roundtrip-task-id-0123456789ab"
        payload = {"frames": [1, 2, 3], "meta": "test"}

        file_path = video_result_path(task_id)
        with open(file_path, "wb") as fh:
            pickle.dump(payload, fh)
        assert os.path.exists(file_path)

        writer.write_response(
            VideoResponse(
                task_id=task_id,
                status=VideoStatus.SUCCESS,
                file_path=file_path,
                error_message="",
            )
        )

        resp = reader.read_response()
        assert resp is not None
        assert resp.file_path == file_path

        with open(resp.file_path, "rb") as fh:
            loaded = pickle.load(fh)
        os.unlink(resp.file_path)

        assert loaded == payload
        assert not os.path.exists(file_path)

    def test_error_response_no_file_created(self, output_pair):
        writer, reader = output_pair
        task_id = "e2e-error-task-id-0123456789abcd"

        writer.write_response(
            VideoResponse(
                task_id=task_id,
                status=VideoStatus.ERROR,
                file_path="",
                error_message="OOM during inference",
            )
        )

        resp = reader.read_response()
        assert resp.status == VideoStatus.ERROR
        assert resp.file_path == ""
        assert resp.error_message == "OOM during inference"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
