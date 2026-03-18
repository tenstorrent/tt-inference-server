# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import struct
import threading
import time
import uuid

import pytest

import sys
from unittest.mock import Mock

sys.modules["ttnn"] = Mock()

mock_settings = Mock()
mock_settings.enable_telemetry = False
mock_settings_module = Mock()
mock_settings_module.settings = mock_settings
mock_settings_module.get_settings = Mock(return_value=mock_settings)
sys.modules["config.settings"] = mock_settings_module
sys.modules["telemetry.telemetry_client"] = Mock()

from ipc.video_shm import FrameResult, FrameStatus, VideoRequest, VideoShm


# ── Helpers ──


def _unique_name(prefix: str = "test_vshm") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _force_cleanup_shm(name: str) -> None:
    """Best-effort removal of a SHM region from /dev/shm/."""
    try:
        from multiprocessing import shared_memory

        s = shared_memory.SharedMemory(name=name, create=False)
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


def _make_frame(
    task_id: str = "abcdef12-3456-7890-abcd-ef1234567890",
    status: FrameStatus = FrameStatus.FRAME,
    frame_index: int = 0,
    total_frames: int = 81,
    height: int = 480,
    width: int = 832,
    channels: int = 3,
    data_byte: int = 0xAB,
    data_len: int = 128,
) -> FrameResult:
    return FrameResult(
        task_id=task_id,
        status=status,
        frame_index=frame_index,
        total_frames=total_frames,
        height=height,
        width=width,
        channels=channels,
        frame_data=bytes([data_byte] * data_len),
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

    def test_open_create_replaces_existing(self):
        """open(create=True) unlinks stale SHM and recreates without fd leak."""
        name = _unique_name("recreate")
        shm1 = VideoShm(name, mode="input")
        shm1.open(create=True)

        shm2 = VideoShm(name, mode="input")
        shm2.open(create=True)
        assert shm2._shm is not None
        assert shm2._buf is not None

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


# ── Frame roundtrip ──


class TestFrameRoundtrip:
    def test_basic_frame(self, output_pair):
        writer, reader = output_pair
        frame = _make_frame()
        writer.write_frame(frame)
        got = reader.read_frame()

        assert got is not None
        assert got.task_id == frame.task_id
        assert got.status == FrameStatus.FRAME
        assert got.frame_index == 0
        assert got.total_frames == 81
        assert got.height == 480
        assert got.width == 832
        assert got.channels == 3
        assert got.frame_data == frame.frame_data

    def test_done_sentinel(self, output_pair):
        writer, reader = output_pair
        done = _make_frame(status=FrameStatus.DONE, data_len=0)
        writer.write_frame(done)
        got = reader.read_frame()
        assert got.status == FrameStatus.DONE
        assert got.frame_data == b""

    def test_error_sentinel(self, output_pair):
        writer, reader = output_pair
        err = _make_frame(status=FrameStatus.ERROR, data_len=0)
        writer.write_frame(err)
        got = reader.read_frame()
        assert got.status == FrameStatus.ERROR

    def test_oversized_frame_raises(self, output_pair):
        writer, _ = output_pair
        oversized = bytes(VideoShm.MAX_FRAME_SIZE + 1)
        frame = FrameResult(
            task_id="tid",
            status=FrameStatus.FRAME,
            frame_index=0,
            total_frames=1,
            height=720,
            width=1280,
            channels=3,
            frame_data=oversized,
        )
        with pytest.raises(ValueError, match="exceeds MAX_FRAME_SIZE"):
            writer.write_frame(frame)

    def test_large_frame_data(self, output_pair):
        writer, reader = output_pair
        large_data = bytes(range(256)) * 1024
        frame = FrameResult(
            task_id="abcdef12-3456-7890-abcd-ef1234567890",
            status=FrameStatus.FRAME,
            frame_index=0,
            total_frames=1,
            height=480,
            width=832,
            channels=3,
            frame_data=large_data,
        )
        writer.write_frame(frame)
        got = reader.read_frame()
        assert got.frame_data == large_data


# ── Slot states ──


class TestSlotStateTransitions:
    def test_slot_starts_free(self, input_shm):
        state = struct.unpack_from("<i", input_shm._buf, 0)[0]
        assert state == VideoShm._FREE

    def test_write_sets_taken(self, input_pair):
        writer, _ = input_pair
        writer.write_request(_make_request())
        state = struct.unpack_from("<i", writer._buf, 0)[0]
        assert state == VideoShm._TAKEN

    def test_read_sets_free(self, input_pair):
        writer, reader = input_pair
        writer.write_request(_make_request())
        reader.read_request()
        state = struct.unpack_from("<i", reader._buf, 0)[0]
        assert state == VideoShm._FREE


# ── Ring buffer wrapping ──


class TestRingBufferWrapping:
    def test_position_wraps_after_slots(self, input_pair):
        writer, reader = input_pair
        req = _make_request()
        for _ in range(VideoShm.SLOTS + 2):
            writer.write_request(req)
            got = reader.read_request()
            assert got is not None
            assert got.task_id == req.task_id
        assert writer._pos == 2
        assert reader._pos == 2

    def test_multiple_frames_wrap(self, output_pair):
        writer, reader = output_pair
        for i in range(VideoShm.SLOTS + 3):
            frame = _make_frame(frame_index=i, data_len=64)
            writer.write_frame(frame)
            got = reader.read_frame()
            assert got.frame_index == i
        assert writer._pos == 3
        assert reader._pos == 3


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

    def test_read_frame_returns_none_on_shutdown(self):
        name = _unique_name("sd_frm")
        shutdown = False

        shm = VideoShm(name, mode="output", is_shutdown=lambda: shutdown)
        shm.open()

        def trigger_shutdown():
            nonlocal shutdown
            time.sleep(0.05)
            shutdown = True

        t = threading.Thread(target=trigger_shutdown)
        t.start()
        result = shm.read_frame()
        t.join()
        assert result is None

        shm.close()
        _force_cleanup_shm(name)

    def test_write_request_returns_on_shutdown(self):
        name = _unique_name("sd_wr")
        shutdown = False

        shm = VideoShm(name, mode="input", is_shutdown=lambda: shutdown)
        shm.open()

        for i in range(VideoShm.SLOTS):
            off = i * shm._slot_size
            struct.pack_into("<i", shm._buf, off, VideoShm._TAKEN)

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
    def test_read_frame_timeout(self):
        """read_frame(timeout_s) returns None after the deadline."""
        name = _unique_name("tmout_f")
        shm = VideoShm(name, mode="output")
        shm.open()

        start = time.monotonic()
        result = shm.read_frame(timeout_s=0.15)
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

    def test_read_frame_returns_before_timeout_when_data_available(self, output_pair):
        """Timeout should not fire when a frame is available promptly."""
        writer, reader = output_pair
        frame = _make_frame(data_len=64)

        def delayed_write():
            time.sleep(0.05)
            writer.write_frame(frame)

        t = threading.Thread(target=delayed_write)
        t.start()

        result = reader.read_frame(timeout_s=5.0)
        t.join()
        assert result is not None
        assert result.frame_data == frame.frame_data


# ── Error frame protocol (end-to-end) ──


class TestErrorFrameProtocol:
    def test_runner_error_frame_received_by_server(self):
        """Simulate runner writing ERROR frame after a pipeline exception."""
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

            # Simulate pipeline that raises after 0 frames
            runner_out.write_frame(
                FrameResult(
                    task_id=req.task_id,
                    status=FrameStatus.ERROR,
                    frame_index=0,
                    total_frames=0,
                    height=0,
                    width=0,
                    channels=0,
                    frame_data=b"",
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
        result = server_out.read_frame(timeout_s=2.0)

        assert result is not None
        assert result.status == FrameStatus.ERROR

        server_in.close()
        server_out.close()
        in_shm_creator.close()
        out_shm_creator.close()
        _force_cleanup_shm(in_name)
        _force_cleanup_shm(out_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
