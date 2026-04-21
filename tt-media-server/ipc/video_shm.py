#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Video shared memory IPC for cross-process video generation.

Wraps a slot-based ring buffer in POSIX shared memory for video request/response
exchange between the server process and an external runner process.

Two modes:
  - "input"  (server -> runner): carries VideoRequest payloads
  - "output" (runner -> server): carries VideoResponse payloads (file-path reference)

Ownership model
---------------
Both sides of the ring (server and runner) call ``open()`` which performs a
POSIX-style *create-or-attach*: the first caller creates, any later caller
attaches to the existing segment. Neither side unlinks on process exit, so
either side can restart and resume at the correct ring position. A separate
``<name>_state`` SHM segment carries monotonic ``writer_index`` / ``reader_index``
counters (u64 each) so that on restart a process picks up exactly where it
left off.

Use ``python -m ipc.video_shm_bootstrap {up,down,status}`` for operator-driven
creation/teardown/inspection of the segments.

The video payload itself is written to a file on ``/dev/shm/`` (RAM-backed tmpfs)
and only the file path (~100 bytes) is transmitted through the SHM output slot.
This avoids multiple large memcpy operations for 100-200 MB video blobs.

Example usage (either side)::

    input_shm  = VideoShm("video_in",  mode="input")
    output_shm = VideoShm("video_out", mode="output")
    input_shm.open()   # create-or-attach; symmetric on both sides
    output_shm.open()

    req = input_shm.read_request()
    path = video_result_path(req.task_id)
    with open(path, "wb") as f:
        f.write(pickle.dumps(video))
    output_shm.write_response(VideoResponse(task_id=..., file_path=path, ...))
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from multiprocessing import resource_tracker
from multiprocessing import shared_memory as _shm
from typing import Callable


class VideoStatus(IntEnum):
    SUCCESS = 0
    ERROR = 1


@dataclass(frozen=True)
class VideoRequest:
    task_id: str
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    seed: int
    height: int
    width: int
    num_frames: int
    guidance_scale: float
    guidance_scale_2: float


VIDEO_FILE_DIR = os.environ.get("TT_VIDEO_FILE_DIR", "/dev/shm")
VIDEO_FILE_GLOB = "tt_video_*.pkl"
MAX_FILE_PATH_LEN = 256


def video_result_path(task_id: str) -> str:
    """Return the file path for a video result on the RAM-backed tmpfs."""
    return os.path.join(VIDEO_FILE_DIR, f"tt_video_{task_id}.pkl")


def cleanup_orphaned_video_files() -> int:
    """Remove any leftover ``tt_video_*.pkl`` files from :data:`VIDEO_FILE_DIR`.

    Returns the number of files removed.  Safe to call at any point –
    individual unlink failures are silently ignored.
    """
    import glob

    removed = 0
    for path in glob.glob(os.path.join(VIDEO_FILE_DIR, VIDEO_FILE_GLOB)):
        try:
            os.unlink(path)
            removed += 1
        except OSError:
            pass
    return removed


@dataclass(frozen=True)
class VideoResponse:
    task_id: str
    status: VideoStatus
    file_path: str
    error_message: str


class VideoShm:
    """Slot-based ring buffer over POSIX shared memory for video IPC.

    Two SHM segments per ring:
      - ``<name>``       : the slot array (ring buffer of requests/responses).
      - ``<name>_state`` : 16 bytes — ``writer_index`` (u64) + ``reader_index`` (u64).
        Storing these in SHM (rather than as instance attributes) lets a process
        on either side restart and resume at the correct ring position.

    Layout per slot depends on *mode*:

    Input slot (2 640 B):
        state(4) task_id(36) prompt_length(4) prompt(2048)
        neg_prompt_length(4) negative_prompt(512)
        num_inference_steps(4) seed(8) height(4) width(4)
        num_frames(4) guidance_scale(4) guidance_scale_2(4)

    Output slot (564 B):
        state(4) task_id(36) status(4)
        error_len(4) error_msg(256)
        file_path_len(4) file_path(256)

    Slot state transitions (writer flips EMPTY→FILLED, reader flips FILLED→EMPTY)
    remain the sole synchronization mechanism; the indices are cursors used to
    pick which slot each role acts on next.

    The actual video payload is written to a file (see :func:`video_result_path`)
    and only the path is transmitted through SHM.
    """

    INPUT_SLOTS = 8
    OUTPUT_SLOTS = 4
    TASK_ID_SIZE = 36
    MAX_PROMPT_LEN = 2048
    MAX_NEG_PROMPT_LEN = 512
    MAX_ERROR_LEN = 256
    # MAX_VIDEO_SIZE = 256 * 1024 * 1024  # 256 MB – fits 168 * 848 * 480 * 3
    MAX_VIDEO_SIZE = (
        1024 * 1024 * 1024
    )  # 1 GB – fits 720p float32 (1280 * 720 * 3 * 81 frames)

    _EMPTY = 0
    _FILLED = 1
    _POLL_INTERVAL_S = 0.0001

    # ── State segment layout (<name>_state) ──
    # Monotonic u64 counters; slot index = counter % _slots.
    # Writer role flips _EMPTY -> _FILLED on the slot, then bumps writer_index.
    # Reader role flips _FILLED -> _EMPTY on the slot, then bumps reader_index.
    _STATE_WRITER_IDX_OFF = 0
    _STATE_READER_IDX_OFF = 8
    STATE_SEGMENT_SIZE = 16
    _STATE_SUFFIX = "_state"

    # ── Input slot field offsets ──
    _IN_STATE = 0
    _IN_TASK_ID = 4
    _IN_PROMPT_LEN = 40
    _IN_PROMPT = 44
    _IN_NEG_PROMPT_LEN = 2092
    _IN_NEG_PROMPT = 2096
    _IN_NUM_INFERENCE_STEPS = 2608
    _IN_SEED = 2612
    _IN_HEIGHT = 2620
    _IN_WIDTH = 2624
    _IN_NUM_FRAMES = 2628
    _IN_GUIDANCE_SCALE = 2632
    _IN_GUIDANCE_SCALE_2 = 2636
    INPUT_SLOT_SIZE = 2640

    # ── Output slot field offsets ──
    #   state(4) task_id(36) status(4) error_len(4) error(256)
    #   file_path_len(4) file_path(256)  → total 564 bytes
    _OUT_STATE = 0
    _OUT_TASK_ID = 4
    _OUT_STATUS = 40
    _OUT_ERROR_LEN = 44
    _OUT_ERROR = 48
    _OUT_FILE_PATH_LEN = 48 + MAX_ERROR_LEN  # 304
    _OUT_FILE_PATH = 48 + MAX_ERROR_LEN + 4  # 308
    OUTPUT_SLOT_SIZE = _OUT_FILE_PATH + MAX_FILE_PATH_LEN  # 564

    def __init__(
        self,
        name: str,
        *,
        mode: str,
        is_shutdown: Callable[[], bool] = lambda: False,
    ):
        if mode not in ("input", "output"):
            raise ValueError(f"mode must be 'input' or 'output', got {mode!r}")
        self._name = name.lstrip("/")
        self._mode = mode
        self._slots = self.INPUT_SLOTS if mode == "input" else self.OUTPUT_SLOTS
        self._slot_size = (
            self.INPUT_SLOT_SIZE if mode == "input" else self.OUTPUT_SLOT_SIZE
        )
        self._total_size = self._slots * self._slot_size
        self._shm: _shm.SharedMemory | None = None
        self._buf: memoryview | None = None
        self._state_shm: _shm.SharedMemory | None = None
        self._state_buf: memoryview | None = None
        self._is_shutdown = is_shutdown

    @property
    def name(self) -> str:
        return self._name

    @property
    def state_name(self) -> str:
        return f"{self._name}{self._STATE_SUFFIX}"

    # ── Lifecycle ──

    def open(self, create: bool | None = None) -> None:
        """Create-or-attach both the ring segment and the index-state segment.

        The first caller creates; later callers attach. Neither caller unlinks
        on process exit, so segments (and ring positions) survive restarts.

        The ``create`` keyword is accepted for backward compatibility and is
        ignored — behaviour is always "create if missing, else attach".
        """
        del create  # kept for backward compatibility with older call sites
        self._shm, _ = self._create_or_attach(self._name, self._total_size)
        self._state_shm, _ = self._create_or_attach(
            self.state_name, self.STATE_SEGMENT_SIZE
        )
        self._buf = self._shm.buf
        self._state_buf = self._state_shm.buf

    @staticmethod
    def _create_or_attach(name: str, size: int) -> tuple[_shm.SharedMemory, bool]:
        """POSIX O_CREAT-style open: create if absent, else attach.

        Returns (segment, created). Always drops the resource_tracker entry so
        this process never unlinks the segment on exit.
        """
        try:
            shm = _shm.SharedMemory(name=name, create=True, size=size)
            created = True
        except FileExistsError:
            shm = _shm.SharedMemory(name=name, create=False)
            if shm.size < size:
                raise RuntimeError(
                    f"Existing SHM {name!r} size {shm.size} < expected {size}. "
                    "Layout changed — run `python -m ipc.video_shm_bootstrap down` "
                    "to reset, or unlink manually."
                ) from None
            created = False

        # Always chmod (idempotent) so whoever created it, either side can attach.
        try:
            os.chmod(f"/dev/shm/{name}", 0o666)
        except (PermissionError, FileNotFoundError):
            pass

        # Critical: nobody should unlink on process exit. Drop resource_tracker
        # for every caller (creator and attacher alike) — segments are owned
        # by the operator via the bootstrap CLI, not by any individual process.
        try:
            resource_tracker.unregister(f"/{name}", "shared_memory")
        except KeyError:
            pass

        return shm, created

    def close(self) -> None:
        """Detach both segments from this process. Does not unlink."""
        if self._shm:
            self._shm.close()
            self._shm = None
            self._buf = None
        if self._state_shm:
            self._state_shm.close()
            self._state_shm = None
            self._state_buf = None

    def unlink(self) -> None:
        """Remove the ring and state segments from /dev/shm/.

        Under the current ownership model this is an *operator-only* action
        (invoked via ``python -m ipc.video_shm_bootstrap down`` or equivalent).
        Regular workers must not call this during normal shutdown — other
        processes may still be attached, and unlinking would destroy their
        ring state.
        """
        for shm in (self._shm, self._state_shm):
            if shm is None:
                continue
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> VideoShm:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── Shared index state (writer_index / reader_index in SHM) ──

    def _get_writer_index(self) -> int:
        return struct.unpack_from("<Q", self._state_buf, self._STATE_WRITER_IDX_OFF)[0]

    def _set_writer_index(self, value: int) -> None:
        struct.pack_into("<Q", self._state_buf, self._STATE_WRITER_IDX_OFF, value)

    def _get_reader_index(self) -> int:
        return struct.unpack_from("<Q", self._state_buf, self._STATE_READER_IDX_OFF)[0]

    def _set_reader_index(self, value: int) -> None:
        struct.pack_into("<Q", self._state_buf, self._STATE_READER_IDX_OFF, value)

    def queue_depth(self) -> int:
        """Number of FILLED slots currently in the ring (writer - reader)."""
        return self._get_writer_index() - self._get_reader_index()

    # ── Input SHM: request read / write ──

    def write_request(self, request: VideoRequest) -> None:
        """Write a VideoRequest into the next free input slot (spin-waits)."""
        buf = self._buf
        widx = self._get_writer_index()
        off = (widx % self._slots) * self._slot_size
        state_off = off + self._IN_STATE

        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._EMPTY:
                break
            time.sleep(self._POLL_INTERVAL_S)
        else:
            return

        self._pack_task_id(buf, off + self._IN_TASK_ID, request.task_id)
        self._pack_string(
            buf,
            off + self._IN_PROMPT_LEN,
            off + self._IN_PROMPT,
            request.prompt,
            self.MAX_PROMPT_LEN,
        )
        self._pack_string(
            buf,
            off + self._IN_NEG_PROMPT_LEN,
            off + self._IN_NEG_PROMPT,
            request.negative_prompt,
            self.MAX_NEG_PROMPT_LEN,
        )

        struct.pack_into(
            "<I", buf, off + self._IN_NUM_INFERENCE_STEPS, request.num_inference_steps
        )
        struct.pack_into("<q", buf, off + self._IN_SEED, request.seed)
        struct.pack_into("<I", buf, off + self._IN_HEIGHT, request.height)
        struct.pack_into("<I", buf, off + self._IN_WIDTH, request.width)
        struct.pack_into("<I", buf, off + self._IN_NUM_FRAMES, request.num_frames)
        struct.pack_into(
            "<f", buf, off + self._IN_GUIDANCE_SCALE, request.guidance_scale
        )
        struct.pack_into(
            "<f", buf, off + self._IN_GUIDANCE_SCALE_2, request.guidance_scale_2
        )

        struct.pack_into("<i", buf, state_off, self._FILLED)
        self._set_writer_index(widx + 1)

    def read_request(self, timeout_s: float | None = None) -> VideoRequest | None:
        """Blocking read of a VideoRequest from the next input slot."""
        buf = self._buf
        ridx = self._get_reader_index()
        off = (ridx % self._slots) * self._slot_size
        state_off = off + self._IN_STATE
        deadline = (time.monotonic() + timeout_s) if timeout_s is not None else None

        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._FILLED:
                break
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(self._POLL_INTERVAL_S)
        else:
            return None

        task_id = self._unpack_task_id(buf, off + self._IN_TASK_ID)
        prompt = self._unpack_string(
            buf, off + self._IN_PROMPT_LEN, off + self._IN_PROMPT
        )
        negative_prompt = self._unpack_string(
            buf,
            off + self._IN_NEG_PROMPT_LEN,
            off + self._IN_NEG_PROMPT,
        )

        num_inference_steps = struct.unpack_from(
            "<I", buf, off + self._IN_NUM_INFERENCE_STEPS
        )[0]
        seed = struct.unpack_from("<q", buf, off + self._IN_SEED)[0]
        height = struct.unpack_from("<I", buf, off + self._IN_HEIGHT)[0]
        width = struct.unpack_from("<I", buf, off + self._IN_WIDTH)[0]
        num_frames = struct.unpack_from("<I", buf, off + self._IN_NUM_FRAMES)[0]
        guidance_scale = struct.unpack_from("<f", buf, off + self._IN_GUIDANCE_SCALE)[0]
        guidance_scale_2 = struct.unpack_from(
            "<f", buf, off + self._IN_GUIDANCE_SCALE_2
        )[0]

        struct.pack_into("<i", buf, state_off, self._EMPTY)
        self._set_reader_index(ridx + 1)

        return VideoRequest(
            task_id=task_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
        )

    # ── Output SHM: response read / write (file-path reference) ──

    def write_response(self, response: VideoResponse) -> None:
        """Write a VideoResponse into the next free output slot."""
        buf = self._buf
        widx = self._get_writer_index()
        off = (widx % self._slots) * self._slot_size
        state_off = off + self._OUT_STATE

        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._EMPTY:
                break
            time.sleep(self._POLL_INTERVAL_S)
        else:
            return

        self._pack_task_id(buf, off + self._OUT_TASK_ID, response.task_id)
        struct.pack_into("<I", buf, off + self._OUT_STATUS, int(response.status))

        self._pack_string(
            buf,
            off + self._OUT_ERROR_LEN,
            off + self._OUT_ERROR,
            response.error_message,
            self.MAX_ERROR_LEN,
        )
        self._pack_string(
            buf,
            off + self._OUT_FILE_PATH_LEN,
            off + self._OUT_FILE_PATH,
            response.file_path,
            MAX_FILE_PATH_LEN,
        )

        struct.pack_into("<i", buf, state_off, self._FILLED)
        self._set_writer_index(widx + 1)

    def read_response(self, timeout_s: float | None = None) -> VideoResponse | None:
        """Blocking read of a VideoResponse from the next output slot."""
        buf = self._buf
        ridx = self._get_reader_index()
        off = (ridx % self._slots) * self._slot_size
        state_off = off + self._OUT_STATE
        deadline = (time.monotonic() + timeout_s) if timeout_s is not None else None

        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._FILLED:
                break
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(self._POLL_INTERVAL_S)
        else:
            return None

        task_id = self._unpack_task_id(buf, off + self._OUT_TASK_ID)
        status = VideoStatus(struct.unpack_from("<I", buf, off + self._OUT_STATUS)[0])
        error_message = self._unpack_string(
            buf, off + self._OUT_ERROR_LEN, off + self._OUT_ERROR
        )
        file_path = self._unpack_string(
            buf, off + self._OUT_FILE_PATH_LEN, off + self._OUT_FILE_PATH
        )

        struct.pack_into("<i", buf, state_off, self._EMPTY)
        self._set_reader_index(ridx + 1)

        return VideoResponse(
            task_id=task_id,
            status=status,
            file_path=file_path,
            error_message=error_message,
        )

    # ── Helpers ──

    @staticmethod
    def _pack_task_id(buf: memoryview, offset: int, task_id: str) -> None:
        raw = task_id.encode("utf-8")[: VideoShm.TASK_ID_SIZE]
        buf[offset : offset + VideoShm.TASK_ID_SIZE] = raw.ljust(
            VideoShm.TASK_ID_SIZE, b"\x00"
        )

    @staticmethod
    def _unpack_task_id(buf: memoryview, offset: int) -> str:
        return (
            bytes(buf[offset : offset + VideoShm.TASK_ID_SIZE])
            .decode("utf-8")
            .rstrip("\x00")
        )

    @staticmethod
    def _pack_string(
        buf: memoryview,
        len_offset: int,
        data_offset: int,
        text: str,
        max_len: int,
    ) -> None:
        raw = text.encode("utf-8")
        if len(raw) > max_len:
            raw = raw[:max_len]
            # Re-decode dropping any incomplete trailing multi-byte sequence
            raw = raw.decode("utf-8", errors="ignore").encode("utf-8")
        struct.pack_into("<I", buf, len_offset, len(raw))
        buf[data_offset : data_offset + max_len] = raw.ljust(max_len, b"\x00")

    @staticmethod
    def _unpack_string(buf: memoryview, len_offset: int, data_offset: int) -> str:
        length = struct.unpack_from("<I", buf, len_offset)[0]
        return bytes(buf[data_offset : data_offset + length]).decode("utf-8")
