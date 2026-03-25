#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Video shared memory IPC for cross-process video generation.

Wraps a slot-based ring buffer in POSIX shared memory for video request/response
exchange between the server process and an external runner process.

Two modes:
  - "input"  (server -> runner): carries VideoRequest payloads
  - "output" (runner -> server): carries VideoResponse payloads (file-path reference)

The video payload itself is written to a file on ``/dev/shm/`` (RAM-backed tmpfs)
and only the file path (~100 bytes) is transmitted through the SHM output slot.
This avoids multiple large memcpy operations for 100-200 MB video blobs.

Runner side (attaches to SHM created by the device worker)::

    input_shm  = VideoShm("video_in",  mode="input")
    output_shm = VideoShm("video_out", mode="output")
    input_shm.open(create=False)
    output_shm.open(create=False)

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

    The actual video payload is written to a file (see :func:`video_result_path`)
    and only the path is transmitted through SHM.
    """

    INPUT_SLOTS = 8
    OUTPUT_SLOTS = 4
    TASK_ID_SIZE = 36
    MAX_PROMPT_LEN = 2048
    MAX_NEG_PROMPT_LEN = 512
    MAX_ERROR_LEN = 256

    _FREE = 0
    _TAKEN = 1
    _POLL_INTERVAL_S = 0.0001

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
        self._pos = 0
        self._is_shutdown = is_shutdown

    @property
    def name(self) -> str:
        return self._name

    # ── Lifecycle ──

    def open(self, *, create: bool = True) -> None:
        if create:
            try:
                self._shm = _shm.SharedMemory(
                    name=self._name, create=True, size=self._total_size
                )
            except FileExistsError:
                temp_shm = _shm.SharedMemory(name=self._name, create=False)
                temp_shm.unlink()
                temp_shm.close()
                self._shm = _shm.SharedMemory(
                    name=self._name, create=True, size=self._total_size
                )
            os.chmod(f"/dev/shm/{self._name}", 0o666)
        else:
            self._shm = _shm.SharedMemory(name=self._name, create=False)
        self._buf = self._shm.buf

    def close(self) -> None:
        if self._shm:
            self._shm.close()
            self._shm = None
            self._buf = None

    def unlink(self) -> None:
        """Remove the SHM region from /dev/shm/. Only the creator should call this."""
        if self._shm:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> VideoShm:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── Input SHM: request read / write ──

    def write_request(self, request: VideoRequest) -> None:
        """Write a VideoRequest into the next free input slot (spin-waits)."""
        buf = self._buf
        off = self._pos * self._slot_size
        state_off = off + self._IN_STATE

        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._FREE:
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

        struct.pack_into("<i", buf, state_off, self._TAKEN)
        self._pos = (self._pos + 1) % self._slots

    def read_request(self, timeout_s: float | None = None) -> VideoRequest | None:
        """Blocking read of a VideoRequest from the next input slot."""
        buf = self._buf
        off = self._pos * self._slot_size
        state_off = off + self._IN_STATE
        deadline = (time.monotonic() + timeout_s) if timeout_s is not None else None

        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._TAKEN:
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

        struct.pack_into("<i", buf, state_off, self._FREE)
        self._pos = (self._pos + 1) % self._slots

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
        off = self._pos * self._slot_size
        state_off = off + self._OUT_STATE

        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._FREE:
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

        struct.pack_into("<i", buf, state_off, self._TAKEN)
        self._pos = (self._pos + 1) % self._slots

    def read_response(self, timeout_s: float | None = None) -> VideoResponse | None:
        """Blocking read of a VideoResponse from the next output slot."""
        buf = self._buf
        off = self._pos * self._slot_size
        state_off = off + self._OUT_STATE
        deadline = (time.monotonic() + timeout_s) if timeout_s is not None else None

        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._TAKEN:
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

        struct.pack_into("<i", buf, state_off, self._FREE)
        self._pos = (self._pos + 1) % self._slots

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
