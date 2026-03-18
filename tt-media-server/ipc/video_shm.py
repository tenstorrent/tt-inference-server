#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Video shared memory IPC for cross-process video generation.

Wraps a slot-based ring buffer in POSIX shared memory for video request/frame
exchange between the server process and an external runner process.

Two modes:
  - "input"  (server -> runner): carries VideoRequest payloads
  - "output" (runner -> server): carries FrameResult payloads

Runner side (attaches to SHM created by the device worker)::

    input_shm  = VideoShm("video_in",  mode="input")
    output_shm = VideoShm("video_out", mode="output")
    input_shm.open(create=False)
    output_shm.open(create=False)

    req = input_shm.read_request()       # blocks until request arrives
    for i in range(req.num_frames):
        output_shm.write_frame(FrameResult(..., status=FrameStatus.FRAME))
    output_shm.write_frame(FrameResult(..., status=FrameStatus.DONE))
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from multiprocessing import shared_memory as _shm
from typing import Callable


class FrameStatus(IntEnum):
    FRAME = 0
    DONE = 1
    ERROR = 2


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


@dataclass(frozen=True)
class FrameResult:
    task_id: str
    status: FrameStatus
    frame_index: int
    total_frames: int
    height: int
    width: int
    channels: int
    frame_data: bytes


class VideoShm:
    """Slot-based ring buffer over POSIX shared memory for video IPC.

    Layout per slot depends on *mode*:

    Input slot (2 640 B):
        state(4) task_id(36) prompt_length(4) prompt(2048)
        neg_prompt_length(4) negative_prompt(512)
        num_inference_steps(4) seed(8) height(4) width(4)
        num_frames(4) guidance_scale(4) guidance_scale_2(4)

    Output slot (2 764 868 B):
        state(4) task_id(36) status(4) frame_index(4)
        total_frames(4) height(4) width(4) channels(4)
        frame_data_length(4) frame_data(MAX_FRAME_SIZE)
    """

    SLOTS = 8
    TASK_ID_SIZE = 36
    MAX_PROMPT_LEN = 2048
    MAX_NEG_PROMPT_LEN = 512
    MAX_FRAME_SIZE = 1280 * 720 * 3

    _FREE = 0
    _TAKEN = 1
    _POLL_INTERVAL_S = 0.0001  # 100µs between slot polls; negligible for video fps

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
    _OUT_STATE = 0
    _OUT_TASK_ID = 4
    _OUT_STATUS = 40
    _OUT_FRAME_INDEX = 44
    _OUT_TOTAL_FRAMES = 48
    _OUT_HEIGHT = 52
    _OUT_WIDTH = 56
    _OUT_CHANNELS = 60
    _OUT_FRAME_DATA_LEN = 64
    _OUT_FRAME_DATA = 68
    OUTPUT_SLOT_SIZE = 68 + MAX_FRAME_SIZE

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
        self._slot_size = (
            self.INPUT_SLOT_SIZE if mode == "input" else self.OUTPUT_SLOT_SIZE
        )
        self._total_size = self.SLOTS * self._slot_size
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
        self._pos = (self._pos + 1) % self.SLOTS

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
        self._pos = (self._pos + 1) % self.SLOTS

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

    # ── Output SHM: frame read / write ──

    def write_frame(self, frame: FrameResult) -> None:
        """Write a FrameResult into the next free output slot (spin-waits)."""
        if len(frame.frame_data) > self.MAX_FRAME_SIZE:
            raise ValueError(
                f"frame_data ({len(frame.frame_data)} bytes) exceeds "
                f"MAX_FRAME_SIZE ({self.MAX_FRAME_SIZE})"
            )
        buf = self._buf
        off = self._pos * self._slot_size
        state_off = off + self._OUT_STATE

        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._FREE:
                break
            time.sleep(self._POLL_INTERVAL_S)
        else:
            return

        self._pack_task_id(buf, off + self._OUT_TASK_ID, frame.task_id)

        struct.pack_into("<I", buf, off + self._OUT_STATUS, int(frame.status))
        struct.pack_into("<I", buf, off + self._OUT_FRAME_INDEX, frame.frame_index)
        struct.pack_into("<I", buf, off + self._OUT_TOTAL_FRAMES, frame.total_frames)
        struct.pack_into("<I", buf, off + self._OUT_HEIGHT, frame.height)
        struct.pack_into("<I", buf, off + self._OUT_WIDTH, frame.width)
        struct.pack_into("<I", buf, off + self._OUT_CHANNELS, frame.channels)

        data_len = len(frame.frame_data)
        struct.pack_into("<I", buf, off + self._OUT_FRAME_DATA_LEN, data_len)
        data_off = off + self._OUT_FRAME_DATA
        buf[data_off : data_off + data_len] = frame.frame_data

        struct.pack_into("<i", buf, state_off, self._TAKEN)
        self._pos = (self._pos + 1) % self.SLOTS

    def read_frame(self, timeout_s: float | None = None) -> FrameResult | None:
        """Blocking read of a FrameResult from the next output slot."""
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

        status = FrameStatus(struct.unpack_from("<I", buf, off + self._OUT_STATUS)[0])
        frame_index = struct.unpack_from("<I", buf, off + self._OUT_FRAME_INDEX)[0]
        total_frames = struct.unpack_from("<I", buf, off + self._OUT_TOTAL_FRAMES)[0]
        height = struct.unpack_from("<I", buf, off + self._OUT_HEIGHT)[0]
        width = struct.unpack_from("<I", buf, off + self._OUT_WIDTH)[0]
        channels = struct.unpack_from("<I", buf, off + self._OUT_CHANNELS)[0]

        data_len = struct.unpack_from("<I", buf, off + self._OUT_FRAME_DATA_LEN)[0]
        data_len = min(data_len, self.MAX_FRAME_SIZE)
        data_off = off + self._OUT_FRAME_DATA
        frame_data = bytes(buf[data_off : data_off + data_len])

        struct.pack_into("<i", buf, state_off, self._FREE)
        self._pos = (self._pos + 1) % self.SLOTS

        return FrameResult(
            task_id=task_id,
            status=status,
            frame_index=frame_index,
            total_frames=total_frames,
            height=height,
            width=width,
            channels=channels,
            frame_data=frame_data,
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
