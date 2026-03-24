#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Python mirror of the C++ SharedMemory class (shared_memory.hpp).

Wraps a ring buffer in POSIX shared memory for cross-process IPC.
One instance per direction (C2P or P2C).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from multiprocessing import shared_memory as _shm
from typing import Callable

PREFILL_MAX_TOKEN_IDS = 131072  # matches C++ sp_pipeline::PREFILL_MAX_TOKEN_IDS (128k)
DECODE_MAX_TOKEN_IDS = 1


@dataclass(frozen=True)
class PrefillMessage:
    task_id: bytes
    token_ids: list[int]
    max_tokens: int


class SharedMemory:
    """Python equivalent of the C++ SharedMemory<MaxTokenIds> template.

    Layout per slot:
        state(4) + max_tokens(4) + num_token_ids(4)
        + task_id(36) + token_ids(max_token_ids × 8)
    """

    SLOTS = 64
    TOKEN_ID_SIZE = 8
    TASK_ID_SIZE = 36
    _HEADER_SIZE = 48

    _STATE_OFF = 0
    _MAX_TOKENS_OFF = 4
    _NUM_TOKEN_IDS_OFF = 8
    _TASK_ID_OFF = 12
    _TOKEN_IDS_OFF = 48
    _FREE = 0
    _TAKEN = 1

    def __init__(
        self,
        name: str,
        *,
        max_token_ids: int = PREFILL_MAX_TOKEN_IDS,
        is_shutdown: Callable[[], bool] = lambda: False,
    ):
        self._name = name.lstrip("/")
        self._max_token_ids = max_token_ids
        self._message_size = self._HEADER_SIZE + max_token_ids * self.TOKEN_ID_SIZE
        self._total_size = self.SLOTS * self._message_size
        self._shm: _shm.SharedMemory | None = None
        self._buf: memoryview | None = None
        self._pos = 0
        self._is_shutdown = is_shutdown

    @property
    def name(self) -> str:
        return self._name

    def open(self) -> None:
        try:
            self._shm = _shm.SharedMemory(
                name=self._name, create=True, size=self._total_size
            )
        except FileExistsError:
            temp_shm = _shm.SharedMemory(name=self._name, create=False)
            temp_shm.unlink()  # delete the existing shared memory block
            self._shm = _shm.SharedMemory(
                name=self._name, create=True, size=self._total_size
            )
        os.chmod(f"/dev/shm/{self._name}", 0o666)
        self._buf = self._shm.buf

    def close(self) -> None:
        if self._shm:
            self._shm.close()
            self._shm = None
            self._buf = None

    def read(self) -> PrefillMessage | None:
        """Blocking read. Spins until a TAKEN slot appears or shutdown is signalled."""
        buf = self._buf
        msg_off = self._pos * self._message_size
        state_off = msg_off + self._STATE_OFF

        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._TAKEN:
                break
        else:
            return None

        max_tokens = struct.unpack_from("<I", buf, msg_off + self._MAX_TOKENS_OFF)[0]
        num_token_ids = struct.unpack_from(
            "<I", buf, msg_off + self._NUM_TOKEN_IDS_OFF
        )[0]

        task_id_off = msg_off + self._TASK_ID_OFF
        task_id = bytes(buf[task_id_off : task_id_off + self.TASK_ID_SIZE])

        token_ids_off = msg_off + self._TOKEN_IDS_OFF
        token_ids = list(struct.unpack_from(f"<{num_token_ids}q", buf, token_ids_off))

        struct.pack_into("<i", buf, state_off, self._FREE)
        self._pos = (self._pos + 1) % self.SLOTS

        return PrefillMessage(
            task_id=task_id, token_ids=token_ids, max_tokens=max_tokens
        )

    def write_token(self, task_id: bytes, token_id: int) -> None:
        """Write a single generated token (decode phase)."""
        buf = self._buf
        msg_off = self._pos * self._message_size
        state_off = msg_off + self._STATE_OFF

        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._FREE:
                break
        else:
            return

        struct.pack_into("<I", buf, msg_off + self._MAX_TOKENS_OFF, 0)
        struct.pack_into("<I", buf, msg_off + self._NUM_TOKEN_IDS_OFF, 1)

        task_id_off = msg_off + self._TASK_ID_OFF
        buf[task_id_off : task_id_off + self.TASK_ID_SIZE] = task_id[
            : self.TASK_ID_SIZE
        ]

        struct.pack_into("<q", buf, msg_off + self._TOKEN_IDS_OFF, int(token_id))

        struct.pack_into("<i", buf, state_off, self._TAKEN)
        self._pos = (self._pos + 1) % self.SLOTS

    def __enter__(self) -> SharedMemory:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


MEMORY_RESULT_MAX_KV_DESTINATIONS = 128

_MEM_REQ_SLOT_SIZE = 48
_MEM_REQ_STATE_OFF = 0
_MEM_REQ_INPUT_SEQ_LEN_OFF = 4
_MEM_REQ_ACTION_OFF = 8
_MEM_REQ_LAYOUT_OFF = 9
_MEM_REQ_TASK_ID_OFF = 12

_MEM_RES_STATE_OFF = 0
_MEM_RES_SUCCESS_OFF = 4
_MEM_RES_NUM_DEST_OFF = 8
_MEM_RES_TASK_ID_OFF = 12
_MEM_RES_DEST_OFF = 48


@dataclass(frozen=True)
class MemoryRequestView:
    task_id: str
    input_seq_len: int
    action: int
    memory_layout: int


@dataclass(frozen=True)
class MemoryResultView:
    task_id: str
    success: bool
    dram_addresses: list[int]
    semaphore_addresses: list[int]


class MemoryRequestSharedMemory:
    """Mirrors C++ SharedMemory<MemoryRequestSlot> (MemoryRequestQueue)."""

    def __init__(
        self,
        name: str,
        *,
        is_shutdown: Callable[[], bool] = lambda: False,
    ):
        self._name = name.lstrip("/")
        self._is_shutdown = is_shutdown
        self._shm: _shm.SharedMemory | None = None
        self._buf: memoryview | None = None
        self._pos = 0
        self._total_size = SharedMemory.SLOTS * _MEM_REQ_SLOT_SIZE

    def open(self, *, create: bool = True) -> None:
        if create:
            try:
                self._shm = _shm.SharedMemory(
                    name=self._name, create=True, size=self._total_size
                )
            except FileExistsError:
                t = _shm.SharedMemory(name=self._name, create=False)
                t.unlink()
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

    def write_request(
        self,
        task_id: str,
        *,
        input_seq_len: int,
        action: int,
        memory_layout: int,
    ) -> None:
        buf = self._buf
        assert buf is not None
        msg_off = self._pos * _MEM_REQ_SLOT_SIZE
        state_off = msg_off + _MEM_REQ_STATE_OFF
        while struct.unpack_from("<i", buf, state_off)[0] != SharedMemory._FREE:
            if self._is_shutdown():
                return
        tid = task_id.encode("utf-8")[:36].ljust(36, b"\x00")
        struct.pack_into(
            "<i", buf, msg_off + _MEM_REQ_INPUT_SEQ_LEN_OFF, int(input_seq_len)
        )
        buf[msg_off + _MEM_REQ_ACTION_OFF] = action & 0xFF
        buf[msg_off + _MEM_REQ_LAYOUT_OFF] = memory_layout & 0xFF
        buf[msg_off + 10 : msg_off + 12] = b"\x00\x00"
        buf[msg_off + _MEM_REQ_TASK_ID_OFF : msg_off + _MEM_REQ_TASK_ID_OFF + 36] = tid
        struct.pack_into("<i", buf, state_off, SharedMemory._TAKEN)
        self._pos = (self._pos + 1) % SharedMemory.SLOTS


class MemoryResultSharedMemory:
    """Mirrors C++ SharedMemory<MemoryResultSlot> (MemoryResultQueue)."""

    def __init__(
        self,
        name: str,
        *,
        is_shutdown: Callable[[], bool] = lambda: False,
    ):
        self._name = name.lstrip("/")
        self._is_shutdown = is_shutdown
        self._shm: _shm.SharedMemory | None = None
        self._buf: memoryview | None = None
        self._pos = 0
        self._slot_size = _MEM_RES_DEST_OFF + MEMORY_RESULT_MAX_KV_DESTINATIONS * 16
        self._total_size = SharedMemory.SLOTS * self._slot_size

    def open(self, *, create: bool = True) -> None:
        if create:
            try:
                self._shm = _shm.SharedMemory(
                    name=self._name, create=True, size=self._total_size
                )
            except FileExistsError:
                t = _shm.SharedMemory(name=self._name, create=False)
                t.unlink()
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

    def try_read_result(self) -> MemoryResultView | None:
        buf = self._buf
        assert buf is not None
        msg_off = self._pos * self._slot_size
        state_off = msg_off + _MEM_RES_STATE_OFF
        if struct.unpack_from("<i", buf, state_off)[0] != SharedMemory._TAKEN:
            return None
        success = struct.unpack_from("<I", buf, msg_off + _MEM_RES_SUCCESS_OFF)[0] != 0
        num_dest = struct.unpack_from("<I", buf, msg_off + _MEM_RES_NUM_DEST_OFF)[0]
        tid_raw = bytes(
            buf[msg_off + _MEM_RES_TASK_ID_OFF : msg_off + _MEM_RES_TASK_ID_OFF + 36]
        )
        tid = tid_raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        dram: list[int] = []
        sem: list[int] = []
        for i in range(min(num_dest, MEMORY_RESULT_MAX_KV_DESTINATIONS)):
            off = msg_off + _MEM_RES_DEST_OFF + i * 16
            dram.append(struct.unpack_from("<Q", buf, off)[0])
            sem.append(struct.unpack_from("<Q", buf, off + 8)[0])
        struct.pack_into("<i", buf, state_off, SharedMemory._FREE)
        self._pos = (self._pos + 1) % SharedMemory.SLOTS
        return MemoryResultView(
            task_id=tid, success=success, dram_addresses=dram, semaphore_addresses=sem
        )
