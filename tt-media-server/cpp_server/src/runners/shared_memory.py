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


@dataclass(frozen=True)
class PrefillMessage:
    task_id: bytes
    token_ids: list[int]
    max_tokens: int


class SharedMemory:
    """Python equivalent of the C++ SharedMemory class.

    Layout per slot (8224 bytes):
        state(4) + pad(16) + payload_length(4) + max_tokens(4)
        + num_token_ids(4) + payload(8192)

    Payload format:
        task_id (36 bytes, raw UUID) + token_ids (N × 8 bytes, little-endian int64)
    """

    SLOTS = 1024
    MAX_PAYLOAD_SIZE = 8192
    MESSAGE_SIZE = 8224
    TOTAL_SIZE = SLOTS * MESSAGE_SIZE
    TASK_ID_SIZE = 36
    TOKEN_ID_SIZE = 8

    _STATE_OFF = 0
    _PAYLOAD_LENGTH_OFF = 20
    _MAX_TOKENS_OFF = 24
    _NUM_TOKEN_IDS_OFF = 28
    _PAYLOAD_OFF = 32
    _FREE = 0
    _TAKEN = 1

    def __init__(self, name: str, *, is_shutdown: Callable[[], bool] = lambda: False):
        self._name = name.lstrip("/")
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
                name=self._name, create=True, size=self.TOTAL_SIZE
            )
        except FileExistsError:
            self._shm = _shm.SharedMemory(name=self._name)
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
        msg_off = self._pos * self.MESSAGE_SIZE
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

        payload_off = msg_off + self._PAYLOAD_OFF
        task_id = bytes(buf[payload_off : payload_off + self.TASK_ID_SIZE])

        token_data_off = payload_off + self.TASK_ID_SIZE
        token_ids = [
            struct.unpack_from("<q", buf, token_data_off + i * self.TOKEN_ID_SIZE)[0]
            for i in range(num_token_ids)
        ]

        struct.pack_into("<i", buf, state_off, self._FREE)
        self._pos = (self._pos + 1) % self.SLOTS

        return PrefillMessage(task_id=task_id, token_ids=token_ids, max_tokens=max_tokens)

    def write_token(self, task_id: bytes, token_id: int) -> None:
        """Write a single generated token (decode phase)."""
        buf = self._buf
        msg_off = self._pos * self.MESSAGE_SIZE
        state_off = msg_off + self._STATE_OFF

        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._FREE:
                break
        else:
            return

        struct.pack_into(
            "<I",
            buf,
            msg_off + self._PAYLOAD_LENGTH_OFF,
            self.TASK_ID_SIZE + self.TOKEN_ID_SIZE,
        )
        struct.pack_into("<I", buf, msg_off + self._MAX_TOKENS_OFF, 0)
        struct.pack_into("<I", buf, msg_off + self._NUM_TOKEN_IDS_OFF, 1)

        payload_off = msg_off + self._PAYLOAD_OFF
        buf[payload_off : payload_off + self.TASK_ID_SIZE] = task_id[: self.TASK_ID_SIZE]
        struct.pack_into("<q", buf, payload_off + self.TASK_ID_SIZE, int(token_id))

        struct.pack_into("<i", buf, state_off, self._TAKEN)
        self._pos = (self._pos + 1) % self.SLOTS

    def __enter__(self) -> SharedMemory:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
