#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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

# Matches sp_pipeline::SharedMemoryState (shared_memory.hpp): uint64_t cursor only.
_STATE_SHM_SIZE = 8


@dataclass(frozen=True)
class PrefillMessage:
    task_id: int  # uint32_t in C++
    token_ids: list[int]
    max_tokens: int


class SharedMemory:
    """Python equivalent of the C++ SlotRingBuffer::Message template.

    Layout per slot (must match C++ exactly):
        state(4) + max_tokens(4) + num_token_ids(4) + task_id(4) + fast_mode(4)
        + padding(4) + token_ids(max_token_ids × 8)
    """

    SLOTS = 64
    TOKEN_ID_SIZE = 8

    # Offsets must match C++ Message struct exactly
    _STATE_OFF = 0
    _MAX_TOKENS_OFF = 4
    _NUM_TOKEN_IDS_OFF = 8
    _TASK_ID_OFF = 12
    _FAST_MODE_OFF = 16
    _TOKEN_IDS_OFF = 24  # With alignas(8), this starts at offset 24

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
        self._message_size = self._TOKEN_IDS_OFF + max_token_ids * self.TOKEN_ID_SIZE
        self._total_size = self.SLOTS * self._message_size
        self._shm: _shm.SharedMemory | None = None
        self._shm_state: _shm.SharedMemory | None = None
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

        # Initialize all slots to FREE state
        for slot_idx in range(self.SLOTS):
            slot_off = slot_idx * self._message_size + self._STATE_OFF
            struct.pack_into("<i", self._buf, slot_off, self._FREE)

        # C++ may create "{name}_state" for the persisted cursor. Open it here too so
        # multiprocessing.resource_tracker registers it and unlinks on process exit
        # (same lifecycle as the ring buffer), even if C++ created the segment first.
        state_name = f"{self._name}_state"
        try:
            self._shm_state = _shm.SharedMemory(name=state_name, create=False)
        except FileNotFoundError:
            self._shm_state = _shm.SharedMemory(
                name=state_name, create=True, size=_STATE_SHM_SIZE
            )
        os.chmod(f"/dev/shm/{state_name}", 0o666)

        # Load cursor from state (matches C++ SlotRingBuffer::open())
        # Should be 0 after fresh creation
        self._pos = struct.unpack_from("<Q", self._shm_state.buf, 0)[0]
        print(
            f"SharedMemory({self._name}): Loaded cursor position={self._pos}",
            file=__import__("sys").stderr,
        )

    def close(self) -> None:
        if self._shm_state:
            self._shm_state.close()
            self._shm_state = None
        if self._shm:
            self._shm.close()
            self._shm = None
            self._buf = None

    def _update_state(self) -> None:
        """Update the persisted cursor state (matches C++ SlotRingBuffer::updateState())"""
        if self._shm_state:
            struct.pack_into("<Q", self._shm_state.buf, 0, self._pos)

    def read(self) -> PrefillMessage | None:
        """Blocking read. Spins until a TAKEN slot appears or shutdown is signalled.
        Advances cursor after reading to match C++ SlotRingBuffer behavior."""
        buf = self._buf
        msg_off = self._pos * self._message_size
        state_off = msg_off + self._STATE_OFF

        print(
            f"SharedMemory({self._name}): read() checking slot {self._pos}",
            file=__import__("sys").stderr,
        )

        iterations = 0
        while not self._is_shutdown():
            slot_state = struct.unpack_from("<i", buf, state_off)[0]
            if slot_state == self._TAKEN:
                print(
                    f"SharedMemory({self._name}): Found TAKEN slot {self._pos}",
                    file=__import__("sys").stderr,
                )
                break
            iterations += 1
            if iterations % 100000000 == 0:
                print(
                    f"SharedMemory({self._name}): Still waiting on slot {self._pos}, state={slot_state}, iterations={iterations}",
                    file=__import__("sys").stderr,
                )
        else:
            return None

        max_tokens = struct.unpack_from("<I", buf, msg_off + self._MAX_TOKENS_OFF)[0]
        num_token_ids = struct.unpack_from(
            "<I", buf, msg_off + self._NUM_TOKEN_IDS_OFF
        )[0]

        task_id = struct.unpack_from("<I", buf, msg_off + self._TASK_ID_OFF)[0]

        token_ids_off = msg_off + self._TOKEN_IDS_OFF
        token_ids = list(struct.unpack_from(f"<{num_token_ids}q", buf, token_ids_off))

        struct.pack_into("<i", buf, state_off, self._FREE)
        self._pos = (self._pos + 1) % self.SLOTS
        self._update_state()

        print(
            f"SharedMemory({self._name}): read() completed, advanced to slot {self._pos}",
            file=__import__("sys").stderr,
        )

        return PrefillMessage(
            task_id=task_id, token_ids=token_ids, max_tokens=max_tokens
        )

    def write_token(self, task_id: int, token_id: int) -> None:
        """Write a single generated token (decode phase).
        Advances cursor after writing to match C++ SlotRingBuffer behavior."""
        buf = self._buf
        msg_off = self._pos * self._message_size
        state_off = msg_off + self._STATE_OFF

        print(
            f"SharedMemory({self._name}): write_token() to slot {self._pos}",
            file=__import__("sys").stderr,
        )

        # Wait for FREE slot
        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._FREE:
                break
        else:
            return

        struct.pack_into("<I", buf, msg_off + self._MAX_TOKENS_OFF, 0)
        struct.pack_into("<I", buf, msg_off + self._NUM_TOKEN_IDS_OFF, 1)
        struct.pack_into("<I", buf, msg_off + self._TASK_ID_OFF, task_id)
        struct.pack_into("<I", buf, msg_off + self._FAST_MODE_OFF, 0)

        struct.pack_into("<q", buf, msg_off + self._TOKEN_IDS_OFF, int(token_id))

        struct.pack_into("<i", buf, state_off, self._TAKEN)
        self._pos = (self._pos + 1) % self.SLOTS
        self._update_state()

        print(
            f"SharedMemory({self._name}): write_token() completed, advanced to slot {self._pos}",
            file=__import__("sys").stderr,
        )

    def __enter__(self) -> SharedMemory:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
