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
from multiprocessing import resource_tracker as _resource_tracker
from multiprocessing import shared_memory as _shm
from typing import Callable


def _detach_from_resource_tracker(shm: _shm.SharedMemory) -> None:
    """Prevent multiprocessing.resource_tracker from shm_unlink()-ing the
    segment on Python process exit. The C++ peer is the segment's real owner;
    if Python unlinks, C++'s still-mapped inode becomes orphaned and any
    subsequent Python restart creates a *different* inode under the same name,
    silently disconnecting the two processes.

    See https://bugs.python.org/issue38119 for context.
    """
    try:
        _resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass


PREFILL_MAX_TOKEN_IDS = 131072  # matches C++ sp_pipeline::PREFILL_MAX_TOKEN_IDS (128k)
DECODE_MAX_TOKEN_IDS = 1

# Matches C++ ipc::SlotRingBufferState (slot_ring_buffer.hpp):
#   uint64_t writerIndex (offset 0) + uint64_t readerIndex (offset 8)
_STATE_SHM_SIZE = 16
_WRITER_INDEX_OFF = 0
_READER_INDEX_OFF = 8


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

    _EMPTY = 0
    _FILLED = 1

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
        self._writer_index = 0
        self._reader_index = 0
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
            self._shm = _shm.SharedMemory(name=self._name, create=False)
        _detach_from_resource_tracker(self._shm)
        os.chmod(f"/dev/shm/{self._name}", 0o666)
        self._buf = self._shm.buf

        # Open (or create) the persisted-cursor segment. We detach from
        # resource_tracker below so Python does NOT unlink on process exit —
        # C++ is the lifecycle owner, and unlinking while C++ still maps the
        # segment would orphan the inode (see _detach_from_resource_tracker).
        state_name = f"{self._name}_state"
        try:
            self._shm_state = _shm.SharedMemory(name=state_name, create=False)
        except FileNotFoundError:
            self._shm_state = _shm.SharedMemory(
                name=state_name, create=True, size=_STATE_SHM_SIZE
            )
        _detach_from_resource_tracker(self._shm_state)
        os.chmod(f"/dev/shm/{state_name}", 0o666)

        # Load writer/reader indices from state (matches C++ SlotRingBuffer::open()).
        # Both are 0 after fresh creation (kernel zero-fill + ftruncate).
        self._writer_index = struct.unpack_from(
            "<Q", self._shm_state.buf, _WRITER_INDEX_OFF
        )[0]
        self._reader_index = struct.unpack_from(
            "<Q", self._shm_state.buf, _READER_INDEX_OFF
        )[0]
        print(
            f"SharedMemory({self._name}): Loaded indices "
            f"writer={self._writer_index} reader={self._reader_index}",
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

    def _advance_reader(self) -> None:
        """Advance reader index and persist only the reader field in state
        (matches C++ SlotRingBuffer::advanceReaderIndex())."""
        self._reader_index = (self._reader_index + 1) % self.SLOTS
        if self._shm_state:
            struct.pack_into(
                "<Q", self._shm_state.buf, _READER_INDEX_OFF, self._reader_index
            )

    def _advance_writer(self) -> None:
        """Advance writer index and persist only the writer field in state
        (matches C++ SlotRingBuffer::advanceWriterIndex())."""
        self._writer_index = (self._writer_index + 1) % self.SLOTS
        if self._shm_state:
            struct.pack_into(
                "<Q", self._shm_state.buf, _WRITER_INDEX_OFF, self._writer_index
            )

    def read(self) -> PrefillMessage | None:
        """Blocking read. Spins until a FILLED slot appears or shutdown is signalled.
        Advances reader_index after reading to match C++ SlotRingBuffer behavior."""
        buf = self._buf
        msg_off = self._reader_index * self._message_size
        state_off = msg_off + self._STATE_OFF

        print(
            f"SharedMemory({self._name}): read() checking slot {self._reader_index}",
            file=__import__("sys").stderr,
        )

        iterations = 0
        while not self._is_shutdown():
            slot_state = struct.unpack_from("<i", buf, state_off)[0]
            if slot_state == self._FILLED:
                print(
                    f"SharedMemory({self._name}): Found FILLED slot {self._reader_index}",
                    file=__import__("sys").stderr,
                )
                break
            iterations += 1
            if iterations % 100000000 == 0:
                print(
                    f"SharedMemory({self._name}): Still waiting on slot {self._reader_index}, state={slot_state}, iterations={iterations}",
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

        struct.pack_into("<i", buf, state_off, self._EMPTY)
        self._advance_reader()

        print(
            f"SharedMemory({self._name}): read() completed, advanced to slot {self._reader_index}",
            file=__import__("sys").stderr,
        )

        return PrefillMessage(
            task_id=task_id, token_ids=token_ids, max_tokens=max_tokens
        )

    def write_token(self, task_id: int, token_id: int) -> None:
        """Write a single generated token (decode phase).
        Advances writer_index after writing to match C++ SlotRingBuffer behavior."""
        buf = self._buf
        msg_off = self._writer_index * self._message_size
        state_off = msg_off + self._STATE_OFF

        print(
            f"SharedMemory({self._name}): write_token() to slot {self._writer_index}",
            file=__import__("sys").stderr,
        )

        # Wait for EMPTY slot
        while not self._is_shutdown():
            if struct.unpack_from("<i", buf, state_off)[0] == self._EMPTY:
                break
        else:
            return

        struct.pack_into("<I", buf, msg_off + self._MAX_TOKENS_OFF, 0)
        struct.pack_into("<I", buf, msg_off + self._NUM_TOKEN_IDS_OFF, 1)
        struct.pack_into("<I", buf, msg_off + self._TASK_ID_OFF, task_id)
        struct.pack_into("<I", buf, msg_off + self._FAST_MODE_OFF, 0)

        struct.pack_into("<q", buf, msg_off + self._TOKEN_IDS_OFF, int(token_id))

        struct.pack_into("<i", buf, state_off, self._FILLED)
        self._advance_writer()

        print(
            f"SharedMemory({self._name}): write_token() completed, advanced to slot {self._writer_index}",
            file=__import__("sys").stderr,
        )

    def __enter__(self) -> SharedMemory:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
