# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#!/usr/bin/env python3
"""Test script to verify shared memory communication with runner.py.

This script mimics the C++ server's shared memory operations:
1. Writes a prefill message (uuid + token_ids + max_tokens) to C2P
2. Reads generated tokens from P2C
3. Validates the roundtrip communication

Usage:
    python test_shm_roundtrip.py
"""

import mmap
import os
import struct
import sys
import time

# ── Shared-memory layout (must match C++ Message<MaxTokenIds> template) ──
# template<int MaxTokenIds>
# struct Message {
#     atomic<int32_t> state;          // offset 0
#     uint32_t        max_tokens;     // offset 4
#     uint32_t        num_token_ids;  // offset 8
#     char            task_id[36];    // offset 12
#     uint64_t        token_ids[MaxTokenIds]; // offset 48
# };
_NUM_SLOTS = 1024
_HEADER_SIZE = 48
_TOKEN_ID_SIZE = 8
_TASK_ID_SIZE = 36

_PREFILL_MAX_TOKEN_IDS = 32768
_DECODE_MAX_TOKEN_IDS = 1

_C2P_MESSAGE_SIZE = _HEADER_SIZE + _PREFILL_MAX_TOKEN_IDS * _TOKEN_ID_SIZE
_P2C_MESSAGE_SIZE = _HEADER_SIZE + _DECODE_MAX_TOKEN_IDS * _TOKEN_ID_SIZE

_STATE_OFF = 0
_MAX_TOKENS_OFF = 4
_NUM_TOKEN_IDS_OFF = 8
_TASK_ID_OFF = 12
_TOKEN_IDS_OFF = 48

_FREE = 0
_TAKEN = 1


def _open_shm(shm_name: str, message_size: int):
    """Open existing shared memory (same as runner.py)."""
    path = f"/dev/shm/{shm_name}"
    if not os.path.exists(path):
        print(f"Error: Shared memory {path} does not exist", file=sys.stderr)
        return None
    shm_size = _NUM_SLOTS * message_size
    fd = os.open(path, os.O_RDWR)
    try:
        return mmap.mmap(
            fd, shm_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )
    finally:
        os.close(fd)


def _shm_write_prefill(
    buf, task_id: str, token_ids: list[int], max_tokens: int, pos: int
) -> int:
    """Write prefill message to C2P shared memory (mimics C++ PrefillSharedMemory::write)."""
    msg_off = pos * _C2P_MESSAGE_SIZE
    state_off = msg_off + _STATE_OFF

    timeout = 5.0
    start = time.time()
    while struct.unpack_from("<i", buf, state_off)[0] != _FREE:
        if time.time() - start > timeout:
            print(f"Error: Timeout waiting for FREE slot at pos {pos}", file=sys.stderr)
            return pos
        time.sleep(0.001)

    task_id_bytes = task_id.encode("utf-8")[:_TASK_ID_SIZE].ljust(
        _TASK_ID_SIZE, b"\x00"
    )

    struct.pack_into("<I", buf, msg_off + _MAX_TOKENS_OFF, max_tokens)
    struct.pack_into("<I", buf, msg_off + _NUM_TOKEN_IDS_OFF, len(token_ids))

    buf[msg_off + _TASK_ID_OFF : msg_off + _TASK_ID_OFF + _TASK_ID_SIZE] = task_id_bytes

    token_ids_off = msg_off + _TOKEN_IDS_OFF
    for i, tid in enumerate(token_ids):
        struct.pack_into("<q", buf, token_ids_off + i * _TOKEN_ID_SIZE, tid)

    buf.flush()
    struct.pack_into("<i", buf, state_off, _TAKEN)
    buf.flush()

    print(
        f"Wrote prefill: task_id={task_id}, token_ids={token_ids}, max_tokens={max_tokens}"
    )
    return (pos + 1) % _NUM_SLOTS


def _shm_read_token(buf, pos: int) -> tuple[str | None, int | None, int]:
    """Read a single token from P2C shared memory (mimics C++ DecodeSharedMemory::try_read)."""
    msg_off = pos * _P2C_MESSAGE_SIZE
    state_off = msg_off + _STATE_OFF

    if struct.unpack_from("<i", buf, state_off)[0] != _TAKEN:
        return None, None, pos

    num_token_ids = struct.unpack_from("<I", buf, msg_off + _NUM_TOKEN_IDS_OFF)[0]

    task_id_bytes = bytes(
        buf[msg_off + _TASK_ID_OFF : msg_off + _TASK_ID_OFF + _TASK_ID_SIZE]
    )
    task_id = task_id_bytes.rstrip(b"\x00").decode("utf-8")

    token_id = None
    if num_token_ids > 0:
        token_id = struct.unpack_from("<q", buf, msg_off + _TOKEN_IDS_OFF)[0]

    struct.pack_into("<i", buf, state_off, _FREE)

    return task_id, token_id, (pos + 1) % _NUM_SLOTS


def test_roundtrip():
    """Test roundtrip communication with runner.py."""
    c2p_name = os.environ.get("TT_IPC_SHM_C2P", "tt_ipc_c2p_12345")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C", "tt_ipc_p2c_12345")

    print(f"Opening shared memory: C2P={c2p_name}, P2C={p2c_name}")

    c2p_buf = _open_shm(c2p_name, _C2P_MESSAGE_SIZE)
    p2c_buf = _open_shm(p2c_name, _P2C_MESSAGE_SIZE)

    if not c2p_buf or not p2c_buf:
        print(
            "Error: Could not open shared memory. Make sure runner.py is running.",
            file=sys.stderr,
        )
        return 1

    try:
        test_task_id = "2a3533ab258758aae0c141f5"
        test_token_ids = [0, 19923, 2058]
        test_max_tokens = 16

        print("\nTest data:")
        print(f"  task_id: {test_task_id}")
        print(f"  token_ids: {test_token_ids}")
        print(f"  max_tokens: {test_max_tokens}")

        write_pos = 0
        write_pos = _shm_write_prefill(
            c2p_buf, test_task_id, test_token_ids, test_max_tokens, write_pos
        )

        print("\nReading tokens from P2C...")
        read_pos = 0
        received_tokens = []
        timeout = 10.0
        start = time.time()

        while len(received_tokens) < test_max_tokens:
            if time.time() - start > timeout:
                print(
                    f"\nWarning: Timeout after {timeout}s, received {len(received_tokens)} tokens"
                )
                break

            task_id, token_id, read_pos = _shm_read_token(p2c_buf, read_pos)
            if task_id is not None and token_id is not None:
                print(
                    f"  Received token {len(received_tokens) + 1}/{test_max_tokens}: {token_id} (task_id={task_id})"
                )
                received_tokens.append(token_id)

                if task_id != test_task_id:
                    print(
                        f"  Warning: task_id mismatch! Expected {test_task_id}, got {task_id}"
                    )
            else:
                time.sleep(0.001)

        print(f"\n{'=' * 60}")
        print("Test Summary:")
        print(f"  Sent: {len(test_token_ids)} token_ids, max_tokens={test_max_tokens}")
        print(f"  Received: {len(received_tokens)} tokens")
        print(f"  Tokens: {received_tokens}")

        if len(received_tokens) == test_max_tokens:
            print(f"  ✓ Received all {test_max_tokens} tokens")
        else:
            print(f"  ✗ Expected {test_max_tokens} tokens, got {len(received_tokens)}")

        print(f"{'=' * 60}")

        return 0

    finally:
        if c2p_buf:
            c2p_buf.close()
        if p2c_buf:
            p2c_buf.close()


if __name__ == "__main__":
    sys.exit(test_roundtrip())
