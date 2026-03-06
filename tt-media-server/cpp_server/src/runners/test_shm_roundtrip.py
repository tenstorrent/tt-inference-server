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

# ── Shared-memory layout (must match C++ SharedMemory and runner.py) ─────────
_NUM_SLOTS = 1024
_MESSAGE_SIZE = 8224
_SHM_SIZE = _NUM_SLOTS * _MESSAGE_SIZE
_STATE_OFFSET = 0
_PAYLOAD_LENGTH_OFFSET = 20  # 4 (state) + 16 (pad)
_MAX_TOKENS_OFFSET = 24
_NUM_TOKEN_IDS_OFFSET = 28
_PAYLOAD_OFFSET = 32
_TASK_ID_SIZE = 36
_TOKEN_ID_SIZE = 8
_FREE = 0
_TAKEN = 1


def _open_shm(shm_name: str):
    """Open existing shared memory (same as runner.py)."""
    path = f"/dev/shm/{shm_name}"
    if not os.path.exists(path):
        print(f"Error: Shared memory {path} does not exist", file=sys.stderr)
        return None
    fd = os.open(path, os.O_RDWR)
    try:
        return mmap.mmap(
            fd, _SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )
    finally:
        os.close(fd)


def _shm_write_prefill(
    buf, task_id: str, token_ids: list[int], max_tokens: int, pos: int
) -> int:
    """Write prefill message to C2P shared memory (mimics C++ SharedMemory::write)."""
    msg_off = pos * _MESSAGE_SIZE
    state_off = msg_off + _STATE_OFFSET

    # Wait for FREE slot
    timeout = 5.0
    start = time.time()
    while struct.unpack_from("<i", buf, state_off)[0] != _FREE:
        if time.time() - start > timeout:
            print(f"Error: Timeout waiting for FREE slot at pos {pos}", file=sys.stderr)
            return pos
        time.sleep(0.001)

    # Encode task_id (36 bytes, pad with zeros)
    task_id_bytes = task_id.encode("utf-8")[:_TASK_ID_SIZE].ljust(
        _TASK_ID_SIZE, b"\x00"
    )

    # Encode token_ids
    token_ids_bytes = b"".join(struct.pack("<q", tid) for tid in token_ids)

    # Calculate payload
    payload = task_id_bytes + token_ids_bytes
    payload_length = len(payload)

    # Write metadata
    struct.pack_into("<I", buf, msg_off + _PAYLOAD_LENGTH_OFFSET, payload_length)
    struct.pack_into("<I", buf, msg_off + _MAX_TOKENS_OFFSET, max_tokens)
    struct.pack_into("<I", buf, msg_off + _NUM_TOKEN_IDS_OFFSET, len(token_ids))

    # Write payload
    buf[msg_off + _PAYLOAD_OFFSET : msg_off + _PAYLOAD_OFFSET + payload_length] = (
        payload
    )

    # Flush and mark TAKEN
    buf.flush()
    struct.pack_into("<i", buf, state_off, _TAKEN)
    buf.flush()

    print(
        f"Wrote prefill: task_id={task_id}, token_ids={token_ids}, max_tokens={max_tokens}"
    )
    return (pos + 1) % _NUM_SLOTS


def _shm_read_token(buf, pos: int) -> tuple[str | None, int | None, int]:
    """Read a single token from P2C shared memory (mimics C++ SharedMemory::try_read)."""
    msg_off = pos * _MESSAGE_SIZE
    state_off = msg_off + _STATE_OFFSET

    # Check if TAKEN
    if struct.unpack_from("<i", buf, state_off)[0] != _TAKEN:
        return None, None, pos

    # Read metadata
    num_token_ids = struct.unpack_from("<I", buf, msg_off + _NUM_TOKEN_IDS_OFFSET)[0]
    payload_length = struct.unpack_from("<I", buf, msg_off + _PAYLOAD_LENGTH_OFFSET)[0]

    # Read payload
    payload_off = msg_off + _PAYLOAD_OFFSET
    task_id_bytes = bytes(buf[payload_off : payload_off + _TASK_ID_SIZE])
    task_id = task_id_bytes.rstrip(b"\x00").decode("utf-8")

    # Read token_id (should be 1 token for decode response)
    if num_token_ids > 0 and payload_length >= _TASK_ID_SIZE + _TOKEN_ID_SIZE:
        token_id = struct.unpack_from("<q", buf, payload_off + _TASK_ID_SIZE)[0]
    else:
        token_id = None

    # Mark FREE
    struct.pack_into("<i", buf, state_off, _FREE)

    return task_id, token_id, (pos + 1) % _NUM_SLOTS


def test_roundtrip():
    """Test roundtrip communication with runner.py."""
    # Get shared memory names from environment
    c2p_name = os.environ.get("TT_IPC_SHM_C2P", "tt_ipc_c2p_12345")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C", "tt_ipc_p2c_12345")

    print(f"Opening shared memory: C2P={c2p_name}, P2C={p2c_name}")

    # Open shared memory
    c2p_buf = _open_shm(c2p_name)
    p2c_buf = _open_shm(p2c_name)

    if not c2p_buf or not p2c_buf:
        print(
            "Error: Could not open shared memory. Make sure runner.py is running.",
            file=sys.stderr,
        )
        return 1

    try:
        # Test data (same as your example)
        test_task_id = "2a3533ab258758aae0c141f5"
        test_token_ids = [0, 19923, 2058]
        test_max_tokens = 16

        print("\nTest data:")
        print(f"  task_id: {test_task_id}")
        print(f"  token_ids: {test_token_ids}")
        print(f"  max_tokens: {test_max_tokens}")

        # Write prefill message
        write_pos = 0
        write_pos = _shm_write_prefill(
            c2p_buf, test_task_id, test_token_ids, test_max_tokens, write_pos
        )

        # Read tokens back
        print("\nReading tokens from P2C...")
        read_pos = 0
        received_tokens = []
        timeout = 10.0  # 10 seconds timeout
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

                # Verify task_id matches
                if task_id != test_task_id:
                    print(
                        f"  Warning: task_id mismatch! Expected {test_task_id}, got {task_id}"
                    )
            else:
                time.sleep(0.001)  # Small sleep to avoid busy-wait

        # Summary
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
