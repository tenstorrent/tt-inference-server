#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Mock runner for testing shared memory IPC without ttnn dependencies.

This script echoes back input tokens one by one to simulate token generation.
Useful for testing the C++ inference server integration without hardware.

Launch directly with:
    python mock_runner.py
"""

import mmap
import os
import signal
import struct
import sys
import time

# ── Shared-memory layout ─────────────────────────────────────────────────────
# Must match the C++ SharedMemory class in shared_memory.hpp.
# Message layout: atomic state (4 B) + pad (16 B) + payload_length (4 B) + max_tokens (4 B) + num_token_ids (4 B) + payload (8192 B) = 8224 B
_NUM_SLOTS = 1024
_MESSAGE_SIZE = 8224
_SHM_SIZE = _NUM_SLOTS * _MESSAGE_SIZE
_STATE_OFFSET = 0
_PAYLOAD_LENGTH_OFFSET = 20  # 4 (state) + 16 (pad)
_MAX_TOKENS_OFFSET = 24  # 4 (state) + 16 (pad) + 4 (payload_length)
_NUM_TOKEN_IDS_OFFSET = 28  # 4 (state) + 16 (pad) + 4 (payload_length) + 4 (max_tokens)
_PAYLOAD_OFFSET = (
    32  # 4 (state) + 16 (pad) + 4 (payload_length) + 4 (max_tokens) + 4 (num_token_ids)
)
_MAX_PAYLOAD_SIZE = 8192
_TASK_ID_SIZE = 36
_TOKEN_ID_SIZE = 8
_FREE = 0
_TAKEN = 1

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _shm_recv(buf, pos: int):
    """Block until a TAKEN slot is available, consume it, and return the parsed message data."""
    msg_off = pos * _MESSAGE_SIZE
    state_off = msg_off + _STATE_OFFSET
    payload_length_off = msg_off + _PAYLOAD_LENGTH_OFFSET
    max_tokens_off = msg_off + _MAX_TOKENS_OFFSET
    num_token_ids_off = msg_off + _NUM_TOKEN_IDS_OFFSET
    payload_off = msg_off + _PAYLOAD_OFFSET

    while not _shutdown:
        if struct.unpack_from("<i", buf, state_off)[0] == _TAKEN:
            break
        time.sleep(0.0001)  # Small sleep to avoid busy-wait
    if _shutdown:
        return None, pos

    # Read metadata
    payload_length = struct.unpack_from("<I", buf, payload_length_off)[0]
    max_tokens = struct.unpack_from("<I", buf, max_tokens_off)[0]
    num_token_ids = struct.unpack_from("<I", buf, num_token_ids_off)[0]

    # Read payload
    data = bytes(buf[payload_off : payload_off + payload_length])

    # Parse: task_id (36 bytes) + token_ids (variable length)
    task_id = data[:_TASK_ID_SIZE].decode("utf-8", errors="ignore").rstrip("\x00")
    token_ids = []
    if num_token_ids > 0:
        token_bytes = data[_TASK_ID_SIZE:]
        for i in range(num_token_ids):
            offset = i * _TOKEN_ID_SIZE
            if offset + _TOKEN_ID_SIZE <= len(token_bytes):
                token_id = struct.unpack_from("<q", token_bytes, offset)[
                    0
                ]  # signed int64
                token_ids.append(token_id)

    struct.pack_into("<i", buf, state_off, _FREE)

    message_data = {
        "task_id": task_id,
        "max_tokens": max_tokens,
        "num_token_ids": num_token_ids,
        "token_ids": token_ids,
    }

    return message_data, (pos + 1) % _NUM_SLOTS


def _shm_send(buf, task_id: str, token_ids: list, max_tokens: int, pos: int) -> int:
    """Block until a FREE slot is available, write the message, and mark TAKEN."""
    msg_off = pos * _MESSAGE_SIZE
    state_off = msg_off + _STATE_OFFSET
    payload_length_off = msg_off + _PAYLOAD_LENGTH_OFFSET
    max_tokens_off = msg_off + _MAX_TOKENS_OFFSET
    num_token_ids_off = msg_off + _NUM_TOKEN_IDS_OFFSET
    payload_off = msg_off + _PAYLOAD_OFFSET

    while not _shutdown:
        if struct.unpack_from("<i", buf, state_off)[0] == _FREE:
            break
        time.sleep(0.001)  # Small sleep to avoid busy-wait
    if _shutdown:
        return pos

    # Encode task_id (36 bytes) + token_ids (variable)
    task_id_bytes = task_id.encode("utf-8")[:_TASK_ID_SIZE].ljust(
        _TASK_ID_SIZE, b"\x00"
    )
    token_ids_bytes = b"".join(
        struct.pack("<q", tid) for tid in token_ids
    )  # signed int64
    payload = task_id_bytes + token_ids_bytes
    payload_length = len(payload)

    # Write metadata
    struct.pack_into("<I", buf, payload_length_off, payload_length)
    struct.pack_into("<I", buf, max_tokens_off, max_tokens)
    struct.pack_into("<I", buf, num_token_ids_off, len(token_ids))

    # Write payload
    buf[payload_off : payload_off + payload_length] = payload

    # Ensure all writes are visible before marking TAKEN
    # This is critical for cross-process communication with C++ atomic operations
    buf.flush()

    struct.pack_into("<i", buf, state_off, _TAKEN)

    # Flush again to ensure TAKEN state is visible
    buf.flush()

    return (pos + 1) % _NUM_SLOTS


def _open_shm(shm_name: str):
    """Create or open shared memory region."""
    import posix_ipc

    # Create shared memory (or open if it exists)
    try:
        shm = posix_ipc.SharedMemory(
            shm_name, flags=posix_ipc.O_CREAT, mode=0o666, size=_SHM_SIZE
        )
    except posix_ipc.ExistentialError:
        shm = posix_ipc.SharedMemory(shm_name)

    try:
        # Map the shared memory into process address space
        buf = mmap.mmap(
            shm.fd, _SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )

        # Initialize to zero if newly created
        if buf[0:4] == b"\x00\x00\x00\x00":
            buf[:] = b"\x00" * _SHM_SIZE

        return buf
    finally:
        shm.close_fd()


def _mock_echo_bridge(c2p_buf, p2c_buf) -> None:
    """Mock bridge: recv tokens → echo them back one by one.

    This simulates token generation by echoing back the input tokens
    one at a time with a small delay to simulate generation time.
    """
    recv_pos = 0
    send_pos = 0

    print("Mock runner: Starting echo bridge", file=sys.stderr)

    while not _shutdown:
        message_data, recv_pos = _shm_recv(c2p_buf, recv_pos)
        if message_data is None:
            break

        task_id = message_data["task_id"]
        max_tokens = message_data["max_tokens"]
        token_ids = message_data["token_ids"]

        print(
            f"Mock runner: Received task_id={task_id}, "
            f"max_tokens={max_tokens}, "
            f"num_tokens={len(token_ids)}, "
            f"tokens={token_ids[:5]}{'...' if len(token_ids) > 5 else ''}",
            file=sys.stderr,
        )

        # Generate tokens up to max_tokens (echo input, then generate more if needed)
        tokens_to_generate = max_tokens if max_tokens > 0 else len(token_ids)

        for i in range(tokens_to_generate):
            # Echo input tokens first, then generate dummy tokens
            if i < len(token_ids):
                token_id = token_ids[i]
            else:
                # Generate a dummy token (just use a fixed value for testing)
                token_id = 12345

            # Send single token
            send_pos = _shm_send(p2c_buf, task_id, [token_id], max_tokens, send_pos)

            print(
                f"Mock runner: Sent token {i + 1}/{tokens_to_generate}: {token_id}",
                file=sys.stderr,
            )

            # Simulate generation time (~0.1ms per token)
            time.sleep(0.0001)

        print(
            f"Mock runner: Finished generating {tokens_to_generate} tokens for task {task_id}",
            file=sys.stderr,
        )


def _run_mock_bridge() -> None:
    """Open shared-memory buffers and run the mock echo bridge."""
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")

    if not (c2p_name and p2c_name):
        print(
            "Error: TT_IPC_SHM_C2P or TT_IPC_SHM_P2C not set",
            file=sys.stderr,
        )
        print("Example usage:", file=sys.stderr)
        print("  export TT_IPC_SHM_C2P=tt_ipc_c2p_12345", file=sys.stderr)
        print("  export TT_IPC_SHM_P2C=tt_ipc_p2c_12345", file=sys.stderr)
        print("  python mock_runner.py", file=sys.stderr)
        sys.exit(1)

    print(
        f"Mock runner: Opening shared memory C2P={c2p_name}, P2C={p2c_name}",
        file=sys.stderr,
    )

    c2p_buf = _open_shm(c2p_name)
    p2c_buf = _open_shm(p2c_name)

    if not (c2p_buf and p2c_buf):
        print("Error: Failed to open shared memory buffers", file=sys.stderr)
        sys.exit(1)

    print("Mock runner: SHM bridge started successfully", file=sys.stderr)

    try:
        _mock_echo_bridge(c2p_buf, p2c_buf)
    except KeyboardInterrupt:
        print("\nMock runner: Interrupted by user", file=sys.stderr)
    finally:
        if c2p_buf:
            c2p_buf.close()
        if p2c_buf:
            p2c_buf.close()
        print("Mock runner: Shutdown complete", file=sys.stderr)


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    print("Mock runner: Starting (no ttnn dependencies)", file=sys.stderr)
    _run_mock_bridge()


if __name__ == "__main__":
    main()
