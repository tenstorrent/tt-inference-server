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
# Must match the C++ Message struct in shared_memory.hpp.
#   state(4) + max_tokens(4) + num_tokens(4) + task_id(36) + reserved(16) = 64-byte header
#   tokens: 128000 × uint64 = 1024000 bytes
#   Total message: 1024064 bytes, 1 slot
_NUM_SLOTS = 1
_MAX_TOKENS_COUNT = 128000
_HEADER_SIZE = 64
_MAX_PAYLOAD_SIZE = _MAX_TOKENS_COUNT * 8
_MESSAGE_SIZE = _HEADER_SIZE + _MAX_PAYLOAD_SIZE
_SHM_SIZE = _NUM_SLOTS * _MESSAGE_SIZE

_STATE_OFFSET = 0
_MAX_TOKENS_OFFSET = 4
_NUM_TOKENS_OFFSET = 8
_TASK_ID_OFFSET = 12
_TASK_ID_SIZE = 36
_TOKENS_OFFSET = 64
_TOKEN_SIZE = 8

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

    while not _shutdown:
        if struct.unpack_from("<i", buf, state_off)[0] == _TAKEN:
            break
        time.sleep(0.0001)
    if _shutdown:
        return None, pos

    max_tokens = struct.unpack_from("<I", buf, msg_off + _MAX_TOKENS_OFFSET)[0]
    num_tokens = struct.unpack_from("<I", buf, msg_off + _NUM_TOKENS_OFFSET)[0]
    task_id_raw = buf[msg_off + _TASK_ID_OFFSET : msg_off + _TASK_ID_OFFSET + _TASK_ID_SIZE]
    task_id = bytes(task_id_raw).decode("utf-8", errors="ignore").rstrip("\x00")

    tokens_off = msg_off + _TOKENS_OFFSET
    token_ids = []
    for i in range(num_tokens):
        token_id = struct.unpack_from("<Q", buf, tokens_off + i * _TOKEN_SIZE)[0]
        token_ids.append(token_id)

    struct.pack_into("<i", buf, state_off, _FREE)

    message_data = {
        "task_id": task_id,
        "max_tokens": max_tokens,
        "num_tokens": num_tokens,
        "token_ids": token_ids,
    }

    return message_data, (pos + 1) % _NUM_SLOTS


def _shm_send(buf, task_id: str, token_ids: list, max_tokens: int, pos: int) -> int:
    """Block until a FREE slot is available, write the message, and mark TAKEN."""
    msg_off = pos * _MESSAGE_SIZE
    state_off = msg_off + _STATE_OFFSET

    while not _shutdown:
        if struct.unpack_from("<i", buf, state_off)[0] == _FREE:
            break
        time.sleep(0.001)
    if _shutdown:
        return pos

    struct.pack_into("<I", buf, msg_off + _MAX_TOKENS_OFFSET, max_tokens)
    struct.pack_into("<I", buf, msg_off + _NUM_TOKENS_OFFSET, len(token_ids))

    task_id_bytes = task_id.encode("utf-8")[:_TASK_ID_SIZE].ljust(_TASK_ID_SIZE, b"\x00")
    buf[msg_off + _TASK_ID_OFFSET : msg_off + _TASK_ID_OFFSET + _TASK_ID_SIZE] = task_id_bytes

    tokens_off = msg_off + _TOKENS_OFFSET
    for i, tid in enumerate(token_ids):
        struct.pack_into("<Q", buf, tokens_off + i * _TOKEN_SIZE, tid)

    struct.pack_into("<i", buf, state_off, _TAKEN)

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
            f"max_tokens={max_tokens}, num_tokens={len(token_ids)}, "
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
