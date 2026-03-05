#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""DeepSeek V3 B1 inference bridge.

Stage-0 (mesh_id == 0) bridges the C++ inference server via shared memory.
On receiving a prefill message (prompt token_ids + max_tokens), it runs the
full inference loop (prefill + decode) and streams each generated token back.

Launch with:
    tt-run --rank-binding bh_4x2_multi_mesh_rank_binding.yaml <path_to_this_file>
"""

import argparse
import mmap
import os
import signal
import struct
import sys
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.demo.model_pipeline import ModelPipeline

# ── Shared-memory layout ─────────────────────────────────────────────────────
# Must match the C++ SharedMemory class in shared_memory.hpp.
# Message layout: state(4) + pad(16) + payload_length(4) + max_tokens(4) + num_token_ids(4) + payload(8192)
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

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _rank() -> int:
    r = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("RANK")
    return int(r) if r is not None else 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("tt-cpp-server runner")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("/mnt/models/deepseek-ai/cache-mar3-2/"),
        help="Path to the weight cache directory (required for --weights real)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        choices=("synthetic", "real"),
        default="synthetic",
        help="Use synthetic or real (cached) weights (default: synthetic)",
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def _shm_recv(buf, pos: int):
    """Block until a TAKEN slot is available, consume it, return (task_id, token_ids, max_tokens)."""
    msg_off = pos * _MESSAGE_SIZE
    state_off = msg_off + _STATE_OFFSET
    while not _shutdown:
        if struct.unpack_from("<i", buf, state_off)[0] == _TAKEN:
            break
    if _shutdown:
        return None, pos

    max_tokens = struct.unpack_from("<I", buf, msg_off + _MAX_TOKENS_OFFSET)[0]
    num_token_ids = struct.unpack_from("<I", buf, msg_off + _NUM_TOKEN_IDS_OFFSET)[0]

    payload_off = msg_off + _PAYLOAD_OFFSET
    task_id = bytes(buf[payload_off : payload_off + _TASK_ID_SIZE])

    token_ids = []
    token_data_off = payload_off + _TASK_ID_SIZE
    for i in range(num_token_ids):
        tid = struct.unpack_from("<q", buf, token_data_off + i * _TOKEN_ID_SIZE)[0]
        token_ids.append(tid)

    struct.pack_into("<i", buf, state_off, _FREE)
    return (task_id, token_ids, max_tokens), (pos + 1) % _NUM_SLOTS


def _shm_send_token(buf, task_id: bytes, token_id: int, pos: int) -> int:
    """Write a single generated token back to the C++ server."""
    msg_off = pos * _MESSAGE_SIZE
    state_off = msg_off + _STATE_OFFSET
    while not _shutdown:
        if struct.unpack_from("<i", buf, state_off)[0] == _FREE:
            break
    if _shutdown:
        return pos

    payload_length = _TASK_ID_SIZE + _TOKEN_ID_SIZE
    struct.pack_into("<I", buf, msg_off + _PAYLOAD_LENGTH_OFFSET, payload_length)
    struct.pack_into("<I", buf, msg_off + _MAX_TOKENS_OFFSET, 0)
    struct.pack_into("<I", buf, msg_off + _NUM_TOKEN_IDS_OFFSET, 1)

    payload_off = msg_off + _PAYLOAD_OFFSET
    buf[payload_off : payload_off + _TASK_ID_SIZE] = task_id[:_TASK_ID_SIZE]
    struct.pack_into("<q", buf, payload_off + _TASK_ID_SIZE, int(token_id))

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


def _shm_is_configured() -> bool:
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")
    if not (c2p_name and p2c_name):
        return False
    return os.path.exists(f"/dev/shm/{c2p_name}") and os.path.exists(
        f"/dev/shm/{p2c_name}"
    )


def _shm_inference_loop(model_pipeline: ModelPipeline, c2p_buf, p2c_buf) -> None:
    """Read prefill requests from C++, run full inference, stream tokens back."""
    recv_pos = 0
    send_pos = 0
    print("Starting inference loop")
    while not _shutdown:
        msg, recv_pos = _shm_recv(c2p_buf, recv_pos)
        if msg is None:
            break
        print(f"Received message: {msg}")
        task_id, token_ids, max_tokens = msg

        def send_token(generated_token_id: int) -> None:
            nonlocal send_pos
            send_pos = _shm_send_token(p2c_buf, task_id, generated_token_id, send_pos)

        print(
            f"Running inference for task {task_id} with {len(token_ids)} token_ids and max_tokens {max_tokens}"
        )
        model_pipeline.run_inference(
            prompt_token_ids=token_ids,
            max_new_tokens=max_tokens,
            on_token=send_token,
        )
        print("Inference completed")


def _run_shm_bridge(model_pipeline: ModelPipeline) -> None:
    """Open shared-memory buffers and run the inference loop."""
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")
    if not (c2p_name and p2c_name):
        raise RuntimeError(
            "Shared memory bridge not configured. Set TT_IPC_SHM_C2P and TT_IPC_SHM_P2C."
        )
    c2p_buf = _open_shm(c2p_name)
    p2c_buf = _open_shm(p2c_name)
    if not (c2p_buf and p2c_buf):
        raise RuntimeError("Shared memory configured but could not be opened")
    try:
        _shm_inference_loop(model_pipeline, c2p_buf, p2c_buf)
    finally:
        c2p_buf.close()
        p2c_buf.close()


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    rank = _rank()
    args = _parse_args(sys.argv[1:])

    try:
        shm_enabled = _shm_is_configured()
        if not shm_enabled:
            raise RuntimeError(
                "Shared memory bridge not configured. Set TT_IPC_SHM_C2P and TT_IPC_SHM_P2C "
                "and ensure both exist under /dev/shm/."
            )

        print(f"Rank {rank}: Opening model pipeline")
        model_pipeline = ModelPipeline(
            cache_path=args.cache_path,
            use_real_weights=args.weights == "real",
        )

        if rank == 0:
            _run_shm_bridge(model_pipeline)
        else:
            print(f"Rank {rank}: Waiting (non-host)")
            while not _shutdown:
                signal.pause()

        model_pipeline.pipeline.terminate()
    finally:
        if ttnn is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
