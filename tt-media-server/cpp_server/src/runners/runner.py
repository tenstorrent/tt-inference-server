#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""4-stage blitz-decode pipeline bridge.

Stage-0 (mesh_id == 0) bridges the C++ inference server via shared memory
while all 4 stages relay data through D2D sockets on a single BH Galaxy.

Launch with:
    tt-run --rank-binding bh_4x2_multi_mesh_rank_binding.yaml <path_to_this_file>
"""

import mmap
import os
import signal
import struct
import sys

import torch
import ttnn
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import (
    ttnn_dtype_from_torch_dtype,
)
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock

# ── Pipeline sizing ──────────────────────────────────────────────────────────
# 64-byte end-to-end pages: 32 bfloat16 elements × 2 bytes = 64 bytes per row.
# Matches the H2D token page size so the same page flows H2D → D2D → D2H.
EMBEDDING_DIM = 32
EMBEDDING_VOCAB = 131072  # covers max token id (~125836) in fixed_reply_sequence
EMBEDDING_DTYPE = torch.bfloat16
EMBEDDING_SIZE_BYTES = EMBEDDING_DIM * 2  # 32 × 2 = 64 bytes
TOKEN_SIZE_BYTES = 64  # H2D page: task_id (36 B) + token_id (8 B) + pad
FIFO_SIZE = 128  # 2 × page size for minimal in-flight buffering
PIPELINE_CORE = ttnn.CoreCoord(0, 0)
FABRIC_MAX_PAYLOAD = 7168

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


def _rank() -> int:
    r = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("RANK")
    return int(r) if r is not None else 0


def _shm_recv(buf, pos: int):
    """Block until a TAKEN slot is available, consume it, and return the parsed message data."""
    msg_off = pos * _MESSAGE_SIZE
    state_off = msg_off + _STATE_OFFSET

    while not _shutdown:
        if struct.unpack_from("<i", buf, state_off)[0] == _TAKEN:
            break
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


def _open_mesh_device():
    fabric_router_config = ttnn._ttnn.fabric.FabricRouterConfig()
    fabric_router_config.max_packet_payload_size_bytes = FABRIC_MAX_PAYLOAD
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        fabric_router_config,
    )
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 2))


def _create_embedding_tensor(mesh_device):
    torch_embedding = torch.randn(
        (1, 1, EMBEDDING_VOCAB, EMBEDDING_DIM), dtype=EMBEDDING_DTYPE
    )
    embedding_tensor = ttnn.from_torch(
        torch_embedding,
        dtype=ttnn_dtype_from_torch_dtype(EMBEDDING_DTYPE),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    return ttnn.to_device(
        embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _create_pipeline_block(mesh_device) -> PipelineBlock:
    pipeline_args = (
        mesh_device,
        PIPELINE_CORE,
        FIFO_SIZE,  # upstream d2d socket fifo size
        FIFO_SIZE,  # downstream d2d socket fifo size
        EMBEDDING_SIZE_BYTES,  # upstream d2d socket page size
        EMBEDDING_SIZE_BYTES,  # downstream d2d socket page size
    )
    if mesh_device.get_system_mesh_id() != 0:
        return PipelineBlock(*pipeline_args)
    return PipelineBlock(
        *pipeline_args,
        h2d_socket_fifo_size=FIFO_SIZE,
        d2h_socket_fifo_size=FIFO_SIZE,
        d2h_socket_page_size=EMBEDDING_SIZE_BYTES,
        embedding_tensor=_create_embedding_tensor(mesh_device),
    )


def _shm_pipeline_bridge(pipeline_block, c2p_buf, p2c_buf) -> None:
    """Single-thread bridge: recv → tensor → write_token → read_output → send.

    A single-thread design avoids Python GIL contention (~5 ms switch interval)
    that a two-thread design would introduce.
    """
    token_elems = TOKEN_SIZE_BYTES // 4
    recv_pos = 0
    send_pos = 0

    while not _shutdown:
        message_data, recv_pos = _shm_recv(c2p_buf, recv_pos)
        if message_data is None:
            break

        task_id = message_data["task_id"]
        max_tokens = message_data["max_tokens"]
        token_ids = message_data["token_ids"]

        # Process token_ids (for now, just use the first token or all tokens)
        # You can extend this logic based on your prefill/decode requirements
        page = bytearray(TOKEN_SIZE_BYTES)

        # Pack token_ids into the page
        for i, token_id in enumerate(token_ids[:token_elems]):
            struct.pack_into("<Q", page, i * 8, token_id)

        token_ints = struct.unpack_from(f"<{token_elems}I", page)

        input_tensor = ttnn.from_torch(
            torch.tensor(token_ints, dtype=torch.uint32).reshape(1, -1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        pipeline_block.write_token(input_tensor)

        output_tensor = ttnn.from_torch(
            torch.zeros(1, EMBEDDING_DIM, dtype=EMBEDDING_DTYPE),
            dtype=ttnn_dtype_from_torch_dtype(EMBEDDING_DTYPE),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        pipeline_block.read_output(output_tensor)

        # Send back response (for now, echo the first token)
        response_token_ids = token_ids[:1] if token_ids else [0]
        send_pos = _shm_send(p2c_buf, task_id, response_token_ids, max_tokens, send_pos)


def _run_shm_bridge(pipeline_block) -> None:
    """Open shared-memory buffers and run the pipeline bridge if env vars are set."""
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")
    if not (c2p_name and p2c_name):
        print(
            "Warning: TT_IPC_SHM_C2P or TT_IPC_SHM_P2C not set, skipping SHM bridge",
            file=sys.stderr,
        )
        return

    c2p_buf = _open_shm(c2p_name)
    p2c_buf = _open_shm(p2c_name)

    if not (c2p_buf and p2c_buf):
        print("Warning: Failed to open shared memory buffers", file=sys.stderr)
        return

    print(f"SHM bridge started: C2P={c2p_name}, P2C={p2c_name}", file=sys.stderr)

    try:
        _shm_pipeline_bridge(pipeline_block, c2p_buf, p2c_buf)
    finally:
        if c2p_buf:
            c2p_buf.close()
        if p2c_buf:
            p2c_buf.close()


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    rank = _rank()

    try:
        mesh_device = _open_mesh_device()
    except Exception as e:
        print(f"Rank {rank}: failed to open mesh device: {e}", file=sys.stderr)
        sys.exit(1)

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    try:
        pipeline_block = _create_pipeline_block(mesh_device)
        pipeline_block.run()

        if pipeline_block.is_first_pipeline_stage():
            _run_shm_bridge(pipeline_block)

        # All ranks must reach terminate() together — it acts as a collective barrier.
        pipeline_block.terminate()
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
