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

from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import (
    ttnn_dtype_from_torch_dtype,
)

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
# Must match the C++ SharedMemory class in shared_memory.hpp.
# Message layout: atomic state (4 B) + pad (16 B) + payload (44 B) = 64 B
_NUM_SLOTS = 1024
_MESSAGE_SIZE = 64
_SHM_SIZE = _NUM_SLOTS * _MESSAGE_SIZE
_STATE_OFFSET = 0
_PAYLOAD_OFFSET = 20  # 4 (state) + 16 (pad)
_PAYLOAD_SIZE = 44
_TASK_ID_SIZE = 36
_TOKEN_ID_OFFSET = 36
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


def _shm_recv(buf, pos: int):
    """Block until a TAKEN slot is available, consume it, and return the payload."""
    msg_off = pos * _MESSAGE_SIZE
    state_off = msg_off + _STATE_OFFSET
    payload_off = msg_off + _PAYLOAD_OFFSET
    while not _shutdown:
        if struct.unpack_from("<i", buf, state_off)[0] == _TAKEN:
            break
    if _shutdown:
        return None, pos
    data = bytes(buf[payload_off : payload_off + _PAYLOAD_SIZE])
    struct.pack_into("<i", buf, state_off, _FREE)
    return data, (pos + 1) % _NUM_SLOTS


def _shm_send(buf, payload: bytes, pos: int) -> int:
    """Block until a FREE slot is available, write the payload, and mark TAKEN."""
    msg_off = pos * _MESSAGE_SIZE
    state_off = msg_off + _STATE_OFFSET
    payload_off = msg_off + _PAYLOAD_OFFSET
    while not _shutdown:
        if struct.unpack_from("<i", buf, state_off)[0] == _FREE:
            break
    if _shutdown:
        return pos
    padded = payload[:_PAYLOAD_SIZE].ljust(_PAYLOAD_SIZE, b"\x00")
    buf[payload_off : payload_off + _PAYLOAD_SIZE] = padded
    struct.pack_into("<i", buf, state_off, _TAKEN)
    return (pos + 1) % _NUM_SLOTS


def _open_shm(shm_name: str):
    path = f"/dev/shm/{shm_name}"
    if not os.path.exists(path):
        return None
    fd = os.open(path, os.O_RDWR)
    try:
        return mmap.mmap(
            fd, _SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )
    finally:
        os.close(fd)


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
        payload, recv_pos = _shm_recv(c2p_buf, recv_pos)
        if payload is None:
            break

        page = bytearray(TOKEN_SIZE_BYTES)
        page[: len(payload)] = payload
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

        send_pos = _shm_send(p2c_buf, payload, send_pos)


def _run_shm_bridge(pipeline_block) -> None:
    """Open shared-memory buffers and run the pipeline bridge if env vars are set."""
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")
    if not (c2p_name and p2c_name):
        return
    c2p_buf = _open_shm(c2p_name)
    p2c_buf = _open_shm(p2c_name)
    if not (c2p_buf and p2c_buf):
        return
    try:
        _shm_pipeline_bridge(pipeline_block, c2p_buf, p2c_buf)
    finally:
        c2p_buf.close()
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
