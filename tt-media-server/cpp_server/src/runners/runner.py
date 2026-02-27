#!/usr/bin/env python3
"""4-stage blitz-decode pipeline bridge using PipelineBlock.
Stage-0 (mesh_id == 0) bridges the C++ inference server via SHM while
all 4 stages relay data through D2D sockets on a single BH Galaxy.
Launch with:
    tt-run --rank-binding bh_4x2_multi_mesh_rank_binding.yaml \\
        ttnn/ttnn/distributed/ttrun_hello_world.py
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

# Pipeline sizing: 64-byte end-to-end pages.
# embedding_dim=32 bfloat16 elements == 64 bytes per row, matching the H2D token page size
# so the same page size flows unchanged through H2D → D2D → D2H.
EMBEDDING_DIM = 32
EMBEDDING_VOCAB = 131072  # covers max token id (~125836) in fixed_reply_sequence
EMBEDDING_DTYPE = torch.bfloat16
EMBEDDING_SIZE_BYTES = EMBEDDING_DIM * 2  # 32 × 2 = 64 bytes
TOKEN_SIZE_BYTES = 64  # H2D page: task_id (36 B) + token_id (8 B) + pad
FIFO_SIZE = 128  # 2 × page size for minimal in-flight buffering
PIPELINE_CORE = ttnn.CoreCoord(0, 0)
FABRIC_MAX_PAYLOAD = 7168

# SharedMemory layout – must match C++ SharedMemory class (shared_memory.hpp)
# Message: atomic state (4B) + pad (16B) + payload (44B) = 64B
_NUM_SLOTS = 1024
_MESSAGE_SIZE = 64
_SHM_SIZE = _NUM_SLOTS * _MESSAGE_SIZE
_STATE_OFF = 0
_PAYLOAD_OFF = 20  # 4 (state) + 16 (pad)
_PAYLOAD_SIZE = 44
_TASK_ID_SIZE = 36
_TOKEN_ID_OFF = 36
_TOKEN_ID_SIZE = 8
_FREE = 0
_TAKEN = 1

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _rank():
    r = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("RANK")
    return int(r) if r is not None else 0


def _shm_recv(buf, pos):
    """Read one payload from a SharedMemory ring buffer (wait for TAKEN, read, set FREE)."""
    msg_off = pos * _MESSAGE_SIZE
    state_off = msg_off + _STATE_OFF
    payload_off = msg_off + _PAYLOAD_OFF
    while not _shutdown:
        state = struct.unpack_from("<i", buf, state_off)[0]
        if state == _TAKEN:
            break
    if _shutdown:
        return None, pos
    data = bytes(buf[payload_off : payload_off + _PAYLOAD_SIZE])
    struct.pack_into("<i", buf, state_off, _FREE)
    return data, (pos + 1) % _NUM_SLOTS


def _shm_send(buf, payload, pos):
    """Write one payload to a SharedMemory ring buffer (wait for FREE, write, set TAKEN)."""
    msg_off = pos * _MESSAGE_SIZE
    state_off = msg_off + _STATE_OFF
    payload_off = msg_off + _PAYLOAD_OFF
    while not _shutdown:
        state = struct.unpack_from("<i", buf, state_off)[0]
        if state == _FREE:
            break
    if _shutdown:
        return pos
    padded = payload[:_PAYLOAD_SIZE].ljust(_PAYLOAD_SIZE, b"\x00")
    buf[payload_off : payload_off + _PAYLOAD_SIZE] = padded
    struct.pack_into("<i", buf, state_off, _TAKEN)
    return (pos + 1) % _NUM_SLOTS


def _open_shm(shm_name):
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


def _create_pipeline_block(mesh_device):
    if mesh_device.get_system_mesh_id() == 0:
        torch_embedding = torch.randn(
            (1, 1, EMBEDDING_VOCAB, EMBEDDING_DIM), dtype=EMBEDDING_DTYPE
        )
        embedding_tensor = ttnn.from_torch(
            torch_embedding,
            dtype=ttnn_dtype_from_torch_dtype(EMBEDDING_DTYPE),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        embedding_tensor = ttnn.to_device(
            embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE,
            FIFO_SIZE,  # upstream d2d socket fifo size
            FIFO_SIZE,  # downstream d2d socket fifo size
            EMBEDDING_SIZE_BYTES,  # upstream d2d socket page size (= d2h page size)
            EMBEDDING_SIZE_BYTES,  # downstream d2d socket page size (= embedding row size)
            h2d_socket_fifo_size=FIFO_SIZE,
            d2h_socket_fifo_size=FIFO_SIZE,
            d2h_socket_page_size=EMBEDDING_SIZE_BYTES,
            embedding_tensor=embedding_tensor,
        )
    return PipelineBlock(
        mesh_device,
        PIPELINE_CORE,
        FIFO_SIZE,
        FIFO_SIZE,
        EMBEDDING_SIZE_BYTES,
        EMBEDDING_SIZE_BYTES,
    )


def _shm_pipeline_bridge(pipeline_block, c2p_buf, p2c_buf):
    """Single-thread bridge: recv → tensor conversion → write_token → read_output → send.
    Avoids Python GIL contention that a two-thread design introduces (~5ms switch interval).
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


def main():
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
            c2p_name = os.environ.get("TT_IPC_SHM_C2P")
            p2c_name = os.environ.get("TT_IPC_SHM_P2C")
            if c2p_name and p2c_name:
                c2p_buf = _open_shm(c2p_name)
                p2c_buf = _open_shm(p2c_name)
                if c2p_buf and p2c_buf:
                    try:
                        _shm_pipeline_bridge(pipeline_block, c2p_buf, p2c_buf)
                    finally:
                        c2p_buf.close()
                        p2c_buf.close()

        # It seems like terminate() will immediatelly terminate the other processes but that will not happen
        # This is because in order for them to terminate, they all need to meet at this barrier
        # And the rank 0 will not reach it since it's in busy loop
        pipeline_block.terminate()
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
