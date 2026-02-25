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

# SHM page layout for c2p/p2c – must match C++ (token_to_page / page_to_result)
_TASK_ID_SIZE = 36
_TOKEN_ID_OFF = 36
_TOKEN_ID_SIZE = 8
_PAGE_SIZE = 64
_NUM_SLOTS = 1024
_CHANNEL_HEADER = 64
_CHANNEL_SIZE = _CHANNEL_HEADER + _NUM_SLOTS * _PAGE_SIZE
_SHM_SIZE = 2 * _CHANNEL_SIZE
_C2P_READ_POS = 0
_C2P_WRITE_POS = 4
_C2P_SLOTS_OFF = _CHANNEL_HEADER
_P2C_READ_POS = _CHANNEL_SIZE
_P2C_WRITE_POS = _CHANNEL_SIZE + 4
_P2C_SLOTS_OFF = _CHANNEL_SIZE + _CHANNEL_HEADER

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _rank():
    r = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("RANK")
    return int(r) if r is not None else 0


def _shm_recv_from_cpp(buf):
    r = struct.unpack_from("<I", buf, _C2P_READ_POS)[0]
    while not _shutdown:
        w = struct.unpack_from("<I", buf, _C2P_WRITE_POS)[0]
        if r != w:
            break
    if _shutdown:
        return None
    slot_off = _C2P_SLOTS_OFF + (r % _NUM_SLOTS) * _PAGE_SIZE
    data = bytes(buf[slot_off : slot_off + _PAGE_SIZE])
    struct.pack_into("<I", buf, _C2P_READ_POS, r + 1)
    return data


def _shm_sync_to_cpp(buf):
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        MS_SYNC = 0x2
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        libc.msync(ctypes.c_void_p(addr), len(buf), MS_SYNC)
    except Exception:
        pass


def _shm_send_to_cpp(buf, payload):
    while not _shutdown:
        r = struct.unpack_from("<I", buf, _P2C_READ_POS)[0]
        w = struct.unpack_from("<I", buf, _P2C_WRITE_POS)[0]
        if (w - r) % (1 << 32) < _NUM_SLOTS:
            break
    if _shutdown:
        return
    slot_off = _P2C_SLOTS_OFF + (w % _NUM_SLOTS) * _PAGE_SIZE
    buf[slot_off : slot_off + _PAGE_SIZE] = payload
    struct.pack_into("<I", buf, _P2C_WRITE_POS, w + 1)
    _shm_sync_to_cpp(buf)


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


def _shm_pipeline_bridge(pipeline_block, shm_buf):
    """Read token pages from C++, push through the pipeline, send output_tensor back.
    The pipeline output (embedding row, 64 bytes) is sent back so the simulated
    pipeline returns what was computed from the input page, not the page itself.
    """
    token_elems = TOKEN_SIZE_BYTES // 4
    while not _shutdown:
        page = _shm_recv_from_cpp(shm_buf)
        if page is None:
            break
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

        # Loop the page back, not the output tensor: pipeline D2H returns embedding rows, not
        # task_id/token_id; C++ page_to_result() and find_sequence() require the original IDs.
        payload = bytearray(_PAGE_SIZE)
        payload[:_TASK_ID_SIZE] = page[:_TASK_ID_SIZE]
        payload[_TOKEN_ID_OFF : _TOKEN_ID_OFF + _TOKEN_ID_SIZE] = page[
            _TOKEN_ID_OFF : _TOKEN_ID_OFF + _TOKEN_ID_SIZE
        ]
        _shm_send_to_cpp(shm_buf, bytes(payload))


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
            shm_name = os.environ.get("TT_IPC_SHM")
            if shm_name:
                shm_buf = _open_shm(shm_name)
                if shm_buf:
                    try:
                        _shm_pipeline_bridge(pipeline_block, shm_buf)
                    finally:
                        shm_buf.close()

        # It seems like terminate() will immediatelly terminate the other processes but that will not happen
        # This is because in order for them to terminate, they all need to meet at this barrier
        # And the rank 0 will not reach it since it's in busy loop
        pipeline_block.terminate()
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
