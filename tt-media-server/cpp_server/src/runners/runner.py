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
import argparse
import struct
import sys
from pathlib import Path

import ttnn

from models.demos.deepseek_v3_b1.demo.model_pipeline import ModelPipeline
from models.demos.deepseek_v3_b1.demo.pipeline import create_fabric_router_config

# ── Token sizing ─────────────────────────────────────────────────────────────
TOKEN_SIZE_BYTES = 64  # H2D page: task_id (36 B) + token_id (8 B) + pad

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
    parser.add_argument(
        "--fabric-max-payload-bytes",
        type=int,
        default=15232,
        help="Fabric router max packet payload bytes (must match DeepSeek V3 B1 pipeline config)",
    )
    parser.add_argument(
        "--trace-region-size-bytes",
        type=int,
        default=573440,
        help="TTNN trace region size bytes (must match DeepSeek V3 B1 pipeline config)",
    )
    parser.add_argument(
        "--fabric-router-sync-timeout-ms",
        type=int,
        default=30000,
        help="Set TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS (ms).",
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


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


def _shm_is_configured() -> bool:
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")
    if not (c2p_name and p2c_name):
        return False
    return os.path.exists(f"/dev/shm/{c2p_name}") and os.path.exists(f"/dev/shm/{p2c_name}")


def _fabric_config_for_num_procs(num_procs: int):
    if num_procs == 4:
        return ttnn.FabricConfig.FABRIC_2D
    if num_procs in (16, 64):
        return ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    raise ValueError(f"Unsupported num_procs for fabric config: {num_procs} (expected 4, 16, or 64)")


def _open_mesh_device(
    fabric_max_payload_bytes: int,
    trace_region_size_bytes: int,
    fabric_router_sync_timeout_ms: int,
):
    os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = str(fabric_router_sync_timeout_ms)

    num_procs = int(ttnn.distributed_context_get_size())
    fabric_router_config = create_fabric_router_config(fabric_max_payload_bytes)
    ttnn.set_fabric_config(
        _fabric_config_for_num_procs(num_procs),
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        fabric_router_config,
    )
    try:
        return ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(4, 2),
            trace_region_size=trace_region_size_bytes,
        )
    except TypeError:
        return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 2))


def _parse_token_id(payload: bytes) -> int:
    if len(payload) < (_TOKEN_ID_OFFSET + _TOKEN_ID_SIZE):
        raise ValueError(f"payload too small: expected at least {_TOKEN_ID_OFFSET + _TOKEN_ID_SIZE} bytes")
    # Extract token_id as little-endian uint64 from payload[36:44].
    return int(struct.unpack_from("<Q", payload, _TOKEN_ID_OFFSET)[0])


def _write_token_id(payload: bytes, token_id: int) -> bytes:
    if len(payload) < _PAYLOAD_SIZE:
        payload = payload.ljust(_PAYLOAD_SIZE, b"\x00")
    out = bytearray(payload[:_PAYLOAD_SIZE])
    # Overwrite token_id as little-endian uint64 at payload[36:44], preserving task_id bytes.
    struct.pack_into("<Q", out, _TOKEN_ID_OFFSET, int(token_id))
    return bytes(out)


def _shm_pipeline_bridge(model_pipeline: ModelPipeline, c2p_buf, p2c_buf) -> None:
    """Single-thread bridge: recv → decode_forward(token_id) → send(updated token_id).

    A single-thread design avoids Python GIL contention (~5 ms switch interval)
    that a two-thread design would introduce.
    """
    recv_pos = 0
    send_pos = 0

    while not _shutdown:
        payload, recv_pos = _shm_recv(c2p_buf, recv_pos)
        if payload is None:
            break

        input_token_id = _parse_token_id(payload)
        output_token_id = model_pipeline.decode_forward(input_token_id)
        out_payload = _write_token_id(payload, output_token_id)
        send_pos = _shm_send(p2c_buf, out_payload, send_pos)


def _run_shm_bridge(model_pipeline: ModelPipeline) -> None:
    """Open shared-memory buffers and run the pipeline bridge if env vars are set."""
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")
    if not (c2p_name and p2c_name):
        return
    c2p_buf = _open_shm(c2p_name)
    p2c_buf = _open_shm(p2c_name)
    if not (c2p_buf and p2c_buf):
        raise RuntimeError("Shared memory configured but could not be opened")
    try:
        _shm_pipeline_bridge(model_pipeline, c2p_buf, p2c_buf)
    finally:
        c2p_buf.close()
        p2c_buf.close()


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    rank = _rank()
    args = _parse_args(sys.argv[1:])

    try:
        try:
            ttnn.init_distributed_context()
        except Exception as e:
            # Some launchers may initialize distributed context before invoking this script.
            if "already" not in str(e).lower():
                raise
        mesh_device = _open_mesh_device(
            fabric_max_payload_bytes=args.fabric_max_payload_bytes,
            trace_region_size_bytes=args.trace_region_size_bytes,
            fabric_router_sync_timeout_ms=args.fabric_router_sync_timeout_ms,
        )
    except Exception as e:
        print(f"Rank {rank}: failed to open mesh device: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        shm_enabled = _shm_is_configured()
        if not shm_enabled:
            raise RuntimeError(
                "Shared memory bridge not configured. Set TT_IPC_SHM_C2P and TT_IPC_SHM_P2C "
                "and ensure both exist under /dev/shm/."
            )
        
        cache_path = args.cache_path
        use_real_weights = args.weights == "real"

        model_pipeline = ModelPipeline(
            mesh_device=mesh_device,
            cache_path=cache_path,
            use_real_weights=use_real_weights,
        )

        if mesh_device.get_system_mesh_id() == 0:
            _run_shm_bridge(model_pipeline)
        else:
            while not _shutdown:
                signal.pause()

        # All ranks must reach terminate() together — it acts as a collective barrier.
        model_pipeline.pipeline.terminate()
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
