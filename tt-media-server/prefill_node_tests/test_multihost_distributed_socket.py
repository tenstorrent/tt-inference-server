# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-host distributed tensor socket test.

This test validates tensor transfer between two SEPARATE physical hosts, each with
a Tenstorrent device. It automatically detects its MPI rank and behaves as
SENDER (rank 0) or RECEIVER (rank 1).

=== MULTI-HOST SETUP ===

For testing tensor transfer between two separate hosts:

1. Create a hostfile listing both hosts:
   ```
   echo "host1.example.com slots=1" > /data/ztorlak/hostfile
   echo "host2.example.com slots=1" >> /data/ztorlak/hostfile
   ```

2. Ensure passwordless SSH between hosts and /data/ztorlak is a shared filesystem.

3. On ONE host, run with tt-run:
   ```bash
   source /data/ztorlak/tt-metal/python_env/bin/activate
   export TT_METAL_HOME=/data/ztorlak/tt-metal
   export PYTHONPATH=/data/ztorlak/tt-metal/ttnn:/data/ztorlak/tt-inference-server/tt-media-server:$PYTHONPATH

   tt-run --rank-binding /data/ztorlak/tt-inference-server/tt-media-server/prefill_node_tests/multihost_n300_rank_bindings.yaml \\
          --mpi-args "--hostfile /data/ztorlak/hostfile --mca btl_tcp_if_exclude docker0,lo" \\
          python /data/ztorlak/tt-inference-server/tt-media-server/prefill_node_tests/test_multihost_distributed_socket.py
   ```

=== SLURM MULTI-NODE ===

For SLURM environments with multiple nodes:

1. Request an allocation spanning both nodes:
   ```bash
   salloc -N 2 -n 2 --ntasks-per-node=1 ...
   ```

2. Run tt-run (it will use srun automatically):
   ```bash
   tt-run --rank-binding /data/ztorlak/tt-inference-server/tt-media-server/prefill_node_tests/multihost_n300_rank_bindings.yaml \\
          python /data/ztorlak/tt-inference-server/tt-media-server/prefill_node_tests/test_multihost_distributed_socket.py
   ```
"""

import socket
import sys
import time
from typing import Any

import torch
import ttnn
from loguru import logger

from prefill_node_tests.node_utils import format_throughput, log_node_info

# Add the project to path
sys.path.insert(0, "/data/ztorlak/tt-inference-server/tt-media-server")

from prefill_node.distributed_tensor_sockets_utilities import (
    build_socket_config,
    create_tensor_socket,
)


def run_sender(device: Any, socket_config: Any, rank: int) -> dict[str, float]:
    """Run the sender side of the test."""
    timings: dict[str, float] = {}

    # Create socket to receiver (rank 1)
    other_rank = 1
    logger.info(f"Rank {rank}: Creating distributed socket to rank {other_rank} as SENDER")
    dist_socket = create_tensor_socket(
        device,
        other_rank,
        socket_config,
        socket_type="MPI",
        endpoint_type="SENDER",
    )
    logger.info(f"Rank {rank}: Distributed socket created")

    # Barrier before transfer
    ttnn.distributed_context_barrier()

    # DeepSeek V3 KV cache parameters
    # From: https://github.com/tenstorrent/tt-metal/pull/36146
    NUM_LAYERS = 61
    KVPE_DIM = 576
    BLOCK_SIZE = 32

    def get_kv_cache_shape(seq_len: int) -> tuple[int, int, int, int]:
        """Get single-layer KV cache shape for DeepSeek V3."""
        num_blocks = seq_len // BLOCK_SIZE
        return (num_blocks, 1, BLOCK_SIZE, KVPE_DIM)

    # Sequence lengths for each mode
    # Single tensor mode: 163840 excluded - too large for host memory (~5.4 GB packed)
    # Layer-by-layer mode: all seq_lengths including 163840
    seq_lengths_single = [1024, 4096, 8192, 32768]
    seq_lengths_lbl = [1024, 4096, 8192, 32768, 163840]

    # Print DeepSeek V3 tensor sizes
    logger.info(f"Rank {rank}: === DeepSeek V3 KV Cache Tensor Sizes ===")
    logger.info(f"Rank {rank}: Parameters: {NUM_LAYERS} layers, kvpe_dim={KVPE_DIM}, block_size={BLOCK_SIZE}")
    for seq_len in seq_lengths_lbl:
        shape = get_kv_cache_shape(seq_len)
        elements = shape[0] * shape[1] * shape[2] * shape[3]
        per_layer_mb = elements * 1 / (1024 * 1024)  # bfloat8_b = 1 byte
        total_mb = per_layer_mb * NUM_LAYERS
        logger.info(
            f"Rank {rank}:   seq_len={seq_len:>6}: shape={shape}, "
            f"per_layer={per_layer_mb:.2f} MB, total_61_layers={total_mb:.2f} MB"
        )
    logger.info(f"Rank {rank}: ================================================")

    test_value = 42.0

    # === WARMUP ===
    warmup_shape = get_kv_cache_shape(1024)  # Use smallest DeepSeek shape
    warmup_elements = warmup_shape[0] * warmup_shape[1] * warmup_shape[2] * warmup_shape[3]
    warmup_size_bytes = warmup_elements * 1  # bfloat8_b = 1 byte
    warmup_size_mb = warmup_size_bytes / (1024 * 1024)
    logger.info(f"Rank {rank}: === WARMUP send ({warmup_size_mb:.2f} MB) ===")

    ttnn.distributed_context_barrier()

    warmup_torch = torch.ones(warmup_shape, dtype=torch.bfloat16) * test_value
    warmup_tt = ttnn.from_torch(
        warmup_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
    )

    # Barrier to sync before send/recv
    ttnn.distributed_context_barrier()

    # Pre-sync to ensure no pending operations affect timing
    ttnn.synchronize_device(device)
    warmup_start = time.perf_counter()
    dist_socket.send(warmup_tt)
    # Use event-based sync for more precise timing (only waits for this specific operation)
    send_complete = ttnn.record_event(device, cq_id=0)
    ttnn.event_synchronize(send_complete)
    warmup_time = time.perf_counter() - warmup_start

    timings["warmup_send"] = warmup_time
    logger.info(
        f"Rank {rank}: WARMUP send complete! "
        f"Time: {warmup_time * 1000:.2f} ms, "
        f"Throughput: {format_throughput(warmup_size_bytes, warmup_time)}"
    )

    # Barrier to wait for receiver to complete
    ttnn.distributed_context_barrier()

    logger.info(f"Rank {rank}: === END WARMUP ===")

    # =========================================================================
    # VARIANT 1: Single Tensor Mode (all 61 layers packed into one tensor)
    # =========================================================================
    logger.info(f"Rank {rank}: ")
    logger.info(f"Rank {rank}: {'=' * 60}")
    logger.info(f"Rank {rank}: === VARIANT 1: SINGLE TENSOR MODE (61 layers packed) ===")
    logger.info(f"Rank {rank}: {'=' * 60}")

    for seq_len in seq_lengths_single:
        shape = get_kv_cache_shape(seq_len)
        # Pack all 61 layers into one tensor: (num_blocks * 61, 1, block_size, kvpe_dim)
        total_shape = (shape[0] * NUM_LAYERS, shape[1], shape[2], shape[3])
        tensor_elements = total_shape[0] * total_shape[1] * total_shape[2] * total_shape[3]
        tensor_size_bytes = tensor_elements * 1  # bfloat8_b = 1 byte
        tensor_size_mb = tensor_size_bytes / (1024 * 1024)
        shape_str = f"seq{seq_len}"

        ttnn.distributed_context_barrier()

        logger.info(
            f"Rank {rank}: Creating tensor {shape_str} - {NUM_LAYERS} layers packed "
            f"({tensor_size_mb:.2f} MB, shape={total_shape})"
        )
        src_torch = torch.ones(total_shape, dtype=torch.bfloat16) * test_value
        src_tt = ttnn.from_torch(
            src_torch,
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
        )

        logger.info(f"Rank {rank}: Ready to send tensor {shape_str}...")

        # Barrier to sync before E2E timing starts
        ttnn.distributed_context_barrier()

        # Pre-sync to ensure no pending operations affect timing
        ttnn.synchronize_device(device)
        # Send timing: measures just the send() operation
        send_start = time.perf_counter()
        dist_socket.send(src_tt)
        # Use event-based sync for more precise timing
        send_complete = ttnn.record_event(device, cq_id=0)
        ttnn.event_synchronize(send_complete)
        send_time = time.perf_counter() - send_start

        timings[f"single_send_{shape_str}"] = send_time
        throughput = format_throughput(tensor_size_bytes, send_time)
        logger.info(
            f"Rank {rank}: Tensor {shape_str} sent! "
            f"Time: {send_time * 1000:.2f} ms, "
            f"Throughput: {throughput}"
        )

        # Barrier to wait for receiver to complete
        ttnn.distributed_context_barrier()

    # =========================================================================
    # VARIANT 2: Layer-by-Layer Mode (61 individual transfers)
    # =========================================================================
    logger.info(f"Rank {rank}: ")
    logger.info(f"Rank {rank}: {'=' * 60}")
    logger.info(f"Rank {rank}: === VARIANT 2: LAYER-BY-LAYER MODE (61 transfers) ===")
    logger.info(f"Rank {rank}: {'=' * 60}")

    for seq_len in seq_lengths_lbl:
        shape = get_kv_cache_shape(seq_len)
        tensor_elements = shape[0] * shape[1] * shape[2] * shape[3]
        tensor_size_bytes = tensor_elements * 1  # bfloat8_b = 1 byte per element
        tensor_size_mb = tensor_size_bytes / (1024 * 1024)
        total_size_mb = tensor_size_mb * NUM_LAYERS
        shape_str = f"seq{seq_len}"

        ttnn.distributed_context_barrier()

        logger.info(
            f"Rank {rank}: Starting {shape_str} - {NUM_LAYERS} layers x {tensor_size_mb:.2f} MB = {total_size_mb:.2f} MB total"
        )

        # Create the per-layer tensor once (reuse for all 61 sends)
        src_torch = torch.ones(shape, dtype=torch.bfloat16) * test_value
        src_tt = ttnn.from_torch(
            src_torch,
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
        )

        # Barrier to sync before starting layer transfers
        ttnn.distributed_context_barrier()

        # Pre-sync to ensure no pending operations affect timing
        ttnn.synchronize_device(device)
        # Send 61 layers one by one
        layer_times = []
        total_start = time.perf_counter()
        for layer_idx in range(NUM_LAYERS):
            layer_start = time.perf_counter()
            dist_socket.send(src_tt)
            # Use event-based sync for more precise timing
            send_complete = ttnn.record_event(device, cq_id=0)
            ttnn.event_synchronize(send_complete)
            layer_time = time.perf_counter() - layer_start
            layer_times.append(layer_time)

        total_time = time.perf_counter() - total_start

        avg_layer_time = sum(layer_times) / len(layer_times)
        total_bytes = tensor_size_bytes * NUM_LAYERS

        timings[f"lbl_total_send_{shape_str}"] = total_time
        timings[f"lbl_avg_layer_send_{shape_str}"] = avg_layer_time

        logger.info(
            f"Rank {rank}: {shape_str} sent {NUM_LAYERS} layers! "
            f"Total: {total_time * 1000:.2f} ms ({format_throughput(total_bytes, total_time)}), "
            f"Avg/layer: {avg_layer_time * 1000:.2f} ms ({format_throughput(tensor_size_bytes, avg_layer_time)})"
        )

        # Barrier to wait for receiver to complete
        ttnn.distributed_context_barrier()

    del dist_socket
    return timings


def run_receiver(device: Any, socket_config: Any, rank: int) -> dict[str, float]:
    """Run the receiver side of the test."""
    timings: dict[str, float] = {}

    # Create socket to sender (rank 0)
    other_rank = 0
    logger.info(f"Rank {rank}: Creating distributed socket to rank {other_rank} as RECEIVER")
    dist_socket = create_tensor_socket(
        device,
        other_rank,
        socket_config,
        socket_type="MPI",
        endpoint_type="RECEIVER",
    )
    logger.info(f"Rank {rank}: Distributed socket created")

    # Barrier before transfer
    ttnn.distributed_context_barrier()

    # DeepSeek V3 KV cache parameters (must match sender)
    NUM_LAYERS = 61
    KVPE_DIM = 576
    BLOCK_SIZE = 32

    def get_kv_cache_shape(seq_len: int) -> tuple[int, int, int, int]:
        """Get single-layer KV cache shape for DeepSeek V3."""
        num_blocks = seq_len // BLOCK_SIZE
        return (num_blocks, 1, BLOCK_SIZE, KVPE_DIM)

    # Sequence lengths for each mode (must match sender)
    seq_lengths_single = [1024, 4096, 8192, 32768]
    seq_lengths_lbl = [1024, 4096, 8192, 32768, 163840]

    test_value = 42.0
    verification_passed = True

    # === WARMUP ===
    warmup_shape = get_kv_cache_shape(1024)  # Use smallest DeepSeek shape
    warmup_elements = warmup_shape[0] * warmup_shape[1] * warmup_shape[2] * warmup_shape[3]
    warmup_size_bytes = warmup_elements * 1  # bfloat8_b = 1 byte
    warmup_size_mb = warmup_size_bytes / (1024 * 1024)
    logger.info(f"Rank {rank}: === WARMUP recv ({warmup_size_mb:.2f} MB) ===")

    ttnn.distributed_context_barrier()

    warmup_tt = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(list(warmup_shape), ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT),
        device,
    )

    # Barrier to sync before E2E timing starts
    ttnn.distributed_context_barrier()

    # Pre-sync to ensure no pending operations affect timing
    ttnn.synchronize_device(device)
    # E2E timing: from sync point (≈ when sender starts send) to recv completion
    warmup_e2e_start = time.perf_counter()

    # Recv timing: just the recv() operation
    warmup_recv_start = time.perf_counter()
    dist_socket.recv(warmup_tt)
    # Use event-based sync for more precise timing
    recv_complete = ttnn.record_event(device, cq_id=0)
    ttnn.event_synchronize(recv_complete)
    warmup_recv_time = time.perf_counter() - warmup_recv_start

    warmup_e2e_time = time.perf_counter() - warmup_e2e_start

    timings["warmup_recv"] = warmup_recv_time
    timings["warmup_e2e"] = warmup_e2e_time
    logger.info(
        f"Rank {rank}: WARMUP recv complete! "
        f"Recv: {warmup_recv_time * 1000:.2f} ms ({format_throughput(warmup_size_bytes, warmup_recv_time)}), "
        f"E2E: {warmup_e2e_time * 1000:.2f} ms ({format_throughput(warmup_size_bytes, warmup_e2e_time)})"
    )

    # Barrier to sync with sender
    ttnn.distributed_context_barrier()

    logger.info(f"Rank {rank}: === END WARMUP ===")

    # =========================================================================
    # VARIANT 1: Single Tensor Mode (all 61 layers packed into one tensor)
    # =========================================================================
    logger.info(f"Rank {rank}: ")
    logger.info(f"Rank {rank}: {'=' * 60}")
    logger.info(f"Rank {rank}: === VARIANT 1: SINGLE TENSOR MODE (61 layers packed) ===")
    logger.info(f"Rank {rank}: {'=' * 60}")

    for seq_len in seq_lengths_single:
        shape = get_kv_cache_shape(seq_len)
        # Pack all 61 layers into one tensor: (num_blocks * 61, 1, block_size, kvpe_dim)
        total_shape = (shape[0] * NUM_LAYERS, shape[1], shape[2], shape[3])
        tensor_elements = total_shape[0] * total_shape[1] * total_shape[2] * total_shape[3]
        tensor_size_bytes = tensor_elements * 1  # bfloat8_b = 1 byte
        tensor_size_mb = tensor_size_bytes / (1024 * 1024)
        shape_str = f"seq{seq_len}"

        ttnn.distributed_context_barrier()

        logger.info(
            f"Rank {rank}: Allocating receive buffer {shape_str} - {NUM_LAYERS} layers packed "
            f"({tensor_size_mb:.2f} MB, shape={total_shape})"
        )
        dst_tt = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(list(total_shape), ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT),
            device,
        )

        logger.info(f"Rank {rank}: Ready to receive tensor {shape_str}...")

        # Barrier to sync before E2E timing starts
        ttnn.distributed_context_barrier()

        # Pre-sync to ensure no pending operations affect timing
        ttnn.synchronize_device(device)
        # E2E timing: from sync point (≈ when sender starts send) to recv completion
        e2e_start = time.perf_counter()

        # Recv timing: just the recv() operation
        recv_start = time.perf_counter()
        dist_socket.recv(dst_tt)
        # Use event-based sync for more precise timing
        recv_complete = ttnn.record_event(device, cq_id=0)
        ttnn.event_synchronize(recv_complete)
        recv_time = time.perf_counter() - recv_start

        e2e_time = time.perf_counter() - e2e_start

        timings[f"single_recv_{shape_str}"] = recv_time
        timings[f"single_e2e_{shape_str}"] = e2e_time

        recv_throughput = format_throughput(tensor_size_bytes, recv_time)
        e2e_throughput = format_throughput(tensor_size_bytes, e2e_time)
        logger.info(
            f"Rank {rank}: Tensor {shape_str} received! "
            f"Recv: {recv_time * 1000:.2f} ms ({recv_throughput}), "
            f"E2E: {e2e_time * 1000:.2f} ms ({e2e_throughput})"
        )

        # Verify the smallest tensor (seq_len=1024) - AFTER E2E timing
        if seq_len == seq_lengths_single[0]:
            dst_torch = ttnn.to_torch(
                ttnn.from_device(dst_tt),
                mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1),
            )
            # Trim if padded
            if dst_torch.shape[-1] > total_shape[-1]:
                dst_torch = dst_torch[..., : total_shape[-1]]

            logger.info(f"Rank {rank}: Received tensor shape: {dst_torch.shape}")
            logger.info(f"Rank {rank}: Received values (first 4): {dst_torch[0, 0, 0, :4].tolist()}")

            expected = torch.ones(total_shape, dtype=torch.bfloat16) * test_value
            if torch.allclose(dst_torch.float(), expected.float(), rtol=1e-2):
                logger.info(f"Rank {rank}: ✓ Single tensor verification PASSED!")
            else:
                logger.error(
                    f"Rank {rank}: ✗ Tensor mismatch! Expected {test_value}, "
                    f"got {dst_torch[0, 0, 0, 0].item()}"
                )
                verification_passed = False

        # Barrier to sync with sender
        ttnn.distributed_context_barrier()

    # =========================================================================
    # VARIANT 2: Layer-by-Layer Mode (61 individual transfers)
    # =========================================================================
    logger.info(f"Rank {rank}: ")
    logger.info(f"Rank {rank}: {'=' * 60}")
    logger.info(f"Rank {rank}: === VARIANT 2: LAYER-BY-LAYER MODE (61 transfers) ===")
    logger.info(f"Rank {rank}: {'=' * 60}")

    for seq_len in seq_lengths_lbl:
        shape = get_kv_cache_shape(seq_len)
        tensor_elements = shape[0] * shape[1] * shape[2] * shape[3]
        tensor_size_bytes = tensor_elements * 1  # bfloat8_b = 1 byte per element
        tensor_size_mb = tensor_size_bytes / (1024 * 1024)
        total_size_mb = tensor_size_mb * NUM_LAYERS
        shape_str = f"seq{seq_len}"

        ttnn.distributed_context_barrier()

        logger.info(
            f"Rank {rank}: Starting {shape_str} - {NUM_LAYERS} layers x {tensor_size_mb:.2f} MB = {total_size_mb:.2f} MB total"
        )

        # Barrier to sync before starting layer transfers
        ttnn.distributed_context_barrier()

        # Pre-sync to ensure no pending operations affect timing
        ttnn.synchronize_device(device)
        # Receive 61 layers one by one
        layer_times = []
        total_start = time.perf_counter()
        for layer_idx in range(NUM_LAYERS):
            # Allocate buffer for each layer
            dst_tt = ttnn.allocate_tensor_on_device(
                ttnn.TensorSpec(list(shape), ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT),
                device,
            )

            layer_start = time.perf_counter()
            dist_socket.recv(dst_tt)
            # Use event-based sync for more precise timing
            recv_complete = ttnn.record_event(device, cq_id=0)
            ttnn.event_synchronize(recv_complete)
            layer_time = time.perf_counter() - layer_start
            layer_times.append(layer_time)

        total_time = time.perf_counter() - total_start

        avg_layer_time = sum(layer_times) / len(layer_times)
        total_bytes = tensor_size_bytes * NUM_LAYERS

        timings[f"lbl_total_recv_{shape_str}"] = total_time
        timings[f"lbl_avg_layer_recv_{shape_str}"] = avg_layer_time

        logger.info(
            f"Rank {rank}: {shape_str} received {NUM_LAYERS} layers! "
            f"Total: {total_time * 1000:.2f} ms ({format_throughput(total_bytes, total_time)}), "
            f"Avg/layer: {avg_layer_time * 1000:.2f} ms ({format_throughput(tensor_size_bytes, avg_layer_time)})"
        )

        # Barrier to sync with sender
        ttnn.distributed_context_barrier()

    del dist_socket
    timings["verification_passed"] = 1.0 if verification_passed else 0.0
    return timings


def print_benchmark_summary(timings: dict[str, float]) -> None:
    """Print a formatted benchmark summary table."""
    # DeepSeek V3 parameters
    NUM_LAYERS = 61
    KVPE_DIM = 576
    BLOCK_SIZE = 32

    def get_total_mb(seq_len: int) -> float:
        num_blocks = seq_len // BLOCK_SIZE
        elements = num_blocks * 1 * BLOCK_SIZE * KVPE_DIM
        per_layer_mb = elements / (1024 * 1024)  # bfloat8_b = 1 byte
        return per_layer_mb * NUM_LAYERS

    # Sequence lengths
    seq_lengths_single = [1024, 4096, 8192, 32768]
    seq_lengths_lbl = [1024, 4096, 8192, 32768, 163840]

    # Build data for table
    table_data = []
    for seq_len in seq_lengths_lbl:
        total_mb = get_total_mb(seq_len)

        # Single tensor data (only for seq_lengths that were tested)
        if seq_len in seq_lengths_single:
            single_e2e_ms = timings.get(f"single_e2e_seq{seq_len}", 0) * 1000
            single_gbs = (total_mb / 1024) / (single_e2e_ms / 1000) if single_e2e_ms > 0 else 0
        else:
            single_e2e_ms = None
            single_gbs = None

        # Layer-by-layer data
        lbl_total_ms = timings.get(f"lbl_total_recv_seq{seq_len}", 0) * 1000
        lbl_avg_layer_ms = timings.get(f"lbl_avg_layer_recv_seq{seq_len}", 0) * 1000
        lbl_total_gbs = (total_mb / 1024) / (lbl_total_ms / 1000) if lbl_total_ms > 0 else 0
        per_layer_mb = total_mb / NUM_LAYERS
        lbl_layer_gbs = (per_layer_mb / 1024) / (lbl_avg_layer_ms / 1000) if lbl_avg_layer_ms > 0 else 0

        table_data.append({
            "seq_len": seq_len,
            "total_mb": total_mb,
            "single_ms": single_e2e_ms,
            "lbl_total_ms": lbl_total_ms,
            "lbl_layer_ms": lbl_avg_layer_ms,
            "single_gbs": single_gbs,
            "lbl_total_gbs": lbl_total_gbs,
            "lbl_layer_gbs": lbl_layer_gbs,
        })

    # Print table
    print("\n")
    print("=" * 130)
    print("DEEPSEEK V3 KV CACHE TRANSFER BENCHMARK (61 layers, kvpe_dim=576, bfloat8_b)")
    print("=" * 130)
    print(
        f"{'Seq Len':<10} {'Total MB':<10} {'Single(ms)':<12} {'LBL Total(ms)':<14} {'LBL/Layer(ms)':<14} "
        f"{'Single GB/s':<12} {'LBL Tot GB/s':<13} {'LBL Lyr GB/s':<12}"
    )
    print("-" * 130)

    for row in table_data:
        seq_len = row["seq_len"]
        total_mb = row["total_mb"]
        single_ms = row["single_ms"]
        lbl_total_ms = row["lbl_total_ms"]
        lbl_layer_ms = row["lbl_layer_ms"]
        single_gbs = row["single_gbs"]
        lbl_total_gbs = row["lbl_total_gbs"]
        lbl_layer_gbs = row["lbl_layer_gbs"]

        if single_ms is not None:
            print(
                f"{seq_len:<10} {total_mb:<10.1f} {single_ms:<12.1f} {lbl_total_ms:<14.1f} {lbl_layer_ms:<14.2f} "
                f"{single_gbs:<12.2f} {lbl_total_gbs:<13.2f} {lbl_layer_gbs:<12.2f}"
            )
        else:
            print(
                f"{seq_len:<10} {total_mb:<10.1f} {'N/A':<12} {lbl_total_ms:<14.1f} {lbl_layer_ms:<14.2f} "
                f"{'N/A':<12} {lbl_total_gbs:<13.2f} {lbl_layer_gbs:<12.2f}"
            )

    print("=" * 130)
    print("")


def main():
    """Main entry point for multi-host distributed socket test."""
    test_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Multi-Host Distributed Tensor Socket Test")
    logger.info("=" * 60)

    # Check distributed context - try to initialize if not already done
    if not ttnn.distributed_context_is_initialized():
        logger.warning("Distributed context not initialized, attempting to initialize...")
        try:
            ttnn.init_distributed_context()
            logger.info("Distributed context initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize distributed context: {e}")
            logger.error("Run with tt-run: tt-run --rank-binding <yaml> python <script>")
            sys.exit(1)

    if not ttnn.distributed_context_is_initialized():
        logger.error("Distributed context still not initialized after init attempt!")
        sys.exit(1)

    rank = int(ttnn.distributed_context_get_rank())
    world_size = int(ttnn.distributed_context_get_size())
    hostname = socket.gethostname()

    logger.info(f"Process started - Rank: {rank}, World Size: {world_size}, Host: {hostname}")

    # Log detailed node information to verify multi-host setup
    log_node_info(rank)

    if world_size != 2:
        logger.error(f"This test requires exactly 2 processes, got {world_size}")
        sys.exit(1)

    # Even MPI sockets require fabric enabled for the underlying mesh socket infrastructure
    logger.info(f"Rank {rank}: Setting fabric config to FABRIC_2D")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    # Open mesh device - use single chip (1x1) to avoid N300 internal connectivity issues
    mesh_shape = ttnn.MeshShape(1, 1)
    visible_device_ids = ttnn.get_device_ids()
    logger.info(f"Rank {rank}: Visible device IDs: {visible_device_ids}")

    logger.info(f"Rank {rank}: Opening mesh device with shape {mesh_shape}")
    device = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        physical_device_ids=[visible_device_ids[0]],  # Use first device only
    )

    try:
        # Build socket config
        socket_config = build_socket_config(
            mesh_shape,
            sender_rank=0,
            receiver_rank=1,
        )

        # Run as sender or receiver based on rank
        if rank == 0:
            timings = run_sender(device, socket_config, rank)
        else:
            timings = run_receiver(device, socket_config, rank)

        ttnn.distributed_context_barrier()

        # Print summary
        total_time = time.perf_counter() - test_start

        # Print formatted benchmark table (only on receiver side - rank 1)
        if rank == 1:
            print_benchmark_summary(timings)

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Rank {rank} @ {hostname}: TEST SUMMARY")
        logger.info("=" * 70)

        # Warmup
        logger.info("--- WARMUP ---")
        for key, value in timings.items():
            if key.startswith("warmup"):
                logger.info(f"  {key}: {value * 1000:.2f} ms")

        # Single Tensor Mode results
        logger.info("")
        logger.info("--- VARIANT 1: SINGLE TENSOR MODE (61 layers packed) ---")
        for key, value in timings.items():
            if key.startswith("single_"):
                logger.info(f"  {key}: {value * 1000:.2f} ms")

        # Layer-by-Layer Mode results
        logger.info("")
        logger.info("--- VARIANT 2: LAYER-BY-LAYER MODE (61 transfers) ---")
        for key, value in timings.items():
            if key.startswith("lbl_"):
                logger.info(f"  {key}: {value * 1000:.2f} ms")

        logger.info("")
        logger.info(f"  Total test time: {total_time * 1000:.2f} ms")

        if rank == 1 and timings.get("verification_passed", 0) < 1:
            logger.error("TEST FAILED: Verification did not pass")
            sys.exit(1)

        logger.info("=" * 70)
        logger.info(f"Rank {rank} @ {hostname}: TEST PASSED")
        logger.info("=" * 70)

    finally:
        logger.info(f"Rank {rank}: Closing mesh device")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
