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
   cd /data/ztorlak/tt-metal && source python_env/bin/activate
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

from prefill_node_tests.node_utils import format_size, format_throughput, log_node_info

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

    # Test tensor shapes
    test_shapes = [
        (1, 1, 32, 32),      # 2 KB
        (1, 1, 128, 128),    # 32 KB
        (1, 1, 512, 512),    # 512 KB
        (1, 1, 1024, 1024),  # 2 MB
    ]

    test_value = 42.0

    for shape in test_shapes:
        tensor_elements = shape[0] * shape[1] * shape[2] * shape[3]
        tensor_size_bytes = tensor_elements * 2  # bfloat16
        shape_str = f"{shape[2]}x{shape[3]}"

        ttnn.distributed_context_barrier()

        logger.info(f"Rank {rank}: Creating tensor {shape_str} ({format_size(tensor_size_bytes)})")
        src_torch = torch.ones(shape, dtype=torch.bfloat16) * test_value
        src_tt = ttnn.from_torch(
            src_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        ttnn.distributed_context_barrier()

        send_start = time.perf_counter()
        logger.info(f"Rank {rank}: Sending tensor {shape_str}...")
        dist_socket.send(src_tt)
        ttnn.synchronize_device(device)
        send_time = time.perf_counter() - send_start

        timings[f"send_{shape_str}"] = send_time
        throughput = format_throughput(tensor_size_bytes, send_time)
        logger.info(
            f"Rank {rank}: Tensor {shape_str} sent! "
            f"Time: {send_time * 1000:.2f} ms, "
            f"Throughput: {throughput}"
        )

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

    # Test tensor shapes (must match sender)
    test_shapes = [
        (1, 1, 32, 32),      # 2 KB
        (1, 1, 128, 128),    # 32 KB
        (1, 1, 512, 512),    # 512 KB
        (1, 1, 1024, 1024),  # 2 MB
    ]

    test_value = 42.0
    verification_passed = True

    for shape in test_shapes:
        tensor_elements = shape[0] * shape[1] * shape[2] * shape[3]
        tensor_size_bytes = tensor_elements * 2  # bfloat16
        shape_str = f"{shape[2]}x{shape[3]}"

        ttnn.distributed_context_barrier()

        logger.info(f"Rank {rank}: Allocating receive buffer {shape_str}")
        dst_tt = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(list(shape), ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT),
            device,
        )

        ttnn.distributed_context_barrier()

        recv_start = time.perf_counter()
        logger.info(f"Rank {rank}: Receiving tensor {shape_str}...")
        dist_socket.recv(dst_tt)
        ttnn.synchronize_device(device)
        recv_time = time.perf_counter() - recv_start

        timings[f"recv_{shape_str}"] = recv_time
        throughput = format_throughput(tensor_size_bytes, recv_time)
        logger.info(
            f"Rank {rank}: Tensor {shape_str} received! "
            f"Time: {recv_time * 1000:.2f} ms, "
            f"Throughput: {throughput}"
        )

        # Verify the smallest tensor
        if shape == test_shapes[0]:
            dst_torch = ttnn.to_torch(
                ttnn.from_device(dst_tt),
                mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1),
            )
            if dst_torch.shape[-1] > shape[-1]:
                dst_torch = dst_torch[..., : shape[-1]]

            logger.info(f"Rank {rank}: Received values: {dst_torch[0, 0, 0, :4].tolist()}")

            expected = torch.ones(shape, dtype=torch.bfloat16) * test_value
            if torch.allclose(dst_torch.float(), expected.float(), rtol=1e-2):
                logger.info(f"Rank {rank}: ✓ Tensor verification PASSED!")
            else:
                logger.error(
                    f"Rank {rank}: ✗ Tensor mismatch! Expected {test_value}, "
                    f"got {dst_torch[0, 0, 0, 0].item()}"
                )
                verification_passed = False

        ttnn.distributed_context_barrier()

    del dist_socket
    timings["verification_passed"] = 1.0 if verification_passed else 0.0
    return timings


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
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Rank {rank} @ {hostname}: TEST SUMMARY")
        logger.info("=" * 60)
        for key, value in timings.items():
            if key != "verification_passed":
                logger.info(f"  {key}: {value * 1000:.2f} ms")
        logger.info(f"  Total time: {total_time * 1000:.2f} ms")

        if rank == 1 and timings.get("verification_passed", 0) < 1:
            logger.error("TEST FAILED: Verification did not pass")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info(f"Rank {rank} @ {hostname}: TEST PASSED")
        logger.info("=" * 60)

    finally:
        logger.info(f"Rank {rank}: Closing mesh device")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
