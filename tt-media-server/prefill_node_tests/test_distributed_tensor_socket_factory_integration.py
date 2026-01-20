# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Integration test for DistributedTensorSocketFactory using ttnn.create_distributed_socket.

This tests our custom wrapper in prefill_node/distributed_tensor_socket_factory.py
which uses:
- ttnn.create_distributed_socket
- ttnn.DistributedSocketType
- ttnn.DistributedEndpointSocketType
- ttnn.DistributedISocket (with send/recv methods)

Run with:
    # First reset devices
    tt-smi -r

    # Rebuild tt-metal (if C++ bindings changed)
    cd /localdev/ztorlak/tt-metal && ./build_metal.sh

    # Run test
    cd /localdev/ztorlak/tt-metal && \\
    source python_env/bin/activate && \\
    export TT_METAL_HOME=/localdev/ztorlak/tt-metal && \\
    export TT_METAL_RUNTIME_ROOT=/localdev/ztorlak/tt-metal && \\
    export PYTHONPATH=/localdev/ztorlak/tt-metal/ttnn:$PYTHONPATH && \\
    tt-run --rank-binding /localdev/ztorlak/tt-inference-server/tt-media-server/prefill_node_tests/dual_n300_rank_bindings.yaml \\
        python -m pytest /localdev/ztorlak/tt-inference-server/tt-media-server/prefill_node_tests/test_distributed_tensor_socket_factory_integration.py \\
        -vv -s --timeout=60
"""

import time

import pytest
import torch
import ttnn
from loguru import logger
from prefill_node.distributed_tensor_socket_factory import (
    DistributedTensorSocketConfig,
    DistributedTensorSocketFactory,
    build_socket_config,
)


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_throughput(size_bytes: int, time_sec: float) -> str:
    """Format throughput to human readable string."""
    if time_sec == 0:
        return "N/A"
    throughput = size_bytes / time_sec
    if throughput < 1024:
        return f"{throughput:.2f} B/s"
    elif throughput < 1024 * 1024:
        return f"{throughput / 1024:.2f} KB/s"
    elif throughput < 1024 * 1024 * 1024:
        return f"{throughput / (1024 * 1024):.2f} MB/s"
    else:
        return f"{throughput / (1024 * 1024 * 1024):.2f} GB/s"


@pytest.mark.timeout(60)
def test_distributed_tensor_socket_factory_with_create_distributed_socket() -> None:
    """
    Test that DistributedTensorSocketFactory can create distributed sockets and transfer tensors.

    This tests:
    1. DistributedTensorSocketFactory.start() - verifies bindings exist
    2. build_socket_config() - helper to create SocketConfig
    3. DistributedTensorSocketFactory.create_tensor_socket() - creates a DistributedISocket
    4. DistributedISocket.send()/recv() - actual tensor transfer between ranks

    The test sends tensors of various sizes from rank 0 (prefill) to rank 1 (decode)
    and measures throughput.
    """
    timings: dict[str, float] = {}
    test_start = time.perf_counter()

    logger.info("=== Starting DistributedTensorSocketFactory integration test ===")

    # Initialize TT-Fabric for inter-device communication
    logger.info("Setting fabric config to FABRIC_2D")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    # Each process gets one N300 (1x2 mesh, 2 chips per board)
    mesh_shape = ttnn.MeshShape(1, 2)

    visible_device_ids = ttnn.get_device_ids()
    logger.info(f"Visible device IDs: {visible_device_ids}")

    if len(visible_device_ids) != 2:
        pytest.skip(
            f"Expected exactly 2 devices for N300 (1x2 mesh), got {len(visible_device_ids)}."
        )

    # Time device initialization
    device_init_start = time.perf_counter()
    physical_device_ids = visible_device_ids
    logger.info(
        f"Opening mesh device with shape {mesh_shape} on devices {physical_device_ids}"
    )
    device = ttnn.open_mesh_device(
        mesh_shape=mesh_shape, physical_device_ids=physical_device_ids
    )
    timings["device_init"] = time.perf_counter() - device_init_start
    logger.info(
        f"[TIMING] Device initialization: {timings['device_init'] * 1000:.2f} ms"
    )

    try:
        if not ttnn.distributed_context_is_initialized():
            pytest.fail("Distributed context not initialized. Run with tt-run.")

        world_size = int(ttnn.distributed_context_get_size())
        if world_size != 2:
            pytest.skip(f"This test requires exactly 2 processes, got {world_size}")

        rank = int(ttnn.distributed_context_get_rank())
        logger.info(f"Process rank={rank} started on N300 device")

        # === Test DistributedTensorSocketFactory ===
        # Configure as SENDER for rank 0, RECEIVER for rank 1
        endpoint_type = "SENDER" if rank == 0 else "RECEIVER"
        config = DistributedTensorSocketConfig(
            socket_type="MPI",
            endpoint_socket_type=endpoint_type,
        )
        logger.info(
            f"Rank {rank}: Creating DistributedTensorSocketFactory with {endpoint_type}"
        )

        socket_factory = DistributedTensorSocketFactory(config)

        # Test 1: Verify start() finds the bindings
        socket_factory_start_time = time.perf_counter()
        logger.info(f"Rank {rank}: Calling socket_factory.start()...")
        try:
            socket_factory.start()
            timings["socket_factory_start"] = (
                time.perf_counter() - socket_factory_start_time
            )
            logger.info(
                f"Rank {rank}: socket_factory.start() succeeded - bindings found! "
                f"({timings['socket_factory_start'] * 1000:.2f} ms)"
            )
        except RuntimeError as e:
            pytest.fail(f"DistributedTensorSocketFactory.start() failed: {e}")

        # Test 2: Verify the distributed socket enums exist
        logger.info(f"Rank {rank}: Checking distributed socket enums...")
        assert hasattr(ttnn, "DistributedSocketType"), (
            "ttnn.DistributedSocketType not found"
        )
        assert hasattr(ttnn, "DistributedEndpointSocketType"), (
            "ttnn.DistributedEndpointSocketType not found"
        )
        assert hasattr(ttnn, "create_distributed_socket"), (
            "ttnn.create_distributed_socket not found"
        )
        logger.info(f"Rank {rank}: All distributed socket bindings present!")

        # Test 3: Build socket config using the helper function
        socket_config = build_socket_config(
            mesh_shape,
            sender_rank=0,
            receiver_rank=1,
        )

        other_rank = 1 if rank == 0 else 0
        logger.info(
            f"Rank {rank}: Creating distributed socket to other_rank={other_rank}..."
        )

        # Test 4: Create distributed socket via our socket factory (TIMED)
        socket_factory_create_start = time.perf_counter()
        dist_socket = socket_factory.create_tensor_socket(
            mesh_device=device,
            other_rank=other_rank,
            socket_config=socket_config,
        )
        timings["socket_factory_create"] = (
            time.perf_counter() - socket_factory_create_start
        )
        logger.info(
            f"Rank {rank}: Distributed socket created successfully! "
            f"({timings['socket_factory_create'] * 1000:.2f} ms)"
        )
        logger.info(f"Rank {rank}: Socket type: {type(dist_socket)}")

        # Test 5: Verify socket has send/recv methods
        assert hasattr(dist_socket, "send"), "Socket missing send() method"
        assert hasattr(dist_socket, "recv"), "Socket missing recv() method"
        logger.info(f"Rank {rank}: Socket has send/recv methods!")

        # Barrier before tensor transfer
        ttnn.distributed_context_barrier()

        # Test 6: Try to send/receive a tensor (TIMED)
        # Test with different tensor sizes
        test_shapes = [
            (1, 1, 32, 32),  # Small: 2 KB (bfloat16)
            (1, 1, 128, 128),  # Medium: 32 KB
            (1, 1, 512, 512),  # Large: 512 KB
            (1, 1, 1024, 1024),  # XLarge: 2 MB
        ]

        for shape in test_shapes:
            # Calculate tensor size in bytes (bfloat16 = 2 bytes per element)
            tensor_elements = shape[0] * shape[1] * shape[2] * shape[3]
            tensor_size_bytes = tensor_elements * 2  # bfloat16

            test_value = 99.0
            shape_str = f"{shape[2]}x{shape[3]}"

            # Barrier to sync before each transfer
            ttnn.distributed_context_barrier()

            if rank == 0:
                # Sender: Create tensor and send
                tensor_create_start = time.perf_counter()
                logger.info(
                    f"Rank 0: Creating tensor {shape_str} "
                    f"({format_size(tensor_size_bytes)})"
                )
                src_torch = torch.ones(shape, dtype=torch.bfloat16) * test_value
                src_tt = ttnn.from_torch(
                    src_torch,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                tensor_create_time = time.perf_counter() - tensor_create_start
                logger.info(
                    f"Rank 0: Tensor created ({tensor_create_time * 1000:.2f} ms)"
                )

                # Barrier to sync tensor creation
                ttnn.distributed_context_barrier()

                # Time the actual send
                send_start = time.perf_counter()
                logger.info(f"Rank 0: Sending tensor {shape_str}...")
                dist_socket.send(src_tt)
                ttnn.synchronize_device(device)
                send_time = time.perf_counter() - send_start

                timings[f"send_{shape_str}"] = send_time
                throughput = format_throughput(tensor_size_bytes, send_time)
                logger.info(
                    f"Rank 0: Tensor {shape_str} sent! "
                    f"Time: {send_time * 1000:.2f} ms, "
                    f"Size: {format_size(tensor_size_bytes)}, "
                    f"Throughput: {throughput}"
                )

                # Barrier after send
                ttnn.distributed_context_barrier()

            else:
                # Receiver: Allocate buffer and receive
                alloc_start = time.perf_counter()
                logger.info(
                    f"Rank 1: Allocating receive buffer {shape_str} "
                    f"({format_size(tensor_size_bytes)})"
                )
                dst_tt = ttnn.allocate_tensor_on_device(
                    ttnn.TensorSpec(
                        list(shape), ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT
                    ),
                    device,
                )
                alloc_time = time.perf_counter() - alloc_start
                logger.info(f"Rank 1: Buffer allocated ({alloc_time * 1000:.2f} ms)")

                # Barrier to sync tensor creation on sender
                ttnn.distributed_context_barrier()

                # Time the actual receive
                recv_start = time.perf_counter()
                logger.info(f"Rank 1: Receiving tensor {shape_str}...")
                dist_socket.recv(dst_tt)
                ttnn.synchronize_device(device)
                recv_time = time.perf_counter() - recv_start

                timings[f"recv_{shape_str}"] = recv_time
                throughput = format_throughput(tensor_size_bytes, recv_time)
                logger.info(
                    f"Rank 1: Tensor {shape_str} received! "
                    f"Time: {recv_time * 1000:.2f} ms, "
                    f"Size: {format_size(tensor_size_bytes)}, "
                    f"Throughput: {throughput}"
                )

                # Barrier after recv
                ttnn.distributed_context_barrier()

                # Verify (only for smallest tensor to save time)
                if shape == test_shapes[0]:
                    dst_torch = ttnn.to_torch(
                        ttnn.from_device(dst_tt),
                        mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1),
                    )
                    if dst_torch.shape[-1] > shape[-1]:
                        dst_torch = dst_torch[..., : shape[-1]]

                    logger.info(f"Rank 1: Received tensor shape: {dst_torch.shape}")
                    logger.info(
                        f"Rank 1: Received values: {dst_torch[0, 0, 0, :4].tolist()}"
                    )

                    expected = torch.ones(shape, dtype=torch.bfloat16) * test_value
                    assert torch.allclose(
                        dst_torch.float(), expected.float(), rtol=1e-2
                    ), (
                        f"Tensor mismatch! Expected {test_value}, "
                        f"got {dst_torch[0, 0, 0, 0].item()}"
                    )
                    logger.info("Rank 1: Tensor verification PASSED!")

        # Cleanup
        del dist_socket

        ttnn.distributed_context_barrier()

        # Print timing summary
        total_time = time.perf_counter() - test_start
        timings["total"] = total_time

        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info(f"Rank {rank}: TIMING SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(
            f"  Device initialization:  {timings['device_init'] * 1000:>10.2f} ms"
        )
        logger.info(
            f"  Socket factory start:          {timings['socket_factory_start'] * 1000:>10.2f} ms"
        )
        logger.info(
            f"  Socket creation:        {timings['socket_factory_create'] * 1000:>10.2f} ms"
        )

        # Transfer times
        logger.info(f"  {'─' * 56}")
        logger.info("  Tensor Transfers:")
        for shape in test_shapes:
            shape_str = f"{shape[2]}x{shape[3]}"
            tensor_size = shape[0] * shape[1] * shape[2] * shape[3] * 2
            if rank == 0 and f"send_{shape_str}" in timings:
                t = timings[f"send_{shape_str}"]
                tp = format_throughput(tensor_size, t)
                logger.info(
                    f"    {shape_str:>10} send: {t * 1000:>8.2f} ms  "
                    f"({format_size(tensor_size):>8}, {tp})"
                )
            elif rank == 1 and f"recv_{shape_str}" in timings:
                t = timings[f"recv_{shape_str}"]
                tp = format_throughput(tensor_size, t)
                logger.info(
                    f"    {shape_str:>10} recv: {t * 1000:>8.2f} ms  "
                    f"({format_size(tensor_size):>8}, {tp})"
                )

        logger.info(f"  {'─' * 56}")
        logger.info(f"  Total test time:        {total_time * 1000:>10.2f} ms")
        logger.info(f"{'=' * 60}")

        logger.info(f"Rank {rank}: Test complete")

    finally:
        logger.info("Closing mesh device")
        ttnn.close_device(device)

    logger.info("=== DistributedTensorSocketFactory integration test PASSED ===")
