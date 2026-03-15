# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Utility functions for distributed tensor sockets.

Provides helpers for creating ttnn distributed sockets for tensor transfer
between prefill and decode nodes in disaggregated inference.
"""

from typing import Any

import ttnn


def build_socket_config(
    mesh_shape: Any,
    *,
    sender_rank: int = 0,
    receiver_rank: int = 1,
    buffer_size: int = 16384,
    buffer_type: Any = None,
) -> Any:
    """
    Build a ttnn.SocketConfig for distributed tensor transfer.

    Args:
        mesh_shape: ttnn.MeshShape (e.g., MeshShape(1, 2) for N300)
        sender_rank: MPI rank of sender (default: 0)
        receiver_rank: MPI rank of receiver (default: 1)
        buffer_size: L1 buffer size in bytes (default: 16384)
        buffer_type: ttnn.BufferType (default: L1)

    Returns:
        ttnn.SocketConfig for use with create_tensor_socket()
    """
    if buffer_type is None:
        buffer_type = ttnn.BufferType.L1

    socket_connections = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
            ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
        )
        for coord in ttnn.MeshCoordinateRange(mesh_shape)
    ]

    return ttnn.SocketConfig(
        socket_connections,
        ttnn.SocketMemoryConfig(buffer_type, buffer_size),
        sender_rank=sender_rank,
        receiver_rank=receiver_rank,
    )


def create_tensor_socket(
    mesh_device: Any,
    other_rank: int,
    socket_config: Any,
    *,
    socket_type: str = "MPI",
    endpoint_type: str = "SENDER",
) -> Any:
    """
    Create a distributed tensor socket between this rank and other_rank.

    Args:
        mesh_device: ttnn.MeshDevice
        other_rank: MPI rank to connect to
        socket_config: ttnn.SocketConfig (use build_socket_config())
        socket_type: "MPI" or "FABRIC" (default: "MPI")
        endpoint_type: "SENDER", "RECEIVER", or "BIDIRECTIONAL" (default: "SENDER")

    Returns:
        DistributedISocket with send(tensor) and recv(tensor) methods
    """
    socket_type_enum = getattr(ttnn.DistributedSocketType, socket_type)
    endpoint_type_enum = getattr(ttnn.DistributedEndpointSocketType, endpoint_type)

    return ttnn.create_distributed_socket(
        socket_type_enum,
        endpoint_type_enum,
        mesh_device,
        other_rank,
        socket_config,
    )
