# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Distributed tensor socket factory for tensor transfer between prefill and decode nodes.

Wraps ttnn.create_distributed_socket to provide distributed tensor sockets
that can send/receive ttnn.Tensor objects between ranks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ttnn


@dataclass(frozen=True)
class DistributedTensorSocketConfig:
    """Configuration for distributed tensor sockets."""

    socket_type: str = "MPI"  # ttnn.DistributedSocketType enum name
    endpoint_socket_type: str = "SENDER"  # ttnn.DistributedEndpointSocketType enum name: SENDER | RECEIVER | BIDIRECTIONAL


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

    sender_coord = ttnn.CoreCoord(0, 0)
    recv_coord = ttnn.CoreCoord(0, 0)

    socket_connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        socket_connections.append(
            ttnn.SocketConnection(
                ttnn.MeshCoreCoord(coord, sender_coord),
                ttnn.MeshCoreCoord(coord, recv_coord),
            )
        )

    socket_mem_config = ttnn.SocketMemoryConfig(buffer_type, buffer_size)

    return ttnn.SocketConfig(
        socket_connections,
        socket_mem_config,
        sender_rank=sender_rank,
        receiver_rank=receiver_rank,
    )


class DistributedTensorSocketFactory:
    """
    Factory for creating distributed tensor sockets via ttnn.create_distributed_socket.

    Used for KV cache transfer in disaggregated prefill-decode inference.
    """

    def __init__(self, config: DistributedTensorSocketConfig):
        self._config = config
        self._create_distributed_socket: Any | None = None

    def start(self) -> None:
        """Initialize socket factory by verifying ttnn distributed socket bindings exist."""
        create_fn = getattr(ttnn, "create_distributed_socket", None)
        if create_fn is None:
            raise RuntimeError("ttnn.create_distributed_socket not found")
        self._create_distributed_socket = create_fn

    def create_tensor_socket(
        self,
        *,
        mesh_device: Any,
        other_rank: int,
        socket_config: Any,
    ) -> Any:
        """
        Create a distributed tensor socket between this rank and other_rank.

        Args:
            mesh_device: ttnn.MeshDevice
            other_rank: MPI rank to connect to
            socket_config: ttnn.SocketConfig (use build_socket_config())

        Returns:
            Socket with send(tensor) and recv(tensor) methods
        """
        if self._create_distributed_socket is None:
            raise RuntimeError("Socket factory not started. Call start() first.")

        socket_type_enum = getattr(ttnn, "DistributedSocketType", None)
        endpoint_enum = getattr(ttnn, "DistributedEndpointSocketType", None)
        if socket_type_enum is None or endpoint_enum is None:
            raise RuntimeError("ttnn distributed socket enums not found")

        try:
            socket_type = getattr(socket_type_enum, self._config.socket_type)
        except AttributeError as e:
            raise ValueError(f"Invalid socket_type: {self._config.socket_type}") from e

        try:
            endpoint_socket_type = getattr(
                endpoint_enum, self._config.endpoint_socket_type
            )
        except AttributeError as e:
            raise ValueError(
                f"Invalid endpoint_socket_type: {self._config.endpoint_socket_type}"
            ) from e

        return self._create_distributed_socket(
            socket_type,
            endpoint_socket_type,
            mesh_device,
            int(other_rank),
            socket_config,
        )

    def close(self) -> None:
        """Close the socket factory."""
        self._create_distributed_socket = None
