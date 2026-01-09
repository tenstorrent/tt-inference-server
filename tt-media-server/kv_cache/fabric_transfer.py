# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Cross-Fabric KV Cache Transfer using TTNN MeshSocket

Based on Aditya's multi-mesh example:
https://github.com/tenstorrent/tt-metal/blob/main/tests/ttnn/distributed/test_multi_mesh.py

This module provides direct device-to-device KV cache transfer via fabric sockets,
bypassing host memory entirely for optimal performance.
"""

import torch
from typing import List, Optional, Tuple
from kv_cache.kv_cache_storage import KVCache, KVCacheMetadata
from utils.logger import TTLogger

# TTNN is only available on Tenstorrent hardware
try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None


class TTNNFabricKVTransfer:
    """
    KV Cache transfer using TTNN MeshSocket for cross-fabric communication

    Based on Aditya's multi-mesh example:
    https://github.com/tenstorrent/tt-metal/blob/main/tests/ttnn/distributed/test_multi_mesh.py
    """

    def __init__(
        self,
        device,
        mesh_shape,
        sender_rank: int,
        receiver_rank: int,
        sender_logical_coord=None,
        recv_logical_coord=None,
        socket_buffer_size: int = 4 * 1024 * 1024,  # 4MB default for KV cache
        logger: Optional[TTLogger] = None,
    ):
        """
        Initialize fabric socket for KV cache transfer

        Args:
            device: TTNN MeshDevice
            mesh_shape: Shape of the mesh (e.g., MeshShape(4, 4))
            sender_rank: Rank of the sender process (typically 0 for prefill worker)
            receiver_rank: Rank of the receiver process (typically 1 for decode worker)
            sender_logical_coord: Logical core coordinate for sender socket (default: (0, 0))
            recv_logical_coord: Logical core coordinate for receiver socket (default: (0, 0))
            socket_buffer_size: Size of socket buffer in bytes (L1 memory)
        """
        if not TTNN_AVAILABLE:
            raise ImportError("TTNN is not available. Fabric transfer requires Tenstorrent hardware.")

        self.device = device
        self.mesh_shape = mesh_shape
        self.sender_rank = sender_rank
        self.receiver_rank = receiver_rank
        self.sender_logical_coord = sender_logical_coord or ttnn.CoreCoord(0, 0)
        self.recv_logical_coord = recv_logical_coord or ttnn.CoreCoord(0, 0)
        self.socket_buffer_size = socket_buffer_size
        self.logger = logger or TTLogger()

        # Setup fabric config for 2D routing between meshes
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
        self.logger.info("Fabric config set to FABRIC_2D for cross-fabric transfer")

        # Verify distributed context is initialized
        if not ttnn.distributed_context_is_initialized():
            raise ValueError("Distributed context not initialized. "
                           "Ensure you're running with tt-run and rank binding.")

        current_rank = int(ttnn.distributed_context_get_rank())
        total_ranks = int(ttnn.distributed_context_get_size())
        self.logger.info(f"Fabric transfer initialized: rank={current_rank}, total_ranks={total_ranks}, "
                        f"sender_rank={sender_rank}, receiver_rank={receiver_rank}")

        # Setup socket connections
        self.socket_config = self._setup_socket_config()
        self.socket = None

    def _setup_socket_config(self) -> ttnn.SocketConfig:
        """
        Setup socket connections between mesh cores

        Each physical device in sender mesh connects to corresponding
        device in receiver mesh via socket.
        """
        socket_connections = []

        # Create connections for each coordinate in the mesh
        for coord in ttnn.MeshCoordinateRange(self.mesh_shape):
            socket_connections.append(
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(coord, self.sender_logical_coord),
                    ttnn.MeshCoreCoord(coord, self.recv_logical_coord)
                )
            )

        # Setup socket memory config (L1 buffer)
        socket_mem_config = ttnn.SocketMemoryConfig(
            ttnn.BufferType.L1,
            self.socket_buffer_size
        )

        return ttnn.SocketConfig(
            socket_connections,
            socket_mem_config,
            self.sender_rank,
            self.receiver_rank
        )

    def _kv_cache_to_ttnn_tensors(
        self,
        kv_cache: KVCache,
        layer_idx: int
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Convert KV cache tensors to TTNN format for transfer

        Args:
            kv_cache: KV cache to convert
            layer_idx: Layer index to convert

        Returns:
            Tuple of (key_tensor, value_tensor) in TTNN format
        """
        key_torch = kv_cache.keys[layer_idx]
        value_torch = kv_cache.values[layer_idx]

        # Convert to TTNN tensors with appropriate layout and mesh mapping
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        key_ttnn = ttnn.from_torch(
            key_torch,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.device,
                dims=(2, 3),  # Shard across seq_len and head_dim
                mesh_shape=self.mesh_shape
            ),
        )

        value_ttnn = ttnn.from_torch(
            value_torch,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.device,
                dims=(2, 3),  # Shard across seq_len and head_dim
                mesh_shape=self.mesh_shape
            ),
        )

        return key_ttnn, value_ttnn

    def _ttnn_tensors_to_kv_cache(
        self,
        key_ttnn: ttnn.Tensor,
        value_ttnn: ttnn.Tensor,
        metadata: KVCacheMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert TTNN tensors back to PyTorch format

        Args:
            key_ttnn: Key tensor in TTNN format
            value_ttnn: Value tensor in TTNN format
            metadata: KV cache metadata

        Returns:
            Tuple of (key_torch, value_torch)
        """
        key_torch = ttnn.to_torch(
            ttnn.from_device(key_ttnn),
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.device,
                mesh_shape=self.mesh_shape,
                dims=(2, 3)
            ),
        )

        value_torch = ttnn.to_torch(
            ttnn.from_device(value_ttnn),
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.device,
                mesh_shape=self.mesh_shape,
                dims=(2, 3)
            ),
        )

        return key_torch, value_torch

    async def send_kv_cache(self, kv_cache: KVCache) -> bool:
        """
        Send KV cache to receiver via cross-fabric socket

        This method sends KV cache layer by layer directly device-to-device.
        No host memory is involved - transfer happens entirely on fabric.

        Args:
            kv_cache: KV cache to send

        Returns:
            True if successful, False otherwise
        """
        try:
            rank = int(ttnn.distributed_context_get_rank())

            if rank != self.sender_rank:
                self.logger.warning(
                    f"Not sender rank (current={rank}, expected={self.sender_rank}), skipping send"
                )
                return False

            self.logger.info(
                f"Sending KV cache via fabric: task_id={kv_cache.metadata.task_id}, "
                f"layers={kv_cache.metadata.num_layers}, seq_len={kv_cache.metadata.seq_len}"
            )

            # Create send socket
            self.socket = ttnn.MeshSocket(self.device, self.socket_config)

            # Send each layer's KV cache
            for layer_idx in range(kv_cache.metadata.num_layers):
                self.logger.debug(f"Sending layer {layer_idx}/{kv_cache.metadata.num_layers}")

                # Convert to TTNN format
                key_ttnn, value_ttnn = self._kv_cache_to_ttnn_tensors(
                    kv_cache, layer_idx
                )

                # Send key tensor asynchronously (direct device-to-device)
                ttnn.experimental.send_async(key_ttnn, self.socket)

                # Send value tensor asynchronously (direct device-to-device)
                ttnn.experimental.send_async(value_ttnn, self.socket)

            self.logger.info(
                f"KV cache sent successfully via fabric: task_id={kv_cache.metadata.task_id}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to send KV cache via fabric: {e}")
            return False

    async def receive_kv_cache(
        self,
        expected_metadata: KVCacheMetadata
    ) -> Optional[KVCache]:
        """
        Receive KV cache from sender via cross-fabric socket

        KV cache is received directly into device memory, bypassing host memory.

        Args:
            expected_metadata: Expected metadata (required to know structure)

        Returns:
            KVCache object if successful, None otherwise
        """
        try:
            rank = int(ttnn.distributed_context_get_rank())

            if rank != self.receiver_rank:
                self.logger.warning(
                    f"Not receiver rank (current={rank}, expected={self.receiver_rank}), skipping receive"
                )
                return None

            self.logger.info(
                f"Receiving KV cache via fabric: task_id={expected_metadata.task_id}, "
                f"layers={expected_metadata.num_layers}, seq_len={expected_metadata.seq_len}"
            )

            # Create receive socket
            self.socket = ttnn.MeshSocket(self.device, self.socket_config)

            num_layers = expected_metadata.num_layers
            keys = []
            values = []

            # Receive each layer's KV cache
            for layer_idx in range(num_layers):
                batch_size = expected_metadata.batch_size
                num_heads = expected_metadata.num_heads
                seq_len = expected_metadata.seq_len
                head_dim = expected_metadata.head_dim

                # Create spec for receiving tensor
                key_spec = ttnn.TensorSpec(
                    ttnn.Shape([batch_size, num_heads, seq_len, head_dim]),
                    ttnn.DataType.BFLOAT16,  # Adjust based on actual dtype
                    ttnn.TILE_LAYOUT
                )

                value_spec = ttnn.TensorSpec(
                    ttnn.Shape([batch_size, num_heads, seq_len, head_dim]),
                    ttnn.DataType.BFLOAT16,
                    ttnn.TILE_LAYOUT
                )

                # Allocate tensors on device
                key_ttnn = ttnn.allocate_tensor_on_device(
                    key_spec,
                    self.device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.device,
                        dims=(2, 3),
                        mesh_shape=self.mesh_shape
                    )
                )

                value_ttnn = ttnn.allocate_tensor_on_device(
                    value_spec,
                    self.device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.device,
                        dims=(2, 3),
                        mesh_shape=self.mesh_shape
                    )
                )

                # Receive asynchronously
                ttnn.experimental.recv_async(key_ttnn, self.socket)
                ttnn.experimental.recv_async(value_ttnn, self.socket)

                # Convert back to PyTorch
                key_torch, value_torch = self._ttnn_tensors_to_kv_cache(
                    key_ttnn, value_ttnn, expected_metadata
                )

                keys.append(key_torch)
                values.append(value_torch)

            # Create KV cache object
            kv_cache = KVCache(
                keys=keys,
                values=values,
                metadata=expected_metadata
            )

            self.logger.info(
                f"KV cache received successfully via fabric: task_id={expected_metadata.task_id}"
            )
            return kv_cache

        except Exception as e:
            self.logger.error(f"Failed to receive KV cache via fabric: {e}")
            return None

    def close(self):
        """Close socket and cleanup"""
        if self.socket is not None:
            self.socket = None

