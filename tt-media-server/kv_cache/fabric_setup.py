# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Helper functions for setting up Fabric socket transfer

This module provides utilities to initialize fabric transfer from device runners
and worker configurations.
"""

import os
from typing import Optional

# TTNN is only available on Tenstorrent hardware
try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None

from kv_cache.fabric_transfer import TTNNFabricKVTransfer
from config.settings import settings
from utils.logger import TTLogger


def create_fabric_transfer_from_device_runner(
    device_runner,
    sender_rank: Optional[int] = None,
    receiver_rank: Optional[int] = None,
    logger: Optional[TTLogger] = None,
) -> Optional[TTNNFabricKVTransfer]:
    """
    Create fabric transfer instance from device runner

    Attempts to extract device and mesh_shape from device runner.

    Args:
        device_runner: Device runner instance (should have device attribute)
        sender_rank: Sender rank (defaults to env var or 0)
        receiver_rank: Receiver rank (defaults to env var or 1)
        logger: Logger instance

    Returns:
        TTNNFabricKVTransfer instance or None if setup fails
    """
    logger = logger or TTLogger()

    if not TTNN_AVAILABLE:
        logger.error("TTNN is not available. Cannot create fabric transfer.")
        return None

    try:
        # Get device from runner
        if not hasattr(device_runner, "device") or device_runner.device is None:
            logger.warning("Device runner does not have device attribute")
            return None

        device = device_runner.device

        # Get mesh shape from settings or device
        if hasattr(settings, "device_mesh_shape"):
            mesh_shape = ttnn.MeshShape(*settings.device_mesh_shape)
        else:
            # Try to infer from device
            mesh_shape = ttnn.MeshShape(4, 4)  # Default
            logger.warning(f"Using default mesh shape {mesh_shape}")

        # Get ranks from env or parameters
        if sender_rank is None:
            sender_rank = int(os.getenv("KV_CACHE__FABRIC_SENDER_RANK", "0"))
        if receiver_rank is None:
            receiver_rank = int(os.getenv("KV_CACHE__FABRIC_RECEIVER_RANK", "1"))

        # Get socket buffer size from settings
        socket_buffer_size = settings.kv_cache.fabric_socket_buffer_size

        # Create fabric transfer
        fabric_transfer = TTNNFabricKVTransfer(
            device=device,
            mesh_shape=mesh_shape,
            sender_rank=sender_rank,
            receiver_rank=receiver_rank,
            socket_buffer_size=socket_buffer_size,
            logger=logger,
        )

        logger.info(
            f"Fabric transfer created: sender_rank={sender_rank}, "
            f"receiver_rank={receiver_rank}, mesh_shape={mesh_shape}"
        )

        return fabric_transfer

    except Exception as e:
        logger.error(f"Failed to create fabric transfer from device runner: {e}")
        return None


def get_fabric_transfer_for_worker(
    worker_id: str,
    device_runner,
    is_prefill: bool,
    logger: Optional[TTLogger] = None,
) -> Optional[TTNNFabricKVTransfer]:
    """
    Get fabric transfer instance for a worker

    Determines sender/receiver rank based on worker type.

    Args:
        worker_id: Worker ID
        device_runner: Device runner instance
        is_prefill: True if prefill worker, False if decode worker
        logger: Logger instance

    Returns:
        TTNNFabricKVTransfer instance or None
    """
    logger = logger or TTLogger()

    try:
        # Determine ranks based on worker type
        if is_prefill:
            sender_rank = int(worker_id) if worker_id.isdigit() else 0
            receiver_rank = int(os.getenv("KV_CACHE__FABRIC_RECEIVER_RANK", "1"))
        else:
            sender_rank = int(os.getenv("KV_CACHE__FABRIC_SENDER_RANK", "0"))
            receiver_rank = int(worker_id) if worker_id.isdigit() else 1

        return create_fabric_transfer_from_device_runner(
            device_runner,
            sender_rank=sender_rank,
            receiver_rank=receiver_rank,
            logger=logger,
        )

    except Exception as e:
        logger.error(f"Failed to get fabric transfer for worker {worker_id}: {e}")
        return None

