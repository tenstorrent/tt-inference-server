# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Communication protocol for prefill-decode node communication.

Defines message formats and protocol constants for MPI communication.
"""

from dataclasses import dataclass
from enum import IntEnum
import struct
from typing import Optional

import numpy as np


class MessageType(IntEnum):
    """Message types for P/D communication."""

    PREFILL_REQUEST = 1
    PREFILL_RESPONSE = 2
    KV_CACHE_LAYER = 3
    SHUTDOWN = 99


@dataclass
class PrefillRequest:
    """Request from decode node to prefill node."""

    request_id: int
    seq_len: int
    # Token IDs would go here in real implementation
    # For POC, we just send seq_len to determine KV cache size

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        # Format: message_type (1 byte) + request_id (4 bytes) + seq_len (4 bytes)
        return struct.pack("<BII", MessageType.PREFILL_REQUEST, self.request_id, self.seq_len)

    @classmethod
    def from_bytes(cls, data: bytes) -> "PrefillRequest":
        """Deserialize from bytes."""
        msg_type, request_id, seq_len = struct.unpack("<BII", data[:9])
        assert msg_type == MessageType.PREFILL_REQUEST
        return cls(request_id=request_id, seq_len=seq_len)

    @staticmethod
    def size() -> int:
        """Size of serialized request in bytes."""
        return 9  # 1 + 4 + 4


@dataclass
class PrefillResponse:
    """Response header from prefill node before streaming KV cache."""

    request_id: int
    num_layers: int
    layer_size_bytes: int
    status: int = 0  # 0 = success, non-zero = error

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        return struct.pack(
            "<BIIII",
            MessageType.PREFILL_RESPONSE,
            self.request_id,
            self.num_layers,
            self.layer_size_bytes,
            self.status,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "PrefillResponse":
        """Deserialize from bytes."""
        msg_type, request_id, num_layers, layer_size_bytes, status = struct.unpack(
            "<BIIII", data[:17]
        )
        assert msg_type == MessageType.PREFILL_RESPONSE
        return cls(
            request_id=request_id,
            num_layers=num_layers,
            layer_size_bytes=layer_size_bytes,
            status=status,
        )

    @staticmethod
    def size() -> int:
        """Size of serialized response in bytes."""
        return 17  # 1 + 4 + 4 + 4 + 4


def create_kv_layer_buffer(layer_size_bytes: int) -> np.ndarray:
    """Create a numpy buffer for receiving KV cache layer."""
    return np.empty(layer_size_bytes, dtype=np.uint8)


def create_mock_kv_layer(layer_size_bytes: int, layer_idx: int) -> np.ndarray:
    """Create mock KV cache data for testing."""
    # Fill with layer index for verification
    data = np.full(layer_size_bytes, layer_idx % 256, dtype=np.uint8)
    return data
