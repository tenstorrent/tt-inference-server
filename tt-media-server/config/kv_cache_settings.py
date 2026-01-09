# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Configuration settings for KV cache transfer"""

from pydantic import BaseModel
from typing import Optional


class KVCacheSettings(BaseModel):
    """Settings for KV cache transfer between prefill and decode workers"""

    # Worker type configuration
    worker_type: Optional[str] = None  # "prefill" or "decode"

    # Worker pairing
    prefill_worker_id: Optional[str] = None  # ID of prefill worker (for decode worker)
    decode_worker_id: Optional[str] = None  # ID of decode worker (for prefill worker)

    # Transfer mechanism
    use_fabric_transfer: bool = False  # Use TTNN fabric socket transfer
    use_storage_fallback: bool = True  # Fallback to shared storage if fabric fails

    # Storage settings
    kv_cache_ttl_seconds: int = 300  # Time-to-live for KV cache in storage

    # Timeout settings
    kv_cache_wait_timeout: float = 30.0  # Timeout for waiting for KV cache (seconds)

    # Fabric transfer settings (when use_fabric_transfer=True)
    fabric_socket_buffer_size: int = 4 * 1024 * 1024  # 4MB socket buffer (L1 memory)
    fabric_sender_rank: int = 0
    fabric_receiver_rank: int = 1

