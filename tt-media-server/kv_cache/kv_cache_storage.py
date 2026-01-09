# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
KV Cache Storage and Data Structures

This module provides data structures and storage mechanisms for KV cache
transfer between prefill and decode workers.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import torch


class WorkerType(Enum):
    """Type of worker - prefill or decode"""
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class KVCacheMetadata:
    """Metadata about KV cache structure"""
    task_id: str  # Original request task ID
    num_layers: int
    num_heads: int
    head_dim: int
    seq_len: int  # Length after prefill
    batch_size: int
    dtype: str
    source_worker_id: str
    created_at: float = field(default_factory=time.time)
    # Additional metadata for decode continuation
    last_token_id: Optional[int] = None
    prompt_tokens: Optional[List[int]] = None


@dataclass
class KVCache:
    """Represents KV cache structure"""
    keys: List[torch.Tensor]  # List of key tensors per layer
    values: List[torch.Tensor]  # List of value tensors per layer
    metadata: KVCacheMetadata


class KVCacheStorage:
    """
    Centralized storage for KV cache metadata and coordination

    Note: For Fabric socket transfer, this stores only metadata.
    The actual KV cache is transferred directly device-to-device via fabric.

    Storage types:
    - "memory": In-memory dict (single process, for metadata only)
    - "multiprocessing": Shared dict via Manager (multi-process, for metadata)
    - "fabric": Fabric socket transfer (no storage, direct device-to-device)
    """

    def __init__(self, storage_type: str = "memory"):
        self.storage_type = storage_type
        # Store only metadata, not the actual KV cache tensors
        # For fabric transfer, KV cache goes directly device-to-device
        self._metadata_store: Dict[str, KVCacheMetadata] = {}
        self._lock = threading.RLock()
        self._cleanup_tasks: Dict[str, threading.Timer] = {}

    def store_metadata(self, task_id: str, metadata: KVCacheMetadata, ttl_seconds: int = 300):
        """
        Store KV cache metadata (for coordination)

        For fabric transfer, this stores only metadata.
        The actual KV cache is transferred directly via fabric socket.
        """
        with self._lock:
            self._metadata_store[task_id] = metadata
            # Schedule cleanup if TTL provided
            if ttl_seconds > 0:
                self._schedule_cleanup(task_id, ttl_seconds)

    def retrieve_metadata(self, task_id: str) -> Optional[KVCacheMetadata]:
        """Retrieve KV cache metadata by task_id"""
        with self._lock:
            return self._metadata_store.get(task_id)

    def remove(self, task_id: str):
        """Remove KV cache metadata from storage"""
        with self._lock:
            if task_id in self._metadata_store:
                del self._metadata_store[task_id]
            if task_id in self._cleanup_tasks:
                self._cleanup_tasks[task_id].cancel()
                del self._cleanup_tasks[task_id]

    def store(self, task_id: str, kv_cache: KVCache, ttl_seconds: int = 300):
        """
        Store KV cache (legacy method - stores only metadata for fabric transfer)

        For fabric transfer, only metadata is stored.
        Use store_metadata() for explicit metadata-only storage.
        """
        self.store_metadata(task_id, kv_cache.metadata, ttl_seconds)

    def retrieve(self, task_id: str) -> Optional[KVCache]:
        """
        Retrieve KV cache metadata (legacy method)

        For fabric transfer, this returns None as actual cache is transferred via fabric.
        Use retrieve_metadata() to get metadata.
        """
        metadata = self.retrieve_metadata(task_id)
        if metadata:
            # Return a placeholder - actual cache comes via fabric
            return None
        return None

    def _schedule_cleanup(self, task_id: str, ttl_seconds: int):
        """Schedule automatic cleanup after TTL"""
        def cleanup():
            self.remove(task_id)

        # Cancel existing cleanup if any
        if task_id in self._cleanup_tasks:
            self._cleanup_tasks[task_id].cancel()

        timer = threading.Timer(ttl_seconds, cleanup)
        timer.start()
        self._cleanup_tasks[task_id] = timer

    def list_active_caches(self) -> List[str]:
        """List all active cache task IDs (by metadata)"""
        with self._lock:
            return list(self._metadata_store.keys())


# Global storage instance (could be shared via multiprocessing.Manager for multi-process)
_global_kv_storage: Optional[KVCacheStorage] = None


def get_kv_storage() -> KVCacheStorage:
    """Get global KV cache storage instance"""
    global _global_kv_storage
    if _global_kv_storage is None:
        _global_kv_storage = KVCacheStorage()
    return _global_kv_storage

