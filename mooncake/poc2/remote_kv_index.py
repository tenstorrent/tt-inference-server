# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
RemoteKvIndex — "where are the KV blocks?" abstraction over Mooncake Store.

This module provides the directory service for the hybrid KV orchestration:
- publish(): advertise that this endpoint owns a block hash
- exist(): lookup which remote endpoints own the given block hashes

The Mooncake Store holds metadata only (endpoint + slot + block index),
NOT the actual KV bytes. The migration worker moves the bytes separately.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union
import json

# Hash type: real system uses uint64_t, but we accept both for flexibility
HashType = Union[int, str]


def _hash_to_key(h: HashType) -> str:
    """Convert hash to string key for storage."""
    return str(h) if isinstance(h, int) else h


@dataclass
class RemoteBlock:
    """Location of a KV block on a remote endpoint."""

    hash: HashType
    block_index: int
    src_endpoint_id: int
    src_slot: int


class RemoteKvIndex(ABC):
    """Abstract interface for the KV block directory."""

    @abstractmethod
    def publish(
        self, block_hash: HashType, endpoint_id: int, slot: int, block_index: int
    ) -> None:
        """
        Advertise that this endpoint owns a block.

        Args:
            block_hash: The hash identifying this KV block
            endpoint_id: The endpoint that owns the block
            slot: The slot number on that endpoint
            block_index: The block's index within the slot
        """
        pass

    @abstractmethod
    def exist(self, hashes: List[HashType]) -> List[RemoteBlock]:
        """
        Look up which hashes exist remotely and where.

        Args:
            hashes: List of block hashes to look up

        Returns:
            List of RemoteBlock for hashes that were found (may be empty)
        """
        pass

    @abstractmethod
    def remove(self, block_hash: HashType) -> bool:
        """
        Remove a block hash from the directory (e.g., on eviction).

        Args:
            block_hash: The hash to remove

        Returns:
            True if removed, False if not found
        """
        pass


class MockRemoteKvIndex(RemoteKvIndex):
    """
    Mock that returns empty for everything — simulates "nothing remote".

    Use this to verify that the scheduler behaves like today when no
    remote blocks are available.
    """

    def publish(
        self, block_hash: HashType, endpoint_id: int, slot: int, block_index: int
    ) -> None:
        pass

    def exist(self, hashes: List[HashType]) -> List[RemoteBlock]:
        return []

    def remove(self, block_hash: HashType) -> bool:
        return False


class InMemoryRemoteKvIndex(RemoteKvIndex):
    """
    In-memory implementation for testing without Mooncake master.

    Stores block locations in a dict. Useful for unit tests that don't
    want to spin up the full Mooncake infrastructure.
    """

    def __init__(self):
        self._index: dict[str, RemoteBlock] = {}

    def publish(
        self, block_hash: HashType, endpoint_id: int, slot: int, block_index: int
    ) -> None:
        key = _hash_to_key(block_hash)
        self._index[key] = RemoteBlock(
            hash=block_hash,
            block_index=block_index,
            src_endpoint_id=endpoint_id,
            src_slot=slot,
        )

    def exist(self, hashes: List[HashType]) -> List[RemoteBlock]:
        return [
            self._index[_hash_to_key(h)]
            for h in hashes
            if _hash_to_key(h) in self._index
        ]

    def remove(self, block_hash: HashType) -> bool:
        key = _hash_to_key(block_hash)
        if key in self._index:
            del self._index[key]
            return True
        return False


class MooncakeRemoteKvIndex(RemoteKvIndex):
    """
    Real implementation over MooncakeDistributedStore.

    Stores block location as JSON: {"endpoint_id": N, "slot": M, "block_index": K}
    Key is the block hash (converted to string).
    """

    def __init__(self, store):
        """
        Args:
            store: A MooncakeDistributedStore instance (already setup'd)
        """
        self._store = store

    def publish(
        self, block_hash: HashType, endpoint_id: int, slot: int, block_index: int
    ) -> None:
        key = _hash_to_key(block_hash)
        location = {
            "endpoint_id": endpoint_id,
            "slot": slot,
            "block_index": block_index,
        }
        value = json.dumps(location).encode("utf-8")
        ret = self._store.put(key, value)
        if ret != 0:
            raise RuntimeError(f"Mooncake put failed for hash {block_hash}: {ret}")

    def exist(self, hashes: List[HashType]) -> List[RemoteBlock]:
        if not hashes:
            return []

        keys = [_hash_to_key(h) for h in hashes]
        results = self._store.batch_is_exist(keys)
        hits = []

        for h, key, exists in zip(hashes, keys, results):
            if exists == 1:
                try:
                    raw = self._store.get(key)
                    if raw:
                        data = json.loads(raw.decode("utf-8"))
                        hits.append(
                            RemoteBlock(
                                hash=h,
                                block_index=data["block_index"],
                                src_endpoint_id=data["endpoint_id"],
                                src_slot=data["slot"],
                            )
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    print(
                        f"[MooncakeRemoteKvIndex] Warning: failed to parse block {h}: {e}"
                    )
                    continue

        return hits

    def remove(self, block_hash: HashType) -> bool:
        ret = self._store.remove(_hash_to_key(block_hash))
        return ret == 0
