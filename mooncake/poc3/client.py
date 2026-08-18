# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
client.py — Helper class wrapping MooncakeDistributedStore for tier testing.

Provides convenient methods for:
- put/get with timing
- batch operations
- soft/hard pinning
- stats and monitoring
"""

import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class PutResult:
    """Result of a put operation."""

    key: str
    size_bytes: int
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class GetResult:
    """Result of a get operation."""

    key: str
    size_bytes: int
    latency_ms: float
    success: bool
    from_dram: Optional[bool] = None  # True=local DRAM, derived from `tier`
    tier: Optional[str] = None  # Authoritative tier label (see classify_read_tier)
    error: Optional[str] = None


class MooncakeClient:
    """
    Wrapper around MooncakeDistributedStore with timing and diagnostics.
    """

    # Default DFS directory (must match master_startup.sh)
    DEFAULT_DFS_DIR = "/tmp/mooncake_dfs_poc3"

    # Tier labels named after Mooncake's own replica types, listed in Mooncake's
    # SelectBestReplica priority order (the order a read prefers them):
    #   local DRAM > remote DRAM > local_disk (offload RPC) > shared disk (DFS)
    TIER_LOCAL_DRAM = "local DRAM"
    TIER_REMOTE_DRAM = "remote DRAM"
    TIER_LOCAL_DISK = "local_disk (remote SSD via offload RPC)"
    TIER_SHARED_DISK = "shared disk (DFS)"
    TIER_MISSING = "MISSING"
    TIER_UNKNOWN = "UNKNOWN"

    def __init__(
        self,
        client_id: str,
        master_addr: str = "localhost:50051",
        metadata_server: str = "http://localhost:8080/metadata",
        dram_size_mb: int = 64,  # Small for testing eviction
        buffer_size_mb: int = 16,
        dfs_dir: Optional[str] = None,
    ):
        self.client_id = client_id
        self.master_addr = master_addr
        self.metadata_server = metadata_server
        self.dram_size_mb = dram_size_mb
        self.buffer_size_mb = buffer_size_mb
        self.dfs_dir = dfs_dir or self.DEFAULT_DFS_DIR

        self._store = None
        self._connected = False
        # This client's own transport endpoint, discovered at connect time.
        # Needed to tell local DRAM apart from a remote node's DRAM, since
        # GetLocalEndpoints() is not bound in the Python API.
        self.local_endpoint = None

        # Stats
        self.puts = 0
        self.gets = 0
        self.hits = 0
        self.misses = 0

    def connect(self) -> bool:
        """Connect to Mooncake master."""
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError:
            print(f"[{self.client_id}] ERROR: mooncake package not installed")
            print(f"[{self.client_id}] Run: pip install mooncake-transfer-engine")
            return False

        self._store = MooncakeDistributedStore()

        ret = self._store.setup(
            "localhost",
            self.metadata_server,
            self.dram_size_mb * 1024 * 1024,  # global_segment_size
            self.buffer_size_mb * 1024 * 1024,  # local_buffer_size
            "tcp",
            "",  # rdma_devices
            self.master_addr,
        )

        if ret != 0:
            print(f"[{self.client_id}] ERROR: setup failed with code {ret}")
            return False

        self._connected = True
        # get_hostname() returns this client's own transport endpoint
        # ("host:port") — the same string that appears as transport_endpoint on
        # buffers held in THIS client's segment. That's what lets classify tell
        # local DRAM apart from a remote node's DRAM.
        self.local_endpoint = self._store.get_hostname()
        print(f"[{self.client_id}] Connected to {self.master_addr}")
        print(f"[{self.client_id}]   DRAM: {self.dram_size_mb}MB")
        print(f"[{self.client_id}]   Buffer: {self.buffer_size_mb}MB")
        print(f"[{self.client_id}]   Endpoint: {self.local_endpoint or 'unknown'}")
        return True

    def close(self):
        """Close connection."""
        if self._store:
            self._store.close()
            self._connected = False
            print(f"[{self.client_id}] Disconnected")

    def is_dfs_enabled(self) -> bool:
        """
        Check if DFS (SSD tier) is enabled.

        Returns True if the DFS directory exists and is writable.
        This doesn't guarantee master was started with --root_fs_dir,
        but it's a good indicator.
        """
        import os

        return os.path.isdir(self.dfs_dir)

    def is_key_on_dfs(self, key: str) -> bool:
        """
        Check if a specific key exists on DFS.

        Mooncake uses a directory structure: mooncake_cluster/{first_char}/{second_char}/{key}
        """
        import os

        # Mooncake uses first 2 chars as subdirs
        if len(key) >= 2:
            subpath = os.path.join(
                self.dfs_dir, "mooncake_cluster", key[0], key[1], key
            )
            if os.path.exists(subpath):
                return True

        # Also try direct path
        cluster_path = os.path.join(self.dfs_dir, "mooncake_cluster", key)
        if os.path.exists(cluster_path):
            return True

        # Fallback: search for the key
        for root, dirs, files in os.walk(
            os.path.join(self.dfs_dir, "mooncake_cluster")
        ):
            if key in files:
                return True

        return False

    def list_dfs_keys(self, limit: int = 20) -> List[str]:
        """List keys stored in DFS (for debugging)."""
        import os

        keys = []

        if not os.path.isdir(self.dfs_dir):
            return keys

        for root, dirs, files in os.walk(self.dfs_dir):
            for f in files:
                keys.append(f)
                if len(keys) >= limit:
                    return keys
        return keys

    def check_dfs_persistence(self, test_key: str = "__dfs_test__") -> bool:
        """
        Actually test if DFS persistence works.

        Puts data, waits for async write, checks if file appears.
        """
        import time

        if not self._connected:
            return False

        # Put test data
        test_data = b"DFS_PERSISTENCE_TEST"
        result = self.put(test_key, test_data)
        if not result.success:
            return False

        # Wait for async DFS write
        time.sleep(1.0)

        # Check if it appeared on disk
        on_dfs = self.is_key_on_dfs(test_key)

        # Cleanup
        self.remove(test_key)

        return on_dfs

    def classify_read_tier(self, key: str) -> str:
        """
        Return the tier a get() would actually read from.

        Authoritative (unlike a latency guess): get_replica_desc returns the
        same replica list the read path selects from, so this mirrors the
        client's SelectBestReplica preference order. Call BEFORE get(), since
        get() may promote remote data into local DRAM.
        """
        if not self._connected:
            return self.TIER_UNKNOWN

        replicas = self._store.get_replica_desc(key)
        if not replicas:
            return self.TIER_MISSING

        return self._tier_from_replicas(replicas)

    def _tier_from_replicas(self, replicas: List[Any]) -> str:
        """Mirror Mooncake's order: local DRAM > remote DRAM > local_disk > disk."""
        memory = [r for r in replicas if r.is_memory_replica()]
        if memory:
            endpoints = [
                r.get_memory_descriptor().buffer_descriptor.transport_endpoint
                for r in memory
            ]
            if self.local_endpoint and self.local_endpoint in endpoints:
                return self.TIER_LOCAL_DRAM
            return self.TIER_REMOTE_DRAM

        if any(r.is_local_disk_replica() for r in replicas):
            return self.TIER_LOCAL_DISK
        if any(r.is_disk_replica() for r in replicas):
            return self.TIER_SHARED_DISK
        return self.TIER_UNKNOWN

    def put(self, key: str, data: bytes) -> PutResult:
        """
        Put data into the store.

        Data goes to DRAM first, then async to DFS (if configured).
        """
        if not self._connected:
            return PutResult(key, 0, 0, False, "Not connected")

        start = time.perf_counter()
        try:
            ret = self._store.put(key, data)
            latency = (time.perf_counter() - start) * 1000

            self.puts += 1

            if ret == 0:
                return PutResult(key, len(data), latency, True)
            else:
                return PutResult(key, len(data), latency, False, f"put returned {ret}")

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return PutResult(key, 0, latency, False, str(e))

    def get(
        self, key: str, inspect_tier: bool = True
    ) -> tuple[Optional[bytes], GetResult]:
        """
        Get data from the store.

        When inspect_tier is True, the serving tier is resolved via
        get_replica_desc BEFORE the read (an extra metadata RPC, outside the
        latency measurement). Set it False on tight latency loops.
        """
        if not self._connected:
            return None, GetResult(key, 0, 0, False, error="Not connected")

        tier = self.classify_read_tier(key) if inspect_tier else None
        from_dram = (tier == self.TIER_LOCAL_DRAM) if tier else None

        start = time.perf_counter()
        try:
            data = self._store.get(key)
            latency = (time.perf_counter() - start) * 1000

            self.gets += 1

            if data is not None:
                self.hits += 1
                return data, GetResult(key, len(data), latency, True, from_dram, tier)
            else:
                self.misses += 1
                return None, GetResult(
                    key, 0, latency, False, tier=tier, error="Key not found"
                )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.misses += 1
            return None, GetResult(key, 0, latency, False, tier=tier, error=str(e))

    def exists(self, keys: List[str]) -> List[bool]:
        """Check if keys exist."""
        if not self._connected:
            return [False] * len(keys)

        results = self._store.batch_is_exist(keys)
        return [r == 1 for r in results]

    def remove(self, key: str) -> bool:
        """Remove a key."""
        if not self._connected:
            return False

        ret = self._store.remove(key)
        return ret == 0

    def batch_put(self, items: Dict[str, bytes]) -> List[PutResult]:
        """Put multiple items."""
        results = []
        for key, data in items.items():
            results.append(self.put(key, data))
        return results

    def batch_get(self, keys: List[str]) -> List[tuple[Optional[bytes], GetResult]]:
        """Get multiple items."""
        results = []
        for key in keys:
            results.append(self.get(key))
        return results

    def fill_dram(self, prefix: str, count: int, size_kb: int = 64) -> List[PutResult]:
        """
        Fill DRAM with test data to trigger eviction.

        Args:
            prefix: Key prefix
            count: Number of items
            size_kb: Size of each item in KB

        Returns:
            List of PutResults
        """
        data = b"X" * (size_kb * 1024)
        results = []

        print(
            f"[{self.client_id}] Filling DRAM: {count} items x {size_kb}KB = {count * size_kb / 1024:.1f}MB"
        )

        for i in range(count):
            key = f"{prefix}_{i:06d}"
            result = self.put(key, data)
            results.append(result)

            if not result.success:
                print(f"[{self.client_id}] put failed at item {i}: {result.error}")
                break

            if (i + 1) % 100 == 0:
                print(f"[{self.client_id}]   ... {i + 1}/{count} items")

        return results

    def stats(self) -> Dict[str, Any]:
        """Get client stats."""
        hit_rate = self.hits / self.gets if self.gets > 0 else 0
        return {
            "client_id": self.client_id,
            "puts": self.puts,
            "gets": self.gets,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1%}",
        }

    def print_stats(self):
        """Print stats."""
        s = self.stats()
        print(
            f"[{self.client_id}] Stats: puts={s['puts']} gets={s['gets']} "
            f"hits={s['hits']} misses={s['misses']} hit_rate={s['hit_rate']}"
        )


def create_test_data(size_kb: int = 64) -> bytes:
    """Create test data of specified size."""
    return b"MOONCAKE_TEST_DATA_" + b"X" * (size_kb * 1024 - 19)


# ===========================================================================
# DeepSeek-R1 KV Cache Format
# ===========================================================================
#
# From defaults.hpp:
#   KV_CACHE_BLOCK_SIZE = 32 (tokens per block)
#
# DeepSeek-R1: 128 kv_heads, 128 head_dim, 61 layers
# Block size: 2 * 32 * 128 * 128 * 2 = 2,097,152 bytes (2 MB per block)
# ===========================================================================

DEEPSEEK_CONFIG = {
    "num_layers": 61,
    "num_kv_heads": 128,
    "head_dim": 128,
    "block_size": 32,
    "dtype_bytes": 2,  # bfloat16
}


def get_kv_block_size(
    num_tokens: int = 32,
    num_kv_heads: int = 128,
    head_dim: int = 128,
    dtype_bytes: int = 2,
) -> int:
    """
    Calculate the size of a KV cache block in bytes.

    DeepSeek-R1 (32 tokens, 128 heads, 128 dim): 2 * 32 * 128 * 128 * 2 = 2,097,152 bytes = 2 MB
    """
    return 2 * num_tokens * num_kv_heads * head_dim * dtype_bytes


def create_kv_block_key(
    session_id: str,
    layer: int,
    block_idx: int,
) -> str:
    """
    Create a key for storing a KV cache block.

    Key format: kv:{session_id}:L{layer:03d}:B{block_idx:05d}

    This matches how the migration layer addresses KV chunks:
    - session_id identifies the inference session (aka slot)
    - layer is the transformer layer index
    - block_idx is the position block within that layer
    """
    return f"kv:{session_id}:L{layer:03d}:B{block_idx:05d}"


def create_kv_block_data(
    session_id: str,
    layer: int,
    block_idx: int,
    num_tokens: int = 32,
    num_kv_heads: int = 128,
    head_dim: int = 128,
    dtype_bytes: int = 2,
) -> bytes:
    """
    Create KV cache block data for DeepSeek-R1.

    Data is deterministic based on (session_id, layer, block_idx) for verification.
    """
    import hashlib

    # K and V are equal-sized halves of the block
    k_size = num_tokens * num_kv_heads * head_dim * dtype_bytes
    v_size = k_size

    # Create deterministic pattern based on (session_id, layer, block_idx)
    # This lets us verify data integrity after migration
    seed = f"{session_id}:{layer}:{block_idx}".encode()
    hash_bytes = hashlib.sha256(seed).digest()

    # Generate K data: repeat hash pattern to fill k_size
    k_data = (hash_bytes * ((k_size // len(hash_bytes)) + 1))[:k_size]

    # Generate V data: use different hash for V
    v_seed = f"{session_id}:{layer}:{block_idx}:V".encode()
    v_hash = hashlib.sha256(v_seed).digest()
    v_data = (v_hash * ((v_size // len(v_hash)) + 1))[:v_size]

    return bytes(k_data) + bytes(v_data)


def verify_kv_block_data(
    data: bytes,
    session_id: str,
    layer: int,
    block_idx: int,
    num_tokens: int = 32,
    num_kv_heads: int = 128,
    head_dim: int = 128,
    dtype_bytes: int = 2,
) -> bool:
    """Verify KV block data matches expected pattern."""
    expected = create_kv_block_data(
        session_id, layer, block_idx, num_tokens, num_kv_heads, head_dim, dtype_bytes
    )
    return data == expected
