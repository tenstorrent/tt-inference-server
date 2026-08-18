#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
test_realistic_kv.py — Test Mooncake with DeepSeek-R1 KV cache format

DeepSeek-R1: 61 layers, 128 KV heads, 128 head_dim, 32 tokens/block
Block size: 2 * 32 * 128 * 128 * 2 = 2,097,152 bytes (2 MB per block)
"""

import sys
from client import (
    MooncakeClient,
    create_kv_block_data,
    create_kv_block_key,
    verify_kv_block_data,
    get_kv_block_size,
    DEEPSEEK_CONFIG,
)


def test_deepseek_blocks():
    """Test put/get of DeepSeek-R1 KV cache blocks (2MB each)."""
    print("=" * 60)
    print("DEEPSEEK-R1 KV CACHE TEST")
    print("=" * 60)
    print()

    cfg = DEEPSEEK_CONFIG
    block_size = get_kv_block_size(
        num_tokens=cfg["block_size"],
        num_kv_heads=cfg["num_kv_heads"],
        head_dim=cfg["head_dim"],
        dtype_bytes=cfg["dtype_bytes"],
    )
    print(f"Block size: {block_size:,} bytes ({block_size / (1024 * 1024):.1f} MB)")
    print(f"Layers: {cfg['num_layers']}, KV heads: {cfg['num_kv_heads']}")
    print()

    # Need enough DRAM for several 2MB blocks
    client = MooncakeClient("deepseek_test", dram_size_mb=256)
    if not client.connect():
        print("[FAIL] Could not connect to Mooncake master")
        return False

    try:
        session_id = "deepseek_session_001"
        num_blocks = 5

        print(f"Testing {num_blocks} blocks...")
        latencies_put = []
        latencies_get = []

        for i in range(num_blocks):
            layer = i % cfg["num_layers"]
            block_idx = i

            key = create_kv_block_key(session_id, layer, block_idx)
            data = create_kv_block_data(
                session_id=session_id,
                layer=layer,
                block_idx=block_idx,
                num_tokens=cfg["block_size"],
                num_kv_heads=cfg["num_kv_heads"],
                head_dim=cfg["head_dim"],
                dtype_bytes=cfg["dtype_bytes"],
            )

            # Put
            put_result = client.put(key, data)
            if not put_result.success:
                print(f"  [FAIL] put L{layer}B{block_idx}: {put_result.error}")
                return False
            latencies_put.append(put_result.latency_ms)

            # Get
            retrieved, get_result = client.get(key)
            if not get_result.success:
                print(f"  [FAIL] get L{layer}B{block_idx}: {get_result.error}")
                return False
            latencies_get.append(get_result.latency_ms)

            # Verify
            if not verify_kv_block_data(
                retrieved,
                session_id,
                layer,
                block_idx,
                num_tokens=cfg["block_size"],
                num_kv_heads=cfg["num_kv_heads"],
                head_dim=cfg["head_dim"],
                dtype_bytes=cfg["dtype_bytes"],
            ):
                print(f"  [FAIL] verify L{layer}B{block_idx}")
                return False

            print(
                f"  [OK] L{layer:02d}B{block_idx}: put={put_result.latency_ms:.2f}ms get={get_result.latency_ms:.2f}ms"
            )

        avg_put = sum(latencies_put) / len(latencies_put)
        avg_get = sum(latencies_get) / len(latencies_get)
        put_throughput = block_size / (avg_put / 1000) / (1024 * 1024)
        get_throughput = block_size / (avg_get / 1000) / (1024 * 1024)

        print()
        print(f"Avg put: {avg_put:.2f}ms ({put_throughput:.0f} MB/s)")
        print(f"Avg get: {avg_get:.2f}ms ({get_throughput:.0f} MB/s)")
        print()
        print("=" * 60)
        print("[PASS] DeepSeek-R1 KV cache test completed")
        print("=" * 60)
        return True

    finally:
        client.close()


if __name__ == "__main__":
    success = test_deepseek_blocks()
    sys.exit(0 if success else 1)
