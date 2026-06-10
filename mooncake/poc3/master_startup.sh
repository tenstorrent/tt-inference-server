#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# Start Mooncake master with SSD/DFS tier enabled
#
# This enables the two-tier storage:
#   - DRAM: fast, limited capacity
#   - DFS: slower, larger, persists evicted data

set -e

# Create DFS directory
DFS_DIR="/tmp/mooncake_dfs_poc3"
mkdir -p "$DFS_DIR"
echo "[master] DFS directory: $DFS_DIR"

# Clean up old data
rm -rf "$DFS_DIR"/*
echo "[master] Cleaned DFS directory"

# Configuration
PORT=50051
METADATA_PORT=8080            # HTTP metadata server (clients connect here)
EVICTION_HIGH_WATERMARK=0.80  # Evict at 80% for easier testing
EVICTION_RATIO=0.10           # Evict 10% at a time
DFS_SIZE=$((1024*1024*1024))  # 1GB DFS limit

echo "[master] Starting Mooncake master..."
echo "[master]   Port: $PORT"
echo "[master]   Metadata port: $METADATA_PORT"
echo "[master]   DFS dir: $DFS_DIR"
echo "[master]   Eviction watermark: $EVICTION_HIGH_WATERMARK"
echo "[master]   Eviction ratio: $EVICTION_RATIO"
echo "[master]   Offload to SSD/DFS: enabled"

# Check if mooncake_master is available
if ! command -v mooncake_master &> /dev/null; then
    echo "[master] ERROR: mooncake_master not found"
    echo "[master] Try: pip install mooncake-transfer-engine"
    exit 1
fi

# Start master with DFS + offload enabled.
#   --enable_http_metadata_server: clients connect via http://localhost:8080/metadata
#   --enable_offload: copy DRAM replicas to SSD/DFS, so evicted keys survive as a
#                     file-tier replica (without it, eviction just deletes the key)
exec mooncake_master \
    --port=$PORT \
    --root_fs_dir="$DFS_DIR" \
    --global_file_segment_size=$DFS_SIZE \
    --enable_http_metadata_server=true \
    --http_metadata_server_port=$METADATA_PORT \
    --enable_offload=true \
    --eviction_high_watermark_ratio=$EVICTION_HIGH_WATERMARK \
    --eviction_ratio=$EVICTION_RATIO
