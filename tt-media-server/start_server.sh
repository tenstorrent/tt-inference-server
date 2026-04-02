#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Production startup script for tt-media-server inside Docker.
# Used by systemd service (svc.sh) for auto-start on boot.

cd "$(dirname "$0")"

# Load Telegram credentials if available
[ -f ~/.tt-telegram.env ] && source ~/.tt-telegram.env

# Send Telegram notification
notify_telegram() {
    local msg="$1"
    if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
        curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}&text=${msg}" >/dev/null 2>&1 || true
    fi
}

notify_telegram "🔄 tt-media-server starting on $(hostname) at $(date)"

# Kill any existing server/worker processes (exclude our own PID)
pkill -9 -f "uvicorn main:app" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 2

# Activate venv
source ~/kmabee_demo/tt-xla/venv/bin/activate

# Data parallel on P300X2 (4 chips, 4 workers)
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/kmabee_demo/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto

export DEVICE=p300x2
export MODEL=Llama-3.1-8B-Instruct
exec ./launch_server.sh
