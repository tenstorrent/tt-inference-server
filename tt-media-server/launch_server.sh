#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

export MODEL=${MODEL:-Llama-3.1-8B}
export DEVICE=${DEVICE:-n150}
export API_KEY=${API_KEY:-your-secret-key}
export ENVIRONMENT=development
export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)/tt-metal}
export IS_GALAXY=${IS_GALAXY:-False}
[ -n "$DEVICE_IDS" ] && export DEVICE_IDS

echo "Starting server: MODEL=$MODEL DEVICE=$DEVICE DEVICE_IDS=${DEVICE_IDS:-auto}"
cd "$(dirname "$0")"
PORT=${PORT:-8000}
uvicorn main:app --lifespan on --port $PORT --log-level warning
