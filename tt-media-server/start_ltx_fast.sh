#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# Start the LTX-2.3-Fast media server (full-time hosting).
#
# Bakes in every env var the server needs so a restart is reproducible and the
# console display name (SERVED_MODEL_NAME) always survives a restart. Each value
# is overridable from the environment; the defaults target this Galaxy box.
#
#   ./start_ltx_fast.sh            # start on :8000 with the baked-in config
#   PORT=8082 ./start_ltx_fast.sh  # override the port
set -euo pipefail

export TT_METAL_HOME="${TT_METAL_HOME:-/home/rsalman/tt-metal}"
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-/home/rsalman/tt-metal/tt_dit_cache}"
export TT_VIDEO_OUTPUT_DIR="${TT_VIDEO_OUTPUT_DIR:-/home/rsalman/tt-media-videos}"
export LTX_YUV_EXPORT="${LTX_YUV_EXPORT:-1}"
# Gate /health on the canary probe: if the device wedges, /health flips to unhealthy
# (instead of staying 200) so a load balancer / restart hook can detect and recover.
export CANARY_GATE_READINESS="${CANARY_GATE_READINESS:-true}"
# Console/API display name — decoupled from the checkpoint (settings.model_weights_path).
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Lightricks/LTX-2.3-Fast}"
export DEVICE="${DEVICE:-galaxy}"
export MODEL="${MODEL:-LTX-2.3-distilled}"
PORT="${PORT:-8000}"

# Run from the media-server dir so `main:app` and ./python_env resolve regardless
# of where the script is invoked from.
cd "$(dirname "$(readlink -f "$0")")"

echo "Starting LTX-2.3-Fast on :${PORT}  (SERVED_MODEL_NAME=${SERVED_MODEL_NAME}, MODEL=${MODEL}, DEVICE=${DEVICE})"
exec ./python_env/bin/uvicorn --host 0.0.0.0 main:app --lifespan on --port "${PORT}"
