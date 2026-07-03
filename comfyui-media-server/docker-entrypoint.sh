#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

################################################################################
# Container entrypoint for comfyui-media-server.
#
# `run.py --docker-server` starts this image detached, publishes the service port
# onto the container's port 8000, and injects env vars — notably:
#   MODEL   = model_spec.model_name   (e.g. "sdxl", "wan22", "sd35" ...)
#   DEVICE  = device_type, lowercased (e.g. "p150", "p300x2", "t3k" ...)
#
# This script translates those env vars into the CLI that server.py expects and
# hands off to launch_server.sh. TT_METAL_HOME / PYTHONPATH / LD_LIBRARY_PATH are
# already exported by the image (see Dockerfile), and the venv lives under
# $TT_METAL_HOME/python_env — launch_server.sh activates it.
################################################################################

set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Map run.py's MODEL to server.py's --model {sdxl,sd35,wan22} --------------
raw_model="$(printf '%s' "${MODEL:-sdxl}" | tr '[:upper:]' '[:lower:]')"
case "${raw_model}" in
    *sdxl*|*xl*)                     SERVER_MODEL="sdxl" ;;
    *wan*)                           SERVER_MODEL="wan22" ;;
    *sd3*|*sd35*|*stable-diffusion-3*) SERVER_MODEL="sd35" ;;
    sdxl|sd35|wan22)                 SERVER_MODEL="${raw_model}" ;;
    *)
        echo "entrypoint: cannot map MODEL='${MODEL:-}' to sdxl/sd35/wan22." >&2
        echo "            Set MODEL to one of those, or extend this mapping." >&2
        exit 1
        ;;
esac

# --- Board comes straight from DEVICE ----------------------------------------
BOARD="$(printf '%s' "${DEVICE:-}" | tr '[:upper:]' '[:lower:]')"

# --- Port: run.py maps the host service port onto container 8000 -------------
PORT="${SERVICE_PORT:-8000}"

ARGS=(--model "${SERVER_MODEL}" --port "${PORT}" --host "0.0.0.0")
if [ -n "${BOARD}" ]; then
    ARGS+=(--board "${BOARD}")
fi
# Optional passthroughs so run.py / dev overrides can reach server.py.
if [ "${DEV_MODE:-0}" = "1" ] || [ "${SERVER_DEV:-0}" = "1" ]; then
    ARGS+=(--dev)
fi

echo "entrypoint: MODEL='${MODEL:-}' DEVICE='${DEVICE:-}' -> launch_server.sh ${ARGS[*]}"
exec "${SERVER_DIR}/launch_server.sh" "${ARGS[@]}"
