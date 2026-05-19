#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# Launch 4 parallel galaxy_t3k docker servers on a Galaxy 6U chassis.
#
# Each instance owns one 8-chip slice (0-7, 8-15, 16-23, 24-31) and listens
# on its own port (8000-8003). Launches are staggered to avoid the fabric
# init race that hits when multiple containers initialise the tt-metal
# control plane simultaneously after a chassis reset.
#
# Env overrides:
#   MODEL                 Model name to pass to run.py (default: Molmo2-8B)
#   STAGGER_DELAY         Seconds between successive launches (default: 30)
#   LOG_DIR               Where to write per-instance run.py logs (default: /tmp)
#   OVERRIDE_DOCKER_IMAGE If set, passed as --override-docker-image
#   EXTRA_RUN_ARGS        Additional flags passed verbatim to run.py
#
# Usage:
#   ./scripts/launch_4x_galaxy_t3k.sh
#   MODEL=Qwen3-8B ./scripts/launch_4x_galaxy_t3k.sh
#   OVERRIDE_DOCKER_IMAGE=ghcr.io/.../foo:bar ./scripts/launch_4x_galaxy_t3k.sh
#
# Requires:
#   JWT_SECRET and HF_TOKEN already exported in the environment.
#   A clean Galaxy chassis (run `tt-smi -glx_reset` first if you're not sure).

set -uo pipefail

MODEL=${MODEL:-Molmo2-8B}
DELAY=${STAGGER_DELAY:-30}
LOG_DIR=${LOG_DIR:-/tmp}
IMG=${OVERRIDE_DOCKER_IMAGE:-}
EXTRA_ARGS=${EXTRA_RUN_ARGS:-}

# Resolve repo root so the script works regardless of where it's invoked from.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

mkdir -p "${LOG_DIR}"

echo "Launching 4× galaxy_t3k servers for ${MODEL} (stagger=${DELAY}s)"
for idx in 0 1 2 3; do
    start=$((idx * 8))
    end=$((start + 7))
    devs=$(seq -s, ${start} ${end})
    port=$((8000 + idx))
    log="${LOG_DIR}/server_${MODEL}_inst${idx}_port${port}.log"

    echo "[+$((idx * DELAY))s] inst ${idx}: devices=${devs} port=${port} log=${log}"
    nohup python3 run.py \
        --model "${MODEL}" \
        --workflow server --docker-server \
        --tt-device galaxy_t3k \
        --device-id "${devs}" \
        --service-port "${port}" \
        --skip-system-sw-validation \
        --dev-mode \
        ${IMG:+--override-docker-image "${IMG}"} \
        ${EXTRA_ARGS} \
        > "${log}" 2>&1 &
    echo "  run.py pid=$!"

    if [ ${idx} -lt 3 ]; then
        sleep "${DELAY}"
    fi
done

echo "All 4 launchers fired. Watch progress:"
for idx in 0 1 2 3; do
    echo "  tail -f ${LOG_DIR}/server_${MODEL}_inst${idx}_port$((8000 + idx)).log"
done
echo "When ready, health-check each: curl http://localhost:800{0..3}/health"
