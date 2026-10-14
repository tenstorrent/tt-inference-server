#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# Launch Molmo2-8B on the four 1x8 slices of a 32-chip Galaxy (galaxy_t3k), one `run.py` local server per slice:
# slice N -> devices 8N..8N+7 -> port 8000+N.   Uses the `id_molmo2_Molmo2-8B_galaxy_t3k` spec
# (workflows/model_specs/dev/vlm.yaml), so MODEL_SPECS_ENV=dev is set.
#
#   ./scripts/launch_4x_galaxy_t3k.sh                 # slices 0 1 2 3 -> ports 8000..8003
#   SLICES="1 2 3" ./scripts/launch_4x_galaxy_t3k.sh  # only some slices
#   ./scripts/launch_4x_galaxy_t3k.sh stop
#
# STATUS: written against run.py's flags and validated only up to spec loading
# (MODEL_SPECS_ENV=dev python -c "from workflows.model_spec import MODEL_SPECS" lists both Molmo2 specs); the
# launch itself has NOT been executed from this script. Requirements: a tt-metal checkout with models/demos/molmo2
# (TT_METAL_HOME), its python venv with vLLM + the TT plugin (see tt-metal models/demos/molmo2/README.md, "Server"),
# HF weights for allenai/Molmo2-8B, and `tt-smi -r && tt-smi -glx_reset` done beforehand.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MODEL_SPECS_ENV=dev JWT_SECRET="${JWT_SECRET:-dummy}"
: "${TT_METAL_HOME:?set TT_METAL_HOME to the tt-metal checkout}"
SLICES="${SLICES:-0 1 2 3}"
PORT_BASE="${PORT_BASE:-8000}"
STAGGER_S="${STAGGER_S:-30}"          # stagger so the tt-metal init phases of the instances do not collide
LOG_DIR="${LOG_DIR:-/tmp/molmo2_run_py}"
mkdir -p "$LOG_DIR"

if [[ "${1:-}" == "stop" ]]; then
    pkill -f "run.py --model allenai/Molmo2-8B" || true
    exit 0
fi

for sl in $SLICES; do
    devices="$(seq -s, $((sl * 8)) $((sl * 8 + 7)))"
    port=$((PORT_BASE + sl))
    echo "slice $sl: devices $devices -> port $port (log $LOG_DIR/run_$sl.log)"
    (
        # The t3k mesh descriptor must be in the environment BEFORE the worker boots: the spec's env_vars are
        # applied after the metal environment is initialised.
        export TT_VISIBLE_DEVICES="$devices"
        export TT_MESH_GRAPH_DESC_PATH="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto"
        export TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS="127.0.0.1:$((50050 + sl))"   # per-instance port for tt-triage
        nohup "${PYTHON:-python3}" run.py --model allenai/Molmo2-8B --workflow server --local-server \
            --tt-device galaxy_t3k --impl molmo2 --device-id "$devices" --service-port "$port" \
            --tt-metal-home "$TT_METAL_HOME" ${TT_METAL_PYTHON_VENV_DIR:+--tt-metal-python-venv-dir "$TT_METAL_PYTHON_VENV_DIR"} \
            --skip-system-sw-validation --no-auth \
            > "$LOG_DIR/run_$sl.log" 2>&1 &
    )
    sleep "$STAGGER_S"
done
for sl in $SLICES; do
    until curl -sf "http://localhost:$((PORT_BASE + sl))/v1/models" > /dev/null; do sleep 5; done
    echo "port $((PORT_BASE + sl)) ready"
done
