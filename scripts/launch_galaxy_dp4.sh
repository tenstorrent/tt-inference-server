#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# Launch ONE Molmo2-8B server on a whole 32-chip Galaxy with in-process data parallelism (DP=4): the tt-metal adapter
# splits the (8,4) mesh into four 1x8 replicas, one request slot each. Uses the `id_molmo2_Molmo2-8B_galaxy` spec
# (workflows/model_specs/dev/vlm.yaml: max_concurrency 4, TT_DATA_PARALLEL=4; vLLM's own data_parallel_size is NOT set).
#
#   ./scripts/launch_galaxy_dp4.sh            # one server on :8000
#   PORT=8000 TT_METAL_HOME=... TT_METAL_PYTHON_VENV_DIR=... ./scripts/launch_galaxy_dp4.sh
#
# Requirements: as scripts/launch_4x_galaxy_t3k.sh (tt-metal with models/demos/molmo2 and a venv with vLLM + the TT
# plugin; weights available; `tt-smi -r && tt-smi -glx_reset` done). tt-metal must be reachable as ./tt-metal (or set
# the spec's relative TT_MESH_GRAPH_DESC_PATH accordingly) when the spec's env var is used.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MODEL_SPECS_ENV=dev JWT_SECRET="${JWT_SECRET:-dummy}"
: "${TT_METAL_HOME:?set TT_METAL_HOME to the tt-metal checkout}"
exec "${PYTHON:-python3}" run.py --model allenai/Molmo2-8B --workflow server --local-server \
    --tt-device galaxy --impl molmo2 --service-port "${PORT:-8000}" \
    --tt-metal-home "$TT_METAL_HOME" ${TT_METAL_PYTHON_VENV_DIR:+--tt-metal-python-venv-dir "$TT_METAL_PYTHON_VENV_DIR"} \
    --skip-system-sw-validation --no-auth
