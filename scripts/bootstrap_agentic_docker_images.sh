#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
HOST_PYTHON="${PYTHON:-python3}"
AGENTIC_PYTHON="${REPO_ROOT}/.workflow_venvs/.venv_evals_agentic/bin/python"
MAX_WORKERS="${MAX_WORKERS:-4}"

usage() {
    cat <<'EOF'
Bootstrap the EVALS_AGENTIC venv and pre-pull Docker images for:
  - Terminal Bench 2.0 and 2.1
  - Tau3-bench (base images; task images are built locally by Harbor)
  - SWE-bench Verified (agent and scoring-harness images)

Usage:
  scripts/bootstrap_agentic_docker_images.sh [pull-script options]

Options are forwarded to scripts/pull_agentic_docker_images.py. Examples:
  --max-workers 8   Override the default of four concurrent pulls
  --force           Pull images even when already present
  --no-progress     Disable the progress bar
  --runtime podman  Use a container runtime other than Docker

Warning: the complete SWE-bench Verified image set requires substantial disk
space. Existing local images are skipped unless --force is supplied.
EOF
}

for arg in "$@"; do
    if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
        usage
        exit 0
    fi
done

command -v "${HOST_PYTHON}" >/dev/null 2>&1 || {
    echo "error: ${HOST_PYTHON} is not available" >&2
    exit 1
}
command -v git >/dev/null 2>&1 || {
    echo "error: git is not available" >&2
    exit 1
}
command -v docker >/dev/null 2>&1 || {
    echo "error: docker is not available" >&2
    exit 1
}
docker info >/dev/null 2>&1 || {
    echo "error: the Docker daemon is unavailable" >&2
    exit 1
}

cd "${REPO_ROOT}"

echo "Bootstrapping EVALS_AGENTIC virtual environment..."
"${HOST_PYTHON}" - <<'PY'
import logging

from workflows.bootstrap_uv import bootstrap_uv
from workflows.workflow_types import WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
bootstrap_uv()
config = VENV_CONFIGS[WorkflowVenvType.EVALS_AGENTIC]
if not config.setup(model_spec=None):
    raise RuntimeError("Failed to setup EVALS_AGENTIC")
PY

if [[ ! -x "${AGENTIC_PYTHON}" ]]; then
    echo "error: agentic venv interpreter was not created: ${AGENTIC_PYTHON}" >&2
    exit 1
fi

echo "Pulling Terminal Bench 2.0/2.1, Tau3, and SWE-bench Verified images..."
"${AGENTIC_PYTHON}" "${SCRIPT_DIR}/pull_agentic_docker_images.py" \
    --benchmark terminal-bench-2 \
    --benchmark tau3-bench \
    --benchmark swe-bench \
    --subset verified \
    --split test \
    --swe-target all \
    --max-workers "${MAX_WORKERS}" \
    "$@"
