#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Create cpp_server/.venv with the migration-CLI deps (confluent-kafka) so
# scripts/migration_cli.py runs. Idempotent.
#
#   bash scripts/setup-migration-cli.sh
#
# The venv lives at cpp_server/.venv; set nothing else. Override the
# interpreter with PYTHON=/usr/bin/python3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_SERVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${CPP_SERVER_DIR}/.venv"
REQ="${SCRIPT_DIR}/migration_cli_requirements.txt"
# Prefer the system python (has venv/ensurepip after apt); avoid re-using a
# no-pip venv interpreter that happens to be first on PATH.
PYTHON="${PYTHON:-/usr/bin/python3}"
command -v "${PYTHON}" >/dev/null 2>&1 || PYTHON="python3"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Creating venv at ${VENV} (using ${PYTHON})..."
  if ! "${PYTHON}" -m venv "${VENV}" 2>/dev/null; then
    echo "ERROR: '${PYTHON} -m venv' failed. Install the venv module first, e.g.:" >&2
    echo "  apt-get install -y python3-venv python3-pip" >&2
    exit 2
  fi
fi

"${VENV}/bin/python" -m pip install --upgrade pip >/dev/null
if [[ -f "${REQ}" ]]; then
  "${VENV}/bin/python" -m pip install -r "${REQ}"
else
  "${VENV}/bin/python" -m pip install 'confluent-kafka>=2.5,<3'
fi

echo "----------------------------------------"
"${VENV}/bin/python" -c 'import confluent_kafka as c; print("confluent-kafka", c.version()[0], "ready in", "'"${VENV}"'")'
echo "The Kafka e2e + scripts/migration_cli.py will auto-use ${VENV}."
echo "If run from elsewhere, export MIGRATION_CLI_VENV=${VENV}"
