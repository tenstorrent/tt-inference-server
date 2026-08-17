#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Install tt-media-server pip dependencies into the active uv-managed venv
# using the same flags as the production Dockerfile:
#
#   --index-strategy unsafe-best-match : consider PyPI as a fallback for
#       packages that also appear on https://download.pytorch.org/whl/cpu
#       (e.g. requests, tqdm, numpy). Without this, uv stops at the first
#       index that contains the package and refuses pinned versions that
#       only exist on PyPI.
#   --overrides uv-overrides.txt       : work around the upstream `lightning`
#       PyPI quarantine and tt-metal's numpy<2 pin. See uv-overrides.txt for
#       the full rationale.
#
# Usage (from any directory):
#   bash tt-media-server/install_requirements.sh
# Or, if your active venv is already correct:
#   ./tt-media-server/install_requirements.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv pip install \
    --index-strategy unsafe-best-match \
    --overrides "${SCRIPT_DIR}/uv-overrides.txt" \
    -r "${SCRIPT_DIR}/requirements.txt" \
    "$@"
