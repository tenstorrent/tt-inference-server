#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

set -euo pipefail

workspace="${1:-$(pwd)}"
workspace="$(cd "${workspace}" && pwd -P)"
workspace_name="$(basename "${workspace}")"
workspace_parent="$(basename "$(dirname "${workspace}")")"
expected_parent="container-dev"

if [ "${workspace_name}" != "tt-inference-server" ] || [ "${workspace_parent}" != "${expected_parent}" ]; then
    cat >&2 <<EOF
ERROR: tt-inference-server devcontainer must be opened from a ${expected_parent} parent directory.

Current workspace:
  ${workspace}

Expected shape example:
  /data/<user>/${expected_parent}/tt-inference-server

Move or clone the repository under ${expected_parent}, then reopen that folder in Cursor.
EOF
    exit 1
fi
