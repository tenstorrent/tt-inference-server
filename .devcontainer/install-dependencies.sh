#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

set -euo pipefail

repo_root="${1:-$(pwd)}"
repo_root="$(cd "${repo_root}" && pwd -P)"
cpp_server="${repo_root}/tt-media-server/cpp_server"

export TT_METAL_HOME="${TT_METAL_HOME:-${cpp_server}/tt-llm-engine/tt-metal}"

cd "${cpp_server}"
./install_dependencies.sh

cd "${cpp_server}/tt-llm-engine"
./setup.sh --all

cd "${cpp_server}"
./build.sh --blaze
