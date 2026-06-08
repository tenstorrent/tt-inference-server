#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

set -euo pipefail

repo_root="${1:-$(pwd)}"
clean_tt_metal=0
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            clean_tt_metal=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [repo-root] [--clean]"
            echo "  --clean    Run tt-metal/build_metal.sh --clean before rebuilding"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

repo_root="$(cd "${repo_root}" && pwd -P)"
cpp_server="${repo_root}/tt-media-server/cpp_server"

export TT_METAL_HOME="${TT_METAL_HOME:-${cpp_server}/tt-llm-engine/tt-metal}"

cd "${cpp_server}"
./install_dependencies.sh

cd "${cpp_server}/tt-llm-engine"
./setup.sh --all

cd "${cpp_server}"
./build.sh --blaze
