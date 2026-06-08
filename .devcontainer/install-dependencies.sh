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

cleanup_kissat_configure_artifacts() {
    local kissat_cache="${TT_METAL_HOME}/.cpmcache/kissat"

    if [ ! -d "${kissat_cache}" ]; then
        return
    fi

    # Kissat's configure script fails if a previous incremental build left this
    # symlink behind. Removing it lets configure recreate the link cleanly.
    for makefile in "${kissat_cache}"/*/src/makefile; do
        [ -e "${makefile}" ] || continue
        rm -f "${makefile}"
    done
}

cd "${cpp_server}"
./install_dependencies.sh

if [ "${clean_tt_metal}" -eq 1 ]; then
    cd "${TT_METAL_HOME}"
    ./build_metal.sh --clean
fi

cleanup_kissat_configure_artifacts

cd "${cpp_server}/tt-llm-engine"
./setup.sh --all

cd "${cpp_server}"
./build.sh --blaze
