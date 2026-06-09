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

    # Kissat's configure script fails if a previous incremental build left
    # symlinks in its cache. Removing only Kissat's generated build dirs keeps
    # the rest of tt-metal's dependency cache intact.
    for build_dir in "${kissat_cache}"/*/build; do
        [ -e "${build_dir}" ] || continue
        rm -rf "${build_dir}"
    done
}

cleanup_moved_cmake_build_dir() {
    local build_dir="$1"
    local source_dir="$2"
    local cache_file="${build_dir}/CMakeCache.txt"

    if [ ! -f "${cache_file}" ]; then
        return
    fi

    if grep -qxF "CMAKE_HOME_DIRECTORY:INTERNAL=${source_dir}" "${cache_file}"; then
        return
    fi

    echo "Removing stale CMake build directory created for a different source path: ${build_dir}"
    rm -rf "${build_dir}"
}

cd "${cpp_server}"
./install_dependencies.sh

if [ "${clean_tt_metal}" -eq 1 ]; then
    cd "${TT_METAL_HOME}"
    ./build_metal.sh --clean
fi

cleanup_kissat_configure_artifacts

llm_engine="${cpp_server}/tt-llm-engine"
cleanup_moved_cmake_build_dir "${llm_engine}/build-full" "${llm_engine}"

cd "${llm_engine}"
./setup.sh --all

cd "${cpp_server}"
cleanup_moved_cmake_build_dir "${cpp_server}/build" "${cpp_server}"
./build.sh --blaze
