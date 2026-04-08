#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# cpp_server build deps: apt + Rust (if needed) + Drogon (if not found).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -n "${TT_METAL_HOME:-}" ] && export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
    build-essential cmake g++ pkg-config curl git wget \
    libjsoncpp-dev uuid-dev zlib1g-dev libssl-dev libboost-all-dev
if ! command -v clang-format-20 >/dev/null 2>&1; then
    LLVM_SH="/tmp/llvm.sh"
    curl -sSL -o "${LLVM_SH}" https://apt.llvm.org/llvm.sh
    chmod +x "${LLVM_SH}"
    $SUDO "${LLVM_SH}" 20
    rm -f "${LLVM_SH}"
    $SUDO apt-get install -y --no-install-recommends clang-format-20
fi
$SUDO rm -rf /var/lib/apt/lists/*

if ! command -v cargo >/dev/null 2>&1; then
    [ -f "${HOME}/.cargo/env" ] && . "${HOME}/.cargo/env"
    if ! command -v cargo >/dev/null 2>&1; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        . "${HOME}/.cargo/env"
    fi
fi

drogon_found() {
    pkg-config --exists drogon 2>/dev/null || [ -f /usr/local/lib/cmake/Drogon/DrogonConfig.cmake ] || [ -f /usr/lib/cmake/Drogon/DrogonConfig.cmake ]
}
if ! drogon_found; then
    DROGON_TMP="/tmp/drogon_build"
    rm -rf "${DROGON_TMP}"
    git clone --depth 1 --branch v1.9.12 --recurse-submodules https://github.com/drogonframework/drogon.git "${DROGON_TMP}"
    mkdir -p "${DROGON_TMP}/build" && cd "${DROGON_TMP}/build"
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_YAML_CONFIG=OFF
    make -j"$(nproc 2>/dev/null || echo 4)"
    $SUDO make install
    [ "$(uname -s)" = "Linux" ] && $SUDO ldconfig
    cd "${SCRIPT_DIR}" && rm -rf "${DROGON_TMP}"
fi
