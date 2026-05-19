#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# cpp_server build deps: apt + Rust (if needed) + Drogon (if not found).
# Optional: ./install_dependencies.sh --kafka for librdkafka when using -DKAFKA_ENABLED=ON.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTALL_KAFKA=0
INSTALL_GRPC=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --kafka)
            INSTALL_KAFKA=1
            shift
            ;;
        --grpc)
            INSTALL_GRPC=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--kafka] [--grpc]"
            echo "  --kafka              Also install librdkafka-dev (C + C++ client for CMake KAFKA_ENABLED=ON)"
            echo "  --grpc               Build and install gRPC from source with abseil (for CMake ENABLE_GRPC=ON)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

[ -n "${TT_METAL_HOME:-}" ] && export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

APT_PKGS=(
    build-essential cmake g++ pkg-config curl git wget
    libjsoncpp-dev uuid-dev zlib1g-dev libssl-dev libboost-all-dev
)
if [ "${INSTALL_KAFKA}" = 1 ]; then
    APT_PKGS+=(librdkafka-dev)
    echo "Kafka deps: will install librdkafka-dev (for KAFKA_ENABLED=ON builds)"
fi
if [ "${INSTALL_GRPC}" = 1 ]; then
    echo "gRPC deps: will build gRPC + abseil + protobuf from source (for ENABLE_GRPC=ON builds)"
fi

$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends "${APT_PKGS[@]}"

# tt-llm-engine's CMakeLists.txt requires CMake >= 3.24. Ubuntu 22.04 ships
# 3.22.1, so upgrade via pip when the apt version is too old. Idempotent on
# newer distros (already-satisfied check below).
CMAKE_REQUIRED_MAJOR=3
CMAKE_REQUIRED_MINOR=24
if command -v cmake >/dev/null 2>&1; then
    CMAKE_VERSION_RAW=$(cmake --version | head -n1 | awk '{print $3}')
    CMAKE_MAJOR=${CMAKE_VERSION_RAW%%.*}
    CMAKE_MINOR=${CMAKE_VERSION_RAW#*.}; CMAKE_MINOR=${CMAKE_MINOR%%.*}
else
    CMAKE_MAJOR=0
    CMAKE_MINOR=0
fi
if [ "${CMAKE_MAJOR}" -lt "${CMAKE_REQUIRED_MAJOR}" ] || \
   { [ "${CMAKE_MAJOR}" -eq "${CMAKE_REQUIRED_MAJOR}" ] && \
     [ "${CMAKE_MINOR}" -lt "${CMAKE_REQUIRED_MINOR}" ]; }; then
    echo "Upgrading CMake (have ${CMAKE_VERSION_RAW:-none}, need >= ${CMAKE_REQUIRED_MAJOR}.${CMAKE_REQUIRED_MINOR})"
    $SUDO apt-get install -y --no-install-recommends python3-pip
    $SUDO pip3 install --quiet --upgrade 'cmake>=3.24,<4'
    hash -r
    cmake --version | head -n1
fi

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
    # Disable Drogon's optional ORM/DB modules. We don't use them, and
    # leaving them on causes Drogon to auto-detect libpq/libmysqlclient/
    # libsqlite3/libhiredis at configure time and link them transitively
    # into our binary via Drogon::Drogon. That made the build artifact's
    # runtime deps a function of whatever happened to be installed on the
    # build runner — see PR history for the libpq.so.5-on-bench-runner
    # incident this avoids.
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_YAML_CONFIG=OFF \
        -DBUILD_POSTGRESQL=OFF -DBUILD_MYSQL=OFF -DBUILD_SQLITE=OFF -DBUILD_REDIS=OFF
    make -j"$(nproc 2>/dev/null || echo 4)"
    $SUDO make install
    [ "$(uname -s)" = "Linux" ] && $SUDO ldconfig
    cd "${SCRIPT_DIR}" && rm -rf "${DROGON_TMP}"
fi

# gRPC from source (includes abseil, protobuf, utf8_range with CMake configs)
grpc_found() {
    [ -f /usr/local/lib/cmake/grpc/gRPCConfig.cmake ] && \
    [ -f /usr/local/lib/cmake/absl/abslConfig.cmake ] && \
    [ -f /usr/local/lib/cmake/protobuf/protobuf-config.cmake ]
}
if [ "${INSTALL_GRPC}" = 1 ] && ! grpc_found; then
    # Remove apt protobuf/grpc to avoid header conflicts with source build
    echo "Removing apt protobuf/grpc packages to avoid version conflicts..."
    $SUDO apt-get remove -y --purge libprotobuf-dev protobuf-compiler libgrpc++-dev libgrpc-dev \
        protobuf-compiler-grpc libabsl-dev 2>/dev/null || true
    $SUDO apt-get autoremove -y 2>/dev/null || true

    GRPC_VERSION="v1.62.0"
    GRPC_TMP="/tmp/grpc_build"
    rm -rf "${GRPC_TMP}"
    echo "Building gRPC ${GRPC_VERSION} from source (includes abseil, protobuf, utf8_range)..."
    git clone --depth 1 --branch "${GRPC_VERSION}" --recurse-submodules --shallow-submodules \
        https://github.com/grpc/grpc.git "${GRPC_TMP}"
    mkdir -p "${GRPC_TMP}/build" && cd "${GRPC_TMP}/build"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DgRPC_BUILD_CSHARP_EXT=OFF \
        -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF \
        -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF \
        -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF \
        -DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF \
        -DgRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF \
        -DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF \
        -Dprotobuf_BUILD_LIBPROTOC=ON \
        -Dprotobuf_INSTALL=ON
    make -j"$(nproc 2>/dev/null || echo 4)"
    $SUDO make install
    [ "$(uname -s)" = "Linux" ] && $SUDO ldconfig
    cd "${SCRIPT_DIR}" && rm -rf "${GRPC_TMP}"
    echo "gRPC installation complete."
fi
