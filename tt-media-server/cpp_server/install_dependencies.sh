#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# cpp_server deps: apt + Drogon, plus build tooling unless --runtime is used.
# Optional: ./install_dependencies.sh --kafka for librdkafka when using -DKAFKA_ENABLED=ON.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTALL_KAFKA=0
RUNTIME_ONLY=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --kafka)
            INSTALL_KAFKA=1
            shift
            ;;
        --runtime)
            RUNTIME_ONLY=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--kafka] [--runtime]"
            echo "  --kafka              Also install librdkafka-dev (C + C++ client for CMake KAFKA_ENABLED=ON)"
            echo "  --runtime            Install runtime deps only; skip LLVM/Rust build tooling"
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

drogon_found() {
    pkg-config --exists drogon 2>/dev/null || [ -f /usr/local/lib/cmake/Drogon/DrogonConfig.cmake ] || [ -f /usr/lib/cmake/Drogon/DrogonConfig.cmake ]
}

APT_PKGS=(
    libjsoncpp-dev uuid-dev zlib1g-dev libssl-dev libboost-all-dev
)
if [ "${RUNTIME_ONLY}" = 0 ]; then
    APT_PKGS+=(build-essential cmake g++ pkg-config curl git wget gnupg ca-certificates ccache)
elif ! drogon_found 2>/dev/null; then
    # Runtime consumers normally rely on a preinstalled Drogon. If it is absent,
    # install the build tools needed to compile Drogon rather than failing later.
    APT_PKGS+=(build-essential cmake g++ pkg-config curl git)
fi
if [ "${INSTALL_KAFKA}" = 1 ]; then
    APT_PKGS+=(librdkafka-dev)
    echo "Kafka deps: will install librdkafka-dev (for KAFKA_ENABLED=ON builds)"
fi

$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends "${APT_PKGS[@]}"

install_llvm_apt_repo() {
    . /etc/os-release
    local codename="${VERSION_CODENAME:-}"
    if [ -z "${codename}" ]; then
        echo "Unable to determine distro codename for LLVM apt repository" >&2
        exit 1
    fi

    local keyring="/usr/share/keyrings/llvm-snapshot.gpg"
    local list_file="/etc/apt/sources.list.d/llvm-toolchain-${codename}-20.list"
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | $SUDO gpg --batch --yes --dearmor -o "${keyring}"
    echo "deb [signed-by=${keyring}] https://apt.llvm.org/${codename}/ llvm-toolchain-${codename}-20 main" \
        | $SUDO tee "${list_file}" >/dev/null
    $SUDO apt-get update -qq
}

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
if [ "${RUNTIME_ONLY}" = 0 ] && \
   { [ "${CMAKE_MAJOR}" -lt "${CMAKE_REQUIRED_MAJOR}" ] || \
     { [ "${CMAKE_MAJOR}" -eq "${CMAKE_REQUIRED_MAJOR}" ] && \
       [ "${CMAKE_MINOR}" -lt "${CMAKE_REQUIRED_MINOR}" ]; }; }; then
    echo "Upgrading CMake (have ${CMAKE_VERSION_RAW:-none}, need >= ${CMAKE_REQUIRED_MAJOR}.${CMAKE_REQUIRED_MINOR})"
    $SUDO apt-get install -y --no-install-recommends python3-pip
    $SUDO pip3 install --quiet --upgrade 'cmake>=3.24,<4'
    hash -r
    cmake --version | head -n1
fi

if [ "${RUNTIME_ONLY}" = 0 ] && { ! command -v clang-format-20 >/dev/null 2>&1 || ! command -v clang-tidy-20 >/dev/null 2>&1; }; then
    install_llvm_apt_repo
    $SUDO apt-get install -y --no-install-recommends clang-format-20 clang-tidy-20
fi
$SUDO rm -rf /var/lib/apt/lists/*

if [ "${RUNTIME_ONLY}" = 0 ] && ! command -v cargo >/dev/null 2>&1; then
    [ -f "${HOME}/.cargo/env" ] && . "${HOME}/.cargo/env"
    if ! command -v cargo >/dev/null 2>&1; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        . "${HOME}/.cargo/env"
    fi
fi

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

APT_PKGS=(
    libibverbs-dev          # RDMA verbs (Transfer Engine)
    libgoogle-glog-dev      # glog
    libnuma-dev             # NUMA awareness in Transfer Engine
    libunwind-dev           # stack unwinder used by glog
    libcurl4-openssl-dev    # USE_HTTP metadata client
    libyaml-cpp-dev         # Transfer Engine topology config
    libmsgpack-dev          # Mooncake RPC
    libjemalloc-dev         # Store allocator
    liburing-dev            # io_uring backend
    libzstd-dev             # Compression
    libxxhash-dev           # Hashing
    libasio-dev             # coro_rpc (yalantinglibs)
    libpython3-dev          # find_package(Python3)
    pybind11-dev            # Python bindings (if you need the wheel)
    patchelf                # Wheel rpath patching
    # libhiredis-dev        # only if USE_REDIS=ON / STORE_USE_REDIS=ON
)

$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends "${APT_PKGS[@]}"

MOONCAKE_REF="v0.3.6.post1"   # pin me
git clone --depth 1 --branch "${MOONCAKE_REF}" \
    --recurse-submodules --shallow-submodules \
    https://github.com/kvcache-ai/Mooncake.git /tmp/mooncake
# Build yalantinglibs from Mooncake's pinned submodule (avoids version drift)
cmake -S /tmp/mooncake/extern/yalantinglibs -B /tmp/mooncake/extern/yalantinglibs/build \
    -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build /tmp/mooncake/extern/yalantinglibs/build -j"$(nproc)"
$SUDO cmake --install /tmp/mooncake/extern/yalantinglibs/build
cmake -S /tmp/mooncake -B /tmp/mooncake/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_SHARED_LIBS=ON \
    -DUSE_CUDA=OFF \
    -DWITH_STORE=ON \
    -DUSE_HTTP=ON \
    -DWITH_P2P_STORE=OFF \
    -DUSE_ETCD=OFF \
    -DSTORE_USE_ETCD=OFF \
    -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF
cmake --build /tmp/mooncake/build -j"$(nproc)"
$SUDO cmake --install /tmp/mooncake/build
$SUDO ldconfig
