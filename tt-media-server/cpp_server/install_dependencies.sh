#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# cpp_server deps: apt + Drogon, plus build tooling unless --runtime is used.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTALL_KAFKA=0
INSTALL_MIGRATION_WORKER=0
RUNTIME_ONLY=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --kafka)
            INSTALL_KAFKA=1
            shift
            ;;
        --migration-worker)
            INSTALL_MIGRATION_WORKER=1
            INSTALL_KAFKA=1
            shift
            ;;
        --runtime)
            RUNTIME_ONLY=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--kafka] [--migration-worker] [--runtime]"
            echo "  --kafka              Also install librdkafka-dev (C + C++ client for CMake KAFKA_ENABLED=ON)"
            echo "  --migration-worker   Install Kafka, KV-table, Mooncake, and yalantinglibs build dependencies"
            echo "  --runtime            Install runtime deps only; skip LLVM/Rust build tooling"
            echo ""
            echo "Environment:"
            echo "  YALANTINGLIBS_SOURCE Override the yalantinglibs source directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [ "${INSTALL_MIGRATION_WORKER}" = 1 ] && [ "${RUNTIME_ONLY}" = 1 ]; then
    echo "Error: --migration-worker is a build dependency mode and cannot be combined with --runtime." >&2
    exit 1
fi
YALANTING_SOURCE=""
if [ "${INSTALL_MIGRATION_WORKER}" = 1 ]; then
    YALANTING_SOURCE="${YALANTINGLIBS_SOURCE:-${SCRIPT_DIR}/third_party/Mooncake/extern/yalantinglibs}"
    if [ ! -f "${YALANTING_SOURCE}/CMakeLists.txt" ]; then
        echo "Error: yalantinglibs source not found at ${YALANTING_SOURCE}" >&2
        exit 1
    fi
fi

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
if [ "${INSTALL_MIGRATION_WORKER}" = 1 ]; then
    # KvChunkAddressTable protobuf generation; fmt 11.1.4 is handled below to
    # match tt-metal rather than using Jammy's incompatible fmt 8 package.
    APT_PKGS+=(libprotobuf-dev protobuf-compiler)

    # Mooncake control plane: TCP/HTTP metadata, config, flags, and logging.
    APT_PKGS+=(
        libasio-dev libcurl4-openssl-dev libgflags-dev libgoogle-glog-dev
        libunwind-dev libyaml-cpp-dev
    )

    # Mooncake data plane: RDMA transport and NUMA-aware memory placement.
    APT_PKGS+=(libibverbs-dev libnuma-dev)

    # Ninja is the generator used by the tt-metal Clang toolchain and yalantinglibs.
    APT_PKGS+=(ninja-build)
fi
if [ "${INSTALL_KAFKA}" = 1 ]; then
    # librdkafka-dev depends on librdkafka1 + librdkafka++1 on Ubuntu/Debian,
    # but list them explicitly so minimal base images that prune transitive
    # deps still end up with a working C++ runtime. CMakeLists.txt:1134-1136
    # requires both -lrdkafka and -lrdkafka++ plus the librdkafka/ headers.
    if [ "${RUNTIME_ONLY}" = 1 ]; then
        APT_PKGS+=(librdkafka1 librdkafka++1)
        echo "Kafka deps: will install librdkafka1 + librdkafka++1 (runtime)"
    else
        APT_PKGS+=(librdkafka-dev librdkafka1 librdkafka++1)
        echo "Kafka deps: will install librdkafka-dev + runtime libs (for KAFKA_ENABLED=ON builds)"
    fi
fi

# The tt-metal builder image ships LLVM libc++/libunwind headers, while Ubuntu's
# glog package requires GNU libunwind-dev. The worker uses the libstdc++ toolchain,
# so replace the conflicting libc++ development headers before dependency solve.
if [ "${INSTALL_MIGRATION_WORKER}" = 1 ] && dpkg-query -W libunwind-20-dev >/dev/null 2>&1; then
    $SUDO apt-get remove -y libunwind-20-dev
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

NEED_CLANG_TOOLS=0
if [ "${RUNTIME_ONLY}" = 0 ] && \
   { ! command -v clang-format-20 >/dev/null 2>&1 || \
     ! command -v clang-tidy-20 >/dev/null 2>&1; }; then
    NEED_CLANG_TOOLS=1
fi
NEED_CLANG_COMPILER=0
if [ "${INSTALL_MIGRATION_WORKER}" = 1 ] && ! command -v clang++-20 >/dev/null 2>&1; then
    NEED_CLANG_COMPILER=1
fi
if [ "${NEED_CLANG_TOOLS}" = 1 ] || [ "${NEED_CLANG_COMPILER}" = 1 ]; then
    install_llvm_apt_repo
    LLVM_PKGS=()
    # Formatting and static analysis are required by the regular C++ CI jobs.
    [ "${NEED_CLANG_TOOLS}" = 1 ] && LLVM_PKGS+=(clang-format-20 clang-tidy-20)
    # The migration worker includes tt-metal reflect headers, which require Clang.
    [ "${NEED_CLANG_COMPILER}" = 1 ] && LLVM_PKGS+=(clang-20)
    $SUDO apt-get install -y --no-install-recommends "${LLVM_PKGS[@]}"
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

if [ "${INSTALL_MIGRATION_WORKER}" = 1 ]; then
    FMT_VERSION="11.1.4"
    FMT_CONFIG="${TT_METAL_HOME:-}/build_Release/lib/cmake/fmt/fmt-config.cmake"
    if [ ! -f "${FMT_CONFIG}" ] && \
       ! pkg-config --atleast-version="${FMT_VERSION}" fmt 2>/dev/null; then
        # Local source builds lack tt-metal's prebuilt fmt package. Install the
        # same version so spdlog, tt-metal headers, and yalantinglibs share one ABI.
        FMT_SOURCE="/tmp/fmt-source"
        FMT_BUILD="/tmp/fmt-build"
        rm -rf "${FMT_SOURCE}" "${FMT_BUILD}"
        git clone --depth 1 --branch "${FMT_VERSION}" \
            https://github.com/fmtlib/fmt.git "${FMT_SOURCE}"
        cmake -S "${FMT_SOURCE}" -B "${FMT_BUILD}" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DFMT_DOC=OFF \
            -DFMT_TEST=OFF
        cmake --build "${FMT_BUILD}"
        $SUDO cmake --install "${FMT_BUILD}"
        rm -rf "${FMT_SOURCE}" "${FMT_BUILD}"
    fi

    # Mooncake's transfer_engine target requires yalantinglibs as an installed
    # CMake package; tests and examples are unrelated to the worker.
    YALANTING_BUILD="/tmp/yalantinglibs-build"
    rm -rf "${YALANTING_BUILD}"
    cmake -S "${YALANTING_SOURCE}" -B "${YALANTING_BUILD}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_BENCHMARK=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_UNIT_TESTS=OFF
    cmake --build "${YALANTING_BUILD}"
    $SUDO cmake --install "${YALANTING_BUILD}"
    rm -rf "${YALANTING_BUILD}"
fi
