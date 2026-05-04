#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# Build cpp_server on exabox (no sudo, ephemeral /home).
#
# All local installs (Drogon, Rust) go to /data/$USER/.local so they
# persist across compute nodes.  System apt packages (libjsoncpp-dev,
# uuid-dev, libboost-all-dev, …) are assumed to already be installed.
#
# Usage:
#   ./build_exabox.sh [OPTIONS]        # same flags as build.sh
#   ./build_exabox.sh --install-deps   # one-time: build+install Drogon & Rust
#
# After building, run the binary with:
#   LD_LIBRARY_PATH=/data/$USER/.local/lib:$LD_LIBRARY_PATH \
#     ./build/tt_media_server_cpp -p 8002

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Persistent prefix — survives across exabox compute nodes (unlike /home)
LOCAL_PREFIX="/data/${USER}/.local"
RUSTUP_DIR="/data/${USER}/.rustup"
CARGO_DIR="/data/${USER}/.cargo"

# ── Parse arguments ──────────────────────────────────────────────────────
BUILD_TYPE="Release"
SANITIZE_THREAD="OFF"
SANITIZE_ADDRESS="OFF"
ENABLE_TRACY="OFF"
ENABLE_BLAZE="OFF"
CLANG_TIDY="OFF"
TOOLCHAIN_PATH_ARG=""
CXX_COMPILER_PATH=""
KAFKA_ENABLED="OFF"
INSTALL_DEPS=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-deps)  INSTALL_DEPS=1; shift ;;
        --debug)         BUILD_TYPE="Debug"; shift ;;
        --tsan)          SANITIZE_THREAD="ON"; BUILD_TYPE="Debug"; shift ;;
        --asan)          SANITIZE_ADDRESS="ON"; BUILD_TYPE="Debug"; shift ;;
        --tracy)         ENABLE_TRACY="ON"; shift ;;
        --blaze)         ENABLE_BLAZE="ON"; shift ;;
        --clang-tidy)    CLANG_TIDY="ON"; shift ;;
        --kafka)         KAFKA_ENABLED="ON"; shift ;;
        --toolchain-path)   TOOLCHAIN_PATH_ARG="$2"; shift 2 ;;
        --cxx-compiler-path) CXX_COMPILER_PATH="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Exabox-specific options:"
            echo "  --install-deps       One-time setup: build Drogon + install Rust to /data/\$USER/.local"
            echo ""
            echo "Build options (same as build.sh):"
            echo "  --debug              Debug build (default: Release)"
            echo "  --tsan               ThreadSanitizer"
            echo "  --asan               AddressSanitizer"
            echo "  --tracy              Tracy profiling"
            echo "  --blaze              tt-blaze pipeline_manager support"
            echo "  --clang-tidy         Run clang-tidy during build"
            echo "  --kafka              Kafka support (needs librdkafka-dev)"
            echo "  --toolchain-path P   CMake toolchain file"
            echo "  --cxx-compiler-path P  C++ compiler path"
            exit 0
            ;;
        *) echo "Unknown option: $1 (use --help)"; exit 1 ;;
    esac
done

if [ "${SANITIZE_THREAD}" = "ON" ] && [ "${SANITIZE_ADDRESS}" = "ON" ]; then
    echo "Error: --tsan and --asan are mutually exclusive."; exit 1
fi

# ── Dependency installation (--install-deps) ─────────────────────────────
install_drogon() {
    local drogon_cmake="${LOCAL_PREFIX}/lib/cmake/Drogon/DrogonConfig.cmake"
    if [ -f "${drogon_cmake}" ]; then
        echo "Drogon already installed at ${LOCAL_PREFIX}"
        return 0
    fi

    echo ""
    echo "Building Drogon → ${LOCAL_PREFIX} ..."

    local drogon_src="${SCRIPT_DIR}/deps/drogon"
    local drogon_tmp=""
    if [ -d "${drogon_src}" ]; then
        echo "  Using bundled source: ${drogon_src}"
    else
        drogon_tmp="/tmp/drogon_build_$$"
        echo "  Cloning drogon v1.9.12 → ${drogon_tmp}"
        git clone --depth 1 --branch v1.9.12 --recurse-submodules \
            https://github.com/drogonframework/drogon.git "${drogon_tmp}"
        drogon_src="${drogon_tmp}"
    fi

    mkdir -p "${drogon_src}/build" && cd "${drogon_src}/build"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${LOCAL_PREFIX}" \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_CTL=OFF \
        -DBUILD_YAML_CONFIG=OFF
    make -j"$(nproc 2>/dev/null || echo 4)"
    make install
    cd "${SCRIPT_DIR}"
    [ -n "${drogon_tmp}" ] && rm -rf "${drogon_tmp}"

    echo "Drogon installed to ${LOCAL_PREFIX}"
}

install_rust() {
    local cargo_bin="${CARGO_DIR}/bin/cargo"
    local rustup_cargo="${RUSTUP_DIR}/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo"

    if [ -x "${cargo_bin}" ] || [ -x "${rustup_cargo}" ]; then
        echo "Rust already installed under /data/${USER}"
        return 0
    fi

    echo ""
    echo "Installing Rust → ${RUSTUP_DIR} ..."
    RUSTUP_HOME="${RUSTUP_DIR}" CARGO_HOME="${CARGO_DIR}" \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- --default-toolchain stable --no-modify-path -y
    echo "Rust installed (rustup=${RUSTUP_DIR}, cargo=${CARGO_DIR})"
}

if [ "${INSTALL_DEPS}" -eq 1 ]; then
    mkdir -p "${LOCAL_PREFIX}"
    install_drogon
    install_rust
    echo ""
    echo "=============================================="
    echo "  Dependencies installed to ${LOCAL_PREFIX}"
    echo "  Re-run without --install-deps to build."
    echo "=============================================="
    exit 0
fi

# ── Verify prerequisites ─────────────────────────────────────────────────
if [ ! -f "${LOCAL_PREFIX}/lib/cmake/Drogon/DrogonConfig.cmake" ]; then
    echo "ERROR: Drogon not found at ${LOCAL_PREFIX}."
    echo "  Run:  $0 --install-deps"
    exit 1
fi

# ── Rust: prefer /data paths over ephemeral /home ────────────────────────
# Rustup proxies in .cargo/bin (both /data and /home) fail unless a default
# toolchain is configured.  Strip ALL .cargo/bin entries from PATH and use
# the direct toolchain binary instead.
CLEAN_PATH=$(echo "${PATH}" | tr ':' '\n' | grep -v '\.cargo/bin' | tr '\n' ':' | sed 's/:$//')
export PATH="${CLEAN_PATH}"

RESOLVED_CARGO=""
find_rust() {
    # 1. Direct toolchain binary (bypasses all rustup proxies)
    local tc_bin="${RUSTUP_DIR}/toolchains/stable-x86_64-unknown-linux-gnu/bin"
    if [ -x "${tc_bin}/cargo" ]; then
        export PATH="${tc_bin}:${PATH}"
        RESOLVED_CARGO="${tc_bin}/cargo"
        return 0
    fi
    # 2. System cargo (if it works)
    if command -v cargo >/dev/null 2>&1 && cargo --version >/dev/null 2>&1; then
        RESOLVED_CARGO="$(command -v cargo)"
        return 0
    fi
    return 1
}

if ! find_rust; then
    echo "ERROR: Rust/cargo not found."
    echo "  Run:  $0 --install-deps"
    exit 1
fi

echo "=============================================="
echo "  Building TT Media Server (C++ Drogon)"
echo "  Build type:    ${BUILD_TYPE}"
echo "  Local prefix:  ${LOCAL_PREFIX}"
echo "  Rust:          $(command -v cargo) ($(cargo --version 2>/dev/null || echo '?'))"
echo "  Blaze:         ${ENABLE_BLAZE}"
echo "  Kafka:         ${KAFKA_ENABLED}"
echo "=============================================="

# ── Tokenizer pre-fetch ──────────────────────────────────────────────────
TOKENIZER_DIR="${SCRIPT_DIR}/tokenizers"
mkdir -p "${TOKENIZER_DIR}"

HF_TOKEN_RESOLVED="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [ -z "${HF_TOKEN_RESOLVED}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
    HF_TOKEN_RESOLVED=$(cat "${HOME}/.cache/huggingface/token")
fi

download_tokenizer() {
    local model_name="$1" hf_repo="$2" requires_auth="$3"
    local model_dir="${TOKENIZER_DIR}/${model_name}"
    local tok_json="${model_dir}/tokenizer.json"
    local tok_config="${model_dir}/tokenizer_config.json"

    if [ -f "${tok_json}" ] && [ -f "${tok_config}" ]; then
        echo "  Using existing ${model_name} tokenizer."
        return 0
    fi
    if [ "${requires_auth}" = "true" ] && [ -z "${HF_TOKEN_RESOLVED}" ]; then
        echo "  Skipping ${model_name} (gated — set HF_TOKEN to download)."
        return 0
    fi

    local wget_args=()
    [ "${requires_auth}" = "true" ] && [ -n "${HF_TOKEN_RESOLVED}" ] && \
        wget_args=(--header "Authorization: Bearer ${HF_TOKEN_RESOLVED}")

    mkdir -p "${model_dir}"
    echo "  Downloading ${model_name} tokenizer..."
    wget -q "${wget_args[@]}" -O "${tok_json}" "${hf_repo}/tokenizer.json" || {
        rm -f "${tok_json}"; echo "  ERROR: failed to download tokenizer.json"; return 1; }
    wget -q "${wget_args[@]}" -O "${tok_config}" "${hf_repo}/tokenizer_config.json" || {
        rm -f "${tok_config}"; echo "  ERROR: failed to download tokenizer_config.json"; return 1; }
}

echo ""
echo "Pre-fetching tokenizers..."
download_tokenizer "deepseek-ai/DeepSeek-R1-0528" \
    "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/raw/main" "false"
download_tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
    "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/raw/main" "true"

# ── tt-metal toolchain detection ─────────────────────────────────────────
USE_METAL_TOOLCHAIN=0
TOOLCHAIN_PATH=""
CMAKE_GENERATOR=""
if [ -n "${TT_METAL_HOME:-}" ] && [ -d "${TT_METAL_HOME}/tt_metal/api" ]; then
    echo ""
    echo "TT_METAL_HOME: ${TT_METAL_HOME} (Metal C++ API enabled)"
    METAL_TOOLCHAIN="${TT_METAL_HOME}/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake"
    if [ -f "${METAL_TOOLCHAIN}" ] && [ "$(uname -m)" = "x86_64" ]; then
        if command -v clang++-20 >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1; then
            USE_METAL_TOOLCHAIN=1
            TOOLCHAIN_PATH="${METAL_TOOLCHAIN}"
            CMAKE_GENERATOR="Ninja"
            echo "Using tt-metal toolchain: ${TOOLCHAIN_PATH}"
        fi
    fi
    if [ "${USE_METAL_TOOLCHAIN}" -eq 0 ]; then
        if command -v clang++ >/dev/null 2>&1; then
            echo "Using fallback: clang++"
        else
            echo "ERROR: TT_METAL_HOME set but no Clang found (required for tt-metal headers)."
            exit 1
        fi
    fi
elif [ -n "${TT_METAL_HOME:-}" ]; then
    echo "WARNING: TT_METAL_HOME set but tt_metal/api not found at ${TT_METAL_HOME}/tt_metal/api"
else
    echo ""
    echo "TT_METAL_HOME not set — building without tt-metal (mock backend only)"
fi

# ── CMake configure ──────────────────────────────────────────────────────
mkdir -p "${BUILD_DIR}"

# Clear any cached cargo path from a previous broken configure
if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    sed -i '/CARGO_EXECUTABLE/d' "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null || true
fi

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DCMAKE_PREFIX_PATH="${LOCAL_PREFIX}"
    -DSANITIZE_THREAD="${SANITIZE_THREAD}"
    -DSANITIZE_ADDRESS="${SANITIZE_ADDRESS}"
    -DENABLE_TRACY="${ENABLE_TRACY}"
    -DENABLE_BLAZE="${ENABLE_BLAZE}"
    -DCLANG_TIDY="${CLANG_TIDY}"
    -DKAFKA_ENABLED="${KAFKA_ENABLED}"
    -DCARGO_EXECUTABLE="${RESOLVED_CARGO}"
)
[ -n "${TT_METAL_HOME:-}" ] && CMAKE_ARGS+=(-DTT_METAL_HOME="${TT_METAL_HOME}")

if [ -n "${CXX_COMPILER_PATH}" ]; then
    CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="${CXX_COMPILER_PATH}")
    command -v ninja >/dev/null 2>&1 && CMAKE_ARGS+=(-G Ninja)
elif [ -n "${TOOLCHAIN_PATH_ARG}" ]; then
    CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_PATH_ARG}")
    command -v ninja >/dev/null 2>&1 && CMAKE_ARGS+=(-G Ninja)
elif [ -n "${TOOLCHAIN_PATH}" ]; then
    CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_PATH}" -G "${CMAKE_GENERATOR}")
elif [ -n "${TT_METAL_HOME:-}" ] && [ -d "${TT_METAL_HOME}/tt_metal/api" ]; then
    CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang)
fi

echo ""
echo "Configuring CMake..."
cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" "${CMAKE_ARGS[@]}"

if [ -f "${BUILD_DIR}/compile_commands.json" ]; then
    ln -sf "${BUILD_DIR}/compile_commands.json" "${SCRIPT_DIR}/compile_commands.json"
fi

# ── Build ────────────────────────────────────────────────────────────────
echo ""
echo "Building..."
NPROC=$(nproc 2>/dev/null || echo 4)
cmake --build "${BUILD_DIR}" -j"${NPROC}"

echo ""
echo "=============================================="
echo "  Build complete!"
echo "  Binary: ${BUILD_DIR}/tt_media_server_cpp"
echo "=============================================="
echo ""
echo "Run with:"
echo "  LD_LIBRARY_PATH=${LOCAL_PREFIX}/lib:\$LD_LIBRARY_PATH \\"
echo "    ./build/tt_media_server_cpp -p 8002"
echo ""
