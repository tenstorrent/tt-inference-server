#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="Release"

# Pull in dependency prefix set up by install_dependencies.sh (if present).
# It exports CMAKE_PREFIX_PATH / PKG_CONFIG_PATH / LD_LIBRARY_PATH so Drogon,
# JsonCpp, libuuid, etc. installed under ${HOME}/.local are discoverable.
if [ -f "${SCRIPT_DIR}/deps/env.sh" ]; then
    # shellcheck source=/dev/null
    . "${SCRIPT_DIR}/deps/env.sh"
fi

# Parse arguments
SANITIZE_THREAD="OFF"
SANITIZE_ADDRESS="OFF"
ENABLE_TRACY="OFF"
ENABLE_BLAZE="OFF"
CLANG_TIDY="OFF"
TOOLCHAIN_PATH_ARG=""
CXX_COMPILER_PATH=""
KAFKA_ENABLED="OFF"
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --tsan)
            SANITIZE_THREAD="ON"
            BUILD_TYPE="Debug"
            shift
            ;;
        --asan)
            SANITIZE_ADDRESS="ON"
            BUILD_TYPE="Debug"
            shift
            ;;
        --tracy)
            ENABLE_TRACY="ON"
            BUILD_TYPE="Debug"
            shift
            ;;
        --blaze)
            ENABLE_BLAZE="ON"
            shift
            ;;
        --clang-tidy)
            CLANG_TIDY="ON"
            shift
            ;;
        --kafka)
            KAFKA_ENABLED="ON"
            shift
            ;;
        --toolchain-path)
            TOOLCHAIN_PATH_ARG="$2"
            shift 2
            ;;
        --cxx-compiler-path)
            CXX_COMPILER_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug              Build in Debug mode (default: Release)"
            echo "  --tsan               Build with ThreadSanitizer for data-race detection"
            echo "  --asan               Build with AddressSanitizer + LeakSanitizer for memory/leak detection"
            echo "  --tracy              Build with Tracy profiling instrumentation"
            echo "  --blaze              Build with tt-blaze pipeline_manager support"
            echo "  --clang-tidy          Run clang-tidy during build (lint = build, same as tt-metal)"
            echo "  --kafka              Enable Kafka (CMake KAFKA_ENABLED=ON; needs librdkafka-dev)"
            echo "  --toolchain-path P   Use CMake toolchain file (overrides TT_METAL_HOME toolchain)"
            echo "  --cxx-compiler-path P  Set C++ compiler (overrides toolchain)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "${SANITIZE_THREAD}" = "ON" ] && [ "${SANITIZE_ADDRESS}" = "ON" ]; then
    echo "Error: --tsan and --asan are mutually exclusive."
    exit 1
fi

echo "=============================================="
echo "  Building TT Media Server (C++ Drogon)"
echo "  Build type: ${BUILD_TYPE}"
echo "  ThreadSanitizer: ${SANITIZE_THREAD}"
echo "  AddressSanitizer: ${SANITIZE_ADDRESS}"
echo "  Tracy: ${ENABLE_TRACY}"
echo "  Blaze: ${ENABLE_BLAZE}"
echo "  Clang-Tidy: ${CLANG_TIDY}"
echo "  AddressSanitizer: ${SANITIZE_ADDRESS}"
echo "  Tracy profiling: ${ENABLE_TRACY}"
echo "  Clang-tidy: ${CLANG_TIDY}"
echo "  Kafka (KAFKA_ENABLED): ${KAFKA_ENABLED}"
echo "=============================================="

# Ensure cargo (Rust) is in PATH for tokenizers-cpp and that rustc is recent
# enough (monostate, pulled transitively, needs rustc >= 1.79).
# CARGO_HOME / RUSTUP_HOME may have been redirected off $HOME by
# install_dependencies.sh when / was full (see deps/env.sh).
_CARGO_HOME="${CARGO_HOME:-${HOME}/.cargo}"
if ! command -v cargo >/dev/null 2>&1; then
    if [ -f "${_CARGO_HOME}/env" ]; then
        echo "Sourcing Rust environment (cargo not in PATH)..."
        # shellcheck source=/dev/null
        . "${_CARGO_HOME}/env"
    fi
fi
# Prefer rustup-managed toolchain over any older system one.
if [ -d "${_CARGO_HOME}/bin" ]; then
    case ":${PATH}:" in
        *":${_CARGO_HOME}/bin:"*) : ;;
        *) export PATH="${_CARGO_HOME}/bin:${PATH}" ;;
    esac
fi
unset _CARGO_HOME
if ! command -v cargo >/dev/null 2>&1; then
    echo ""
    echo "ERROR: cargo (Rust) not found. tokenizers-cpp requires Rust >= 1.79."
    echo "  Run ./install_dependencies.sh (installs rustup into ~/.cargo)."
    echo ""
    exit 1
fi
_rustc_ver="$(rustc --version 2>/dev/null | awk '{print $2}')"
_rustc_major="${_rustc_ver%%.*}"
_rustc_minor="${_rustc_ver#*.}"; _rustc_minor="${_rustc_minor%%.*}"
if [ -z "${_rustc_major}" ] || [ -z "${_rustc_minor}" ] \
   || [ "${_rustc_major}" -lt 1 ] \
   || { [ "${_rustc_major}" -eq 1 ] && [ "${_rustc_minor}" -lt 79 ]; }; then
    echo ""
    echo "ERROR: rustc ${_rustc_ver:-<unknown>} is too old for tokenizers-cpp"
    echo "  (monostate requires rustc >= 1.79)."
    echo "  Run ./install_dependencies.sh to install a newer toolchain into"
    echo "  \$HOME/.cargo (no sudo required)."
    echo ""
    exit 1
fi
unset _rustc_ver _rustc_major _rustc_minor

# Check for Drogon (system-wide, Homebrew, or deps prefix from install_dependencies.sh)
DROGON_FOUND=0
if pkg-config --exists drogon 2>/dev/null; then
    DROGON_FOUND=1
else
    for _drogon_cfg in \
        "${HOME}/.local/lib/cmake/Drogon/DrogonConfig.cmake" \
        "${HOME}/.local/lib64/cmake/Drogon/DrogonConfig.cmake" \
        "/usr/local/lib/cmake/Drogon/DrogonConfig.cmake" \
        "/usr/lib/cmake/Drogon/DrogonConfig.cmake" \
        "/opt/homebrew/lib/cmake/Drogon/DrogonConfig.cmake"
    do
        if [ -f "${_drogon_cfg}" ]; then
            DROGON_FOUND=1
            break
        fi
    done
fi

if [ "${DROGON_FOUND}" -eq 0 ]; then
    echo ""
    echo "ERROR: Drogon framework not found."
    echo "  Run ./install_dependencies.sh first (installs Drogon + JsonCpp"
    echo "  + libuuid into \$HOME/.local by default, no sudo required)."
    echo "  Or install system-wide: sudo apt install libdrogon-dev"
    echo ""
    exit 1
fi

# ---------------------------------------------------------------------------
# Pre-fetch tokenizer files for all supported models
# ---------------------------------------------------------------------------
TOKENIZER_DIR="${SCRIPT_DIR}/tokenizers"
mkdir -p "${TOKENIZER_DIR}"

HF_TOKEN_RESOLVED="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [ -z "${HF_TOKEN_RESOLVED}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
    HF_TOKEN_RESOLVED=$(cat "${HOME}/.cache/huggingface/token")
fi

download_tokenizer() {
    local model_name="$1"
    local hf_repo="$2"
    local requires_auth="$3"

    local model_dir="${TOKENIZER_DIR}/${model_name}"
    local tok_json="${model_dir}/tokenizer.json"
    local tok_config="${model_dir}/tokenizer_config.json"

    # Skip download if tokenizer files already exist (faster rebuilds, no HF_TOKEN needed)
    if [ -f "${tok_json}" ] && [ -f "${tok_config}" ]; then
        echo "  Using existing ${model_name} tokenizer."
        return 0
    fi

    if [ "${requires_auth}" = "true" ] && [ -z "${HF_TOKEN_RESOLVED}" ]; then
        echo "  Skipping ${model_name} (gated model — set HF_TOKEN to download)."
        return 0
    fi

    local wget_args=()
    if [ "${requires_auth}" = "true" ] && [ -n "${HF_TOKEN_RESOLVED}" ]; then
        wget_args=(--header "Authorization: Bearer ${HF_TOKEN_RESOLVED}")
    fi

    mkdir -p "${model_dir}"

    echo "Downloading ${model_name} tokenizer..."
    if wget -q "${wget_args[@]}" -O "${tok_json}" "${hf_repo}/tokenizer.json" 2>&1; then
        echo "  tokenizer.json downloaded to ${tok_json}"
    else
        rm -f "${tok_json}"
        echo "  ERROR: Failed to download ${model_name} tokenizer.json."
        echo "  URL: ${hf_repo}/tokenizer.json"
        if [ "${requires_auth}" = "true" ]; then
            echo "  This is a gated model. Make sure you have:"
            echo "    1. A valid HF_TOKEN set in your environment"
            echo "    2. Accepted the model license at https://huggingface.co/${model_name}"
        fi
        echo "  Debug: wget ${wget_args[*]} -S -O /dev/null ${hf_repo}/tokenizer.json"
        return 1
    fi

    if wget -q "${wget_args[@]}" -O "${tok_config}" "${hf_repo}/tokenizer_config.json" 2>&1; then
        echo "  tokenizer_config.json downloaded to ${tok_config}"
    else
        rm -f "${tok_config}"
        echo "  ERROR: Failed to download ${model_name} tokenizer_config.json."
        return 1
    fi
}

echo ""
echo "Pre-fetching tokenizer files for supported models..."

# DeepSeek R1-0528 (public, no auth) — required for default build
download_tokenizer \
    "deepseek-ai/DeepSeek-R1-0528" \
    "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/raw/main" \
    "false"

# Llama 3.1 8B Instruct (gated, requires HF_TOKEN)
download_tokenizer \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/raw/main" \
    "true"

echo ""

# TT_METAL_HOME: enables Metal C++ API includes and intellisense
# TT-metal headers use the reflect library which requires Clang (fails with GCC).
# Match tt-metal build_metal.sh: use same toolchain (clang-20) and Ninja when available.
USE_METAL_TOOLCHAIN=0
TOOLCHAIN_PATH=""
CMAKE_GENERATOR=""
if [ -n "${TT_METAL_HOME}" ]; then
    if [ -d "${TT_METAL_HOME}/tt_metal/api" ]; then
        echo "TT_METAL_HOME: ${TT_METAL_HOME} (Metal C++ API enabled)"
        METAL_TOOLCHAIN="${TT_METAL_HOME}/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake"
        if [ -f "${METAL_TOOLCHAIN}" ] && [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
            if command -v clang++-20 >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1; then
                USE_METAL_TOOLCHAIN=1
                TOOLCHAIN_PATH="${METAL_TOOLCHAIN}"
                CMAKE_GENERATOR="Ninja"
                echo "Using tt-metal toolchain: ${TOOLCHAIN_PATH} (clang-20, Ninja)"
            fi
        fi
        if [ "${USE_METAL_TOOLCHAIN}" -eq 0 ]; then
            if command -v clang++ >/dev/null 2>&1; then
                echo "Using fallback: clang++ (tt-metal toolchain not used)"
            else
                echo ""
                echo "ERROR: TT_METAL_HOME is set but no suitable Clang found."
                echo "  tt-metal headers require Clang (reflect library is incompatible with GCC)."
                echo "  For x86_64 Linux: install clang-20 to match tt-metal build"
                echo "    e.g. https://apt.llvm.org/ or build_metal.sh's toolchain"
                echo "  Or install generic clang: sudo apt install clang"
                echo "  Or unset TT_METAL_HOME to build without Metal C++ API."
                echo ""
                exit 1
            fi
        fi
    else
        echo "WARNING: TT_METAL_HOME set but tt_metal/api not found at ${TT_METAL_HOME}/tt_metal/api"
    fi
else
    echo "TT_METAL_HOME not set - building without tt-metal (mock device backend only)"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"

# Configure (use -B/-S for explicit paths to avoid ambiguity)
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DSANITIZE_THREAD="${SANITIZE_THREAD}"
    -DSANITIZE_ADDRESS="${SANITIZE_ADDRESS}"
    -DENABLE_TRACY="${ENABLE_TRACY}"
    -DENABLE_BLAZE="${ENABLE_BLAZE}"
    -DCLANG_TIDY="${CLANG_TIDY}"
    -DKAFKA_ENABLED="${KAFKA_ENABLED}"
)
[ -n "${TT_METAL_HOME}" ] && CMAKE_ARGS+=(-DTT_METAL_HOME="${TT_METAL_HOME}")

# Compiler/toolchain: --cxx-compiler-path overrides --toolchain-path overrides auto-detection (match build_metal.sh)
if [ -n "${CXX_COMPILER_PATH}" ]; then
    echo "Using C++ compiler: ${CXX_COMPILER_PATH}"
    CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="${CXX_COMPILER_PATH}")
    command -v ninja >/dev/null 2>&1 && CMAKE_ARGS+=(-G Ninja)
elif [ -n "${TOOLCHAIN_PATH_ARG}" ]; then
    echo "Using toolchain: ${TOOLCHAIN_PATH_ARG}"
    CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_PATH_ARG}")
    command -v ninja >/dev/null 2>&1 && CMAKE_ARGS+=(-G Ninja)
elif [ -n "${TOOLCHAIN_PATH}" ]; then
    CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_PATH}" -G "${CMAKE_GENERATOR}")
elif [ -n "${TT_METAL_HOME}" ] && [ -d "${TT_METAL_HOME}/tt_metal/api" ]; then
    CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang)
fi

echo ""
echo "Configuring CMake..."
cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" "${CMAKE_ARGS[@]}"

# Symlink compile_commands.json to project root for intellisense (clangd, VSCode C++)
if [ -f "${BUILD_DIR}/compile_commands.json" ]; then
    ln -sf "${BUILD_DIR}/compile_commands.json" "${SCRIPT_DIR}/compile_commands.json"
fi

# Build (cmake --build works with any generator: Makefiles or Ninja)
echo ""
echo "Building..."
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
cmake --build "${BUILD_DIR}" -j"${NPROC}"

echo ""
echo "=============================================="
echo "  Build complete!"
echo "  Binary: ${BUILD_DIR}/tt_media_server_cpp"
echo "=============================================="
echo ""
echo "Run with: ./build/tt_media_server_cpp [options]"
echo "  -h, --host HOST     Listen host (default: 0.0.0.0)"
echo "  -p, --port PORT     Listen port (default: 8000)"
echo "  -t, --threads N     Number of IO threads"
echo ""
