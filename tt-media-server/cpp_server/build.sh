#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="Release"

# Parse arguments
SANITIZE_THREAD="OFF"
SANITIZE_ADDRESS="OFF"
TOOLCHAIN_PATH_ARG=""
CXX_COMPILER_PATH=""
MODEL_TYPE="DEEPSEEK_V3"
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
        --toolchain-path)
            TOOLCHAIN_PATH_ARG="$2"
            shift 2
            ;;
        --cxx-compiler-path)
            CXX_COMPILER_PATH="$2"
            shift 2
            ;;
        --model)
            case "$2" in
                deepseek-ai/DeepSeek-V3|DeepSeek-V3|deepseek)
                    MODEL_TYPE="DEEPSEEK_V3"
                    ;;
                meta-llama/Llama-3.1-8B|Llama-3.1-8B|llama)
                    MODEL_TYPE="LLAMA_3_1_8B"
                    ;;
                *)
                    echo "Unknown model: $2"
                    echo "Supported: deepseek-ai/DeepSeek-V3 (default), meta-llama/Llama-3.1-8B"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug              Build in Debug mode (default: Release)"
            echo "  --tsan               Build with ThreadSanitizer for data-race detection"
            echo "  --asan               Build with AddressSanitizer + LeakSanitizer for memory/leak detection"
            echo "  --toolchain-path P   Use CMake toolchain file (overrides TT_METAL_HOME toolchain)"
            echo "  --cxx-compiler-path P  Set C++ compiler (overrides toolchain)"
            echo "  --model MODEL        Target model: deepseek-ai/DeepSeek-V3 (default) or meta-llama/Llama-3.1-8B"
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

if [ "${MODEL_TYPE}" = "DEEPSEEK_V3" ]; then
    MODEL_DISPLAY="deepseek-ai/DeepSeek-V3"
else
    MODEL_DISPLAY="meta-llama/Llama-3.1-8B"
fi

echo "=============================================="
echo "  Building TT Media Server (C++ Drogon)"
echo "  Model: ${MODEL_DISPLAY}"
echo "  Build type: ${BUILD_TYPE}"
echo "  ThreadSanitizer: ${SANITIZE_THREAD}"
echo "  AddressSanitizer: ${SANITIZE_ADDRESS}"
echo "=============================================="

# Ensure cargo (Rust) is in PATH for tokenizers-cpp
if ! command -v cargo >/dev/null 2>&1; then
    if [ -f "${HOME}/.cargo/env" ]; then
        echo "Sourcing Rust environment (cargo not in PATH)..."
        # shellcheck source=/dev/null
        . "${HOME}/.cargo/env"
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        echo ""
        echo "ERROR: cargo (Rust) not found. tokenizers-cpp requires Rust."
        echo "  Install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo "  Then: source ~/.cargo/env  (or start a new terminal)"
        echo ""
        exit 1
    fi
fi

# Check for Drogon
DROGON_FOUND=0
if pkg-config --exists drogon 2>/dev/null; then
    DROGON_FOUND=1
elif [ -f "/usr/local/lib/cmake/Drogon/DrogonConfig.cmake" ]; then
    DROGON_FOUND=1
elif [ -f "/usr/lib/cmake/Drogon/DrogonConfig.cmake" ]; then
    DROGON_FOUND=1
elif [ -f "/opt/homebrew/lib/cmake/Drogon/DrogonConfig.cmake" ]; then
    DROGON_FOUND=1
fi

if [ "${DROGON_FOUND}" -eq 0 ]; then
    echo ""
    echo "Drogon not found. Installing dependencies..."
    echo ""

    # Check if we need to build Drogon from deps
    DROGON_DIR="${SCRIPT_DIR}/deps/drogon"
    if [ -d "${DROGON_DIR}" ]; then
        echo "Building Drogon from ${DROGON_DIR}..."
        mkdir -p "${DROGON_DIR}/build"
        cd "${DROGON_DIR}/build"
        cmake -DCMAKE_BUILD_TYPE=Release \
              -DBUILD_EXAMPLES=OFF \
              -DBUILD_CTL=OFF \
              -DBUILD_YAML_CONFIG=OFF \
              ..
        NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
        make -j"${NPROC}"
        sudo make install
        if [ "$(uname -s)" = "Linux" ]; then
            sudo ldconfig
        fi
        cd "${SCRIPT_DIR}"
    else
        echo "Please install Drogon framework first:"
        echo "  Ubuntu/Debian: sudo apt install libdrogon-dev"
        echo "  Or build from source: https://github.com/drogonframework/drogon"
        exit 1
    fi
fi

# Download tokenizer and tokenizer_config if not present (or if model changed)
TOKENIZER_DIR="${SCRIPT_DIR}/tokenizers"
TOKENIZER_JSON="${TOKENIZER_DIR}/tokenizer.json"
TOKENIZER_CONFIG_JSON="${TOKENIZER_DIR}/tokenizer_config.json"
MODEL_MARKER="${TOKENIZER_DIR}/.model"

mkdir -p "${TOKENIZER_DIR}"

# Re-download if model changed since last build
if [ -f "${MODEL_MARKER}" ]; then
    PREV_MODEL=$(cat "${MODEL_MARKER}")
    if [ "${PREV_MODEL}" != "${MODEL_TYPE}" ]; then
        echo "Model changed (${PREV_MODEL} -> ${MODEL_TYPE}). Re-downloading tokenizer files..."
        rm -f "${TOKENIZER_JSON}" "${TOKENIZER_CONFIG_JSON}"
    fi
fi

if [ "${MODEL_TYPE}" = "DEEPSEEK_V3" ]; then
    HF_REPO="https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main"
else
    HF_REPO="https://huggingface.co/meta-llama/Llama-3.1-8B/raw/main"
fi

HF_TOKEN_RESOLVED="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [ -z "${HF_TOKEN_RESOLVED}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
    HF_TOKEN_RESOLVED=$(cat "${HOME}/.cache/huggingface/token")
fi
WGET_AUTH_ARGS=()
if [ -n "${HF_TOKEN_RESOLVED}" ]; then
    WGET_AUTH_ARGS=(--header "Authorization: Bearer ${HF_TOKEN_RESOLVED}")
fi

if [ ! -f "${TOKENIZER_JSON}" ]; then
    echo ""
    echo "Tokenizer not found. Downloading ${MODEL_DISPLAY} tokenizer..."
    if wget -q "${WGET_AUTH_ARGS[@]}" -O "${TOKENIZER_JSON}" "${HF_REPO}/tokenizer.json"; then
        echo "Tokenizer downloaded successfully to ${TOKENIZER_JSON}"
    else
        rm -f "${TOKENIZER_JSON}"
        echo ""
        echo "ERROR: Failed to download tokenizer. tokenizer.json is required for build and tests."
        if [ "${MODEL_TYPE}" = "LLAMA_3_1_8B" ]; then
            echo "  Llama models are gated on HuggingFace. Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) and retry."
        else
            echo "  Check network access to huggingface.co"
        fi
        echo "  Manual: wget -O ${TOKENIZER_JSON} ${HF_REPO}/tokenizer.json"
        echo ""
        exit 1
    fi
    echo ""
fi

if [ ! -f "${TOKENIZER_CONFIG_JSON}" ]; then
    echo ""
    echo "Tokenizer config not found. Downloading tokenizer_config.json..."
    if wget -q "${WGET_AUTH_ARGS[@]}" -O "${TOKENIZER_CONFIG_JSON}" "${HF_REPO}/tokenizer_config.json"; then
        echo "Tokenizer config downloaded successfully to ${TOKENIZER_CONFIG_JSON}"
    else
        rm -f "${TOKENIZER_CONFIG_JSON}"
        echo "Warning: Failed to download tokenizer_config.json. Chat template may use defaults."
        echo "  Manual: wget -O cpp_server/tokenizers/tokenizer_config.json ${HF_REPO}/tokenizer_config.json"
    fi
    echo ""
fi

# Ensure tokenizer.json exists before proceeding (download block above exits on failure)
if [ ! -f "${TOKENIZER_JSON}" ]; then
    echo "ERROR: tokenizer.json missing at ${TOKENIZER_JSON}"
    exit 1
fi

echo "${MODEL_TYPE}" > "${MODEL_MARKER}"

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
    -DLLM_ENGINE_DEBUG_BUILD=OFF
    -DSANITIZE_THREAD="${SANITIZE_THREAD}"
    -DSANITIZE_ADDRESS="${SANITIZE_ADDRESS}"
    -DMODEL_TYPE="${MODEL_TYPE}"
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
