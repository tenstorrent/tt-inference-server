#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="Release"
ENABLE_TTNN="OFF"

# Parse arguments
TEST="OFF"
SANITIZE_THREAD="OFF"
SANITIZE_ADDRESS="OFF"
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --ttnn)
            ENABLE_TTNN="ON"
            shift
            ;;
        --test)
            TEST="ON"
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
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug    Build in Debug mode (default: Release)"
            echo "  --ttnn     Enable TTNN test runner (requires Python + ttnn)"
            echo "  --test     Build for PR gate: LLM only (no Python required)"
            echo "  --tsan     Build with ThreadSanitizer for data-race detection"
            echo "  --asan     Build with AddressSanitizer + LeakSanitizer for memory/leak detection"
            echo "  --help     Show this help message"
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
echo "  TTNN enabled: ${ENABLE_TTNN}"
echo "  Test build: ${TEST}"
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

# If TTNN is enabled, ensure we have the right Python
if [ "${ENABLE_TTNN}" = "ON" ]; then
    if [ -z "${VIRTUAL_ENV}" ]; then
        echo ""
        echo "WARNING: No VIRTUAL_ENV detected. For TTNN support, you should"
        echo "         activate the tt-metal Python environment first:"
        echo "         source /path/to/tt-metal/python_env/bin/activate"
        echo ""
    else
        echo "Using Python from: ${VIRTUAL_ENV}"
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

# Download tokenizer and tokenizer_config if not present
TOKENIZER_DIR="${SCRIPT_DIR}/tokenizers"
TOKENIZER_JSON="${TOKENIZER_DIR}/tokenizer.json"
TOKENIZER_CONFIG_JSON="${TOKENIZER_DIR}/tokenizer_config.json"
HF_DEEPSEEK_REPO="https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main"

mkdir -p "${TOKENIZER_DIR}"

if [ ! -f "${TOKENIZER_JSON}" ]; then
    echo ""
    echo "Tokenizer not found. Downloading DeepSeek V3 tokenizer..."
    if wget -q -O "${TOKENIZER_JSON}" "${HF_DEEPSEEK_REPO}/tokenizer.json"; then
        echo "Tokenizer downloaded successfully to ${TOKENIZER_JSON}"
    else
        echo "Warning: Failed to download tokenizer. You can manually download it later:"
        echo "  mkdir -p cpp_server/tokenizers"
        echo "  wget -O cpp_server/tokenizers/tokenizer.json ${HF_DEEPSEEK_REPO}/tokenizer.json"
    fi
    echo ""
fi

if [ ! -f "${TOKENIZER_CONFIG_JSON}" ]; then
    echo ""
    echo "Tokenizer config not found. Downloading tokenizer_config.json..."
    if wget -q -O "${TOKENIZER_CONFIG_JSON}" "${HF_DEEPSEEK_REPO}/tokenizer_config.json"; then
        echo "Tokenizer config downloaded successfully to ${TOKENIZER_CONFIG_JSON}"
    else
        echo "Warning: Failed to download tokenizer_config.json. Chat template may use defaults."
        echo "  wget -O cpp_server/tokenizers/tokenizer_config.json ${HF_DEEPSEEK_REPO}/tokenizer_config.json"
    fi
    echo ""
fi

# Create build directory
mkdir -p "${BUILD_DIR}"

# Configure (use -B/-S for explicit paths to avoid ambiguity)
echo ""
echo "Configuring CMake..."
cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DENABLE_TTNN="${ENABLE_TTNN}" \
      -DLLM_ENGINE_DEBUG_BUILD=OFF \
      -DTEST="${TEST}" \
      -DSANITIZE_THREAD="${SANITIZE_THREAD}" \
      -DSANITIZE_ADDRESS="${SANITIZE_ADDRESS}"

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
if [ -f "${BUILD_DIR}/engine_demo" ] 2>/dev/null; then
    echo "LLM engine demo: ./build/engine_demo"
    echo ""
fi
echo "Run with: ./build/tt_media_server_cpp [options]"
echo "  -h, --host HOST     Listen host (default: 0.0.0.0)"
echo "  -p, --port PORT     Listen port (default: 8000)"
echo "  -t, --threads N     Number of IO threads"
echo ""
if [ "${ENABLE_TTNN}" = "ON" ]; then
    echo "TTNN runner enabled. Set TT_RUNNER_TYPE=ttnn_test to use it."
    echo ""
fi
