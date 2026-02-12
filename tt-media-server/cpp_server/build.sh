#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="Release"
ENABLE_TTNN="OFF"

# TT-Metal is the main dependency: always use TT_METAL_HOME/build_Release
TT_METAL_HOME="${TT_METAL_HOME:-${SCRIPT_DIR}/deps/tt-metal}"
TT_METAL_BUILD="${TT_METAL_HOME}/build_Release"
if [ ! -d "${TT_METAL_BUILD}" ]; then
    echo "Error: tt-metal build required at ${TT_METAL_BUILD}"
    echo "       Set TT_METAL_HOME or build: cmake -B build_Release -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build_Release"
    exit 1
fi
# Use same compiler as tt-metal (usually Clang) to avoid ABI/header issues
if [ -z "${CXX}" ] && command -v clang++-17 >/dev/null 2>&1; then
    export CXX=clang++-17 CC=clang-17
fi

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
echo "  TT-Metal: ${TT_METAL_BUILD}"
echo "  TTNN enabled: ${ENABLE_TTNN}"
echo "  Test build: ${TEST}"
echo "  ThreadSanitizer: ${SANITIZE_THREAD}"
echo "  AddressSanitizer: ${SANITIZE_ADDRESS}"
echo "=============================================="

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

# Create build directory
mkdir -p "${BUILD_DIR}"

# Configure (use -B/-S for explicit paths to avoid ambiguity)
echo ""
echo "Configuring CMake..."
cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DTT_METAL_HOME="${TT_METAL_HOME}" \
      -DTT_METAL_BUILD="${TT_METAL_BUILD}" \
      -DENABLE_TTNN="${ENABLE_TTNN}" \
      -DLLM_ENGINE_DEBUG_BUILD=ON \
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
echo "For tt-metal mesh device: set TT_METAL_HOME when running (e.g. export TT_METAL_HOME=/path/to/tt-metal)."
echo ""
if [ "${ENABLE_TTNN}" = "ON" ]; then
    echo "TTNN runner enabled. Set TT_RUNNER_TYPE=ttnn_test to use it."
    echo ""
fi
