#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="Release"

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
echo "  Kafka (KAFKA_ENABLED): ${KAFKA_ENABLED}"
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

# ---------------------------------------------------------------------------
# Pre-fetch tokenizer files for all supported models
# ---------------------------------------------------------------------------
TOKENIZER_DIR="${SCRIPT_DIR}/tokenizers"
mkdir -p "${TOKENIZER_DIR}"

HF_TOKEN_RESOLVED="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [ -z "${HF_TOKEN_RESOLVED}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
    HF_TOKEN_RESOLVED=$(cat "${HOME}/.cache/huggingface/token")
fi

# Download a single file from HuggingFace with optional auth.
# Args: dest_path hf_url requires_auth model_name
# Returns 0 on success, 1 on failure.
download_hf_file() {
    local dest="$1"
    local url="$2"
    local requires_auth="$3"
    local model_name="$4"
    local filename
    filename=$(basename "${dest}")

    if [ -f "${dest}" ]; then
        return 0
    fi

    local wget_args=()
    if [ "${requires_auth}" = "true" ] && [ -n "${HF_TOKEN_RESOLVED}" ]; then
        wget_args=(--header "Authorization: Bearer ${HF_TOKEN_RESOLVED}")
    fi

    if wget -q "${wget_args[@]}" -O "${dest}" "${url}" 2>&1; then
        echo "  ${filename} downloaded"
        return 0
    fi

    rm -f "${dest}"
    echo "  ERROR: Failed to download ${model_name}/${filename}."
    echo "  URL: ${url}"
    if [ "${requires_auth}" = "true" ]; then
        echo "  This is a gated model. Make sure you have:"
        echo "    1. A valid HF_TOKEN set in your environment"
        echo "    2. Accepted the model license at https://huggingface.co/${model_name}"
    fi
    return 1
}

# Download tokenizer files for a model.
# Args:
#   model_name          - HF model path (e.g., "deepseek-ai/DeepSeek-R1-0528")
#   hf_repo             - Base URL for raw files
#   requires_auth       - "true" if gated model requiring HF_TOKEN
#   placeholder_config  - Fallback config.json content if HF fetch fails
#   tokenizer_type      - "json" (default) for tokenizer.json, "tiktoken" for tiktoken.model
download_tokenizer() {
    local model_name="$1"
    local hf_repo="$2"
    local requires_auth="$3"
    local placeholder_config="${4:-}"
    local tokenizer_type="${5:-json}"

    local model_dir="${TOKENIZER_DIR}/${model_name}"
    local model_config="${model_dir}/config.json"

    # Determine required files based on tokenizer type
    local required_files=("tokenizer_config.json")
    if [ "${tokenizer_type}" = "tiktoken" ]; then
        required_files+=("tiktoken.model" "chat_template.jinja")
    else
        required_files+=("tokenizer.json")
    fi

    # Check if all required files exist
    local all_exist=true
    for f in "${required_files[@]}"; do
        if [ ! -f "${model_dir}/${f}" ]; then
            all_exist=false
            break
        fi
    done
    if [ "${all_exist}" = "true" ] && [ -f "${model_config}" ]; then
        echo "  Using existing ${model_name} tokenizer + config."
        return 0
    fi

    # Skip gated models without auth token
    if [ "${requires_auth}" = "true" ] && [ -z "${HF_TOKEN_RESOLVED}" ]; then
        echo "  Skipping ${model_name} (gated model — set HF_TOKEN to download)."
        return 0
    fi

    mkdir -p "${model_dir}"
    echo "Downloading ${model_name} tokenizer..."

    # Download required tokenizer files
    for f in "${required_files[@]}"; do
        if ! download_hf_file "${model_dir}/${f}" "${hf_repo}/${f}" "${requires_auth}" "${model_name}"; then
            return 1
        fi
    done

    # Download config.json with placeholder fallback
    if [ ! -f "${model_config}" ]; then
        if download_hf_file "${model_config}" "${hf_repo}/config.json" "${requires_auth}" "${model_name}" 2>/dev/null; then
            :
        elif [ -n "${placeholder_config}" ]; then
            echo "  config.json HF fetch failed; writing minimal placeholder."
            printf '%s\n' "${placeholder_config}" > "${model_config}"
        else
            echo "  WARN: ${model_name} config.json missing and no placeholder supplied."
            echo "  Dynamo frontend will fail with 'unable to extract config.json'."
            return 1
        fi
    fi
}

echo ""
echo "Pre-fetching tokenizer files for supported models..."

# DeepSeek R1-0528 (public, no auth) — required for default build.
# The placeholder mirrors dynamo_frontend/deploy.sh's offline fallback:
# `model_type` is what makes Dynamo's frontend pick a HF transformer
# architecture; `architectures` lets the loader pass its sanity check.
download_tokenizer \
    "deepseek-ai/DeepSeek-R1-0528" \
    "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/raw/main" \
    "false" \
    '{"model_type":"deepseek_v3","architectures":["DeepseekV3ForCausalLM"]}'

# Llama 3.1 8B Instruct (gated, requires HF_TOKEN)
download_tokenizer \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/raw/main" \
    "true" \
    '{"model_type":"llama","architectures":["LlamaForCausalLM"]}'

# Kimi K2.6 (public tiktoken tokenizer + jinja chat template; no tokenizer.json on HF)
download_tokenizer \
    "moonshotai/Kimi-K2.6" \
    "https://huggingface.co/moonshotai/Kimi-K2.6/raw/main" \
    "false" \
    '{"model_type":"kimi_k25","architectures":["KimiK25ForConditionalGeneration"]}' \
    "tiktoken"

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
cmake --fresh -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" "${CMAKE_ARGS[@]}"

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
