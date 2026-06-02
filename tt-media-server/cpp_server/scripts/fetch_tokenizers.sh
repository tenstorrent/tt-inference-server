#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# Pre-fetch tokenizer assets for all supported models into a target directory,
# laid out as:
#   <target>/<hf_model_id>/{config.json, tokenizer_config.json,
#                           tokenizer.json | tiktoken.model[, chat_template.jinja]}
#
# Single source of truth shared by:
#   * cpp_server/build.sh                  -> bakes assets into the worker image
#                                             at cpp_server/tokenizers
#   * dynamo_frontend/Dockerfile.frontend  -> bakes the SAME assets into the
#                                             frontend image at the absolute path
#                                             the worker advertises in its MDC
# Keeping the fetch in one place guarantees both images carry identical files,
# so the Dynamo frontend resolves the worker's MDC paths locally instead of
# treating them as HuggingFace repo ids (which 404).
#
# Usage: fetch_tokenizers.sh [TARGET_DIR]
#   TARGET_DIR  destination tokenizers root. Defaults to the TOKENIZER_DIR env
#               var, else <this-script>/../tokenizers (i.e. cpp_server/tokenizers).
#               The positional arg wins over the env var.
#
# Requires: wget. Honors HF_TOKEN / HUGGING_FACE_HUB_TOKEN (or
# ~/.cache/huggingface/token) for gated models.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKENIZER_DIR="${1:-${TOKENIZER_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)/tokenizers}}"
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

echo "Pre-fetching tokenizer files for supported models into ${TOKENIZER_DIR}..."

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
# Uses /resolve/main for tiktoken.model (LFS file) but /raw/main for text files
download_tokenizer \
    "moonshotai/Kimi-K2.6" \
    "https://huggingface.co/moonshotai/Kimi-K2.6/resolve/main" \
    "false" \
    '{"model_type":"kimi_k25","architectures":["KimiK25ForConditionalGeneration"]}' \
    "tiktoken"

echo ""
