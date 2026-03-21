#!/usr/bin/env bash
# Patches OLMo fixes into the base Docker image to avoid --dev-mode bind-mounts.
#
# Usage:
#   TT_METAL_HOME=/home/tt-admin/ssinghal/tt-metal bash scripts/patch_olmo_image.sh
#
# Produces: <BASE_IMAGE>-olmo (same tag with -olmo suffix)

set -euo pipefail

BASE_IMAGE="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.11.0-b0c2f63-8f36910"
NEW_IMAGE="${BASE_IMAGE}-olmo"
CONTAINER_NAME="olmo_patch_$$"

TT_METAL_HOME="${TT_METAL_HOME:-/home/tt-admin/ssinghal/tt-metal}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CONTAINER_TT_METAL="/home/container_app_user/tt-metal"
CONTAINER_VLLM="/home/container_app_user/vllm"

echo "=== Patching OLMo fixes into Docker image ==="
echo "Base:   ${BASE_IMAGE}"
echo "New:    ${NEW_IMAGE}"
echo "tt-metal source: ${TT_METAL_HOME}"
echo ""

# 1. Create a stopped container from the base image
echo "[1/4] Creating container..."
docker create --name "${CONTAINER_NAME}" "${BASE_IMAGE}" /bin/bash

# 2. Copy tt_platform_tt.py overlay (OLMo model registration in vLLM)
echo "[2/5] Patching vllm/platforms/tt.py (OLMo model registration)..."
docker cp \
    "${REPO_ROOT}/docker_overlays/tt_platform_tt.py" \
    "${CONTAINER_NAME}:${CONTAINER_VLLM}/vllm/platforms/tt.py"

# 3. Copy run_vllm_api_server.py fix (--override-tt-config instead of --additional-config)
echo "[3/5] Patching app/src/run_vllm_api_server.py (--override-tt-config fix)..."
docker cp \
    "${REPO_ROOT}/vllm-tt-metal/src/run_vllm_api_server.py" \
    "${CONTAINER_NAME}:/home/container_app_user/app/src/run_vllm_api_server.py"

# 4. Copy changed tt-metal model files
echo "[4/5] Patching tt-metal models/demos/llama3_70b_galaxy/..."

GALAXY_SRC="${TT_METAL_HOME}/models/demos/llama3_70b_galaxy"
GALAXY_DST="${CONTAINER_TT_METAL}/models/demos/llama3_70b_galaxy"

# Core tt files with OLMo fixes
TT_FILES=(
    "tt/generator.py"
    "tt/generator_vllm.py"
    "tt/llama_attention.py"
    "tt/llama_ccl.py"
    "tt/llama_mlp.py"
    "tt/llama_model.py"
    "tt/olmo_model_config.py"
    "tt/prefetcher_common.py"
)

for f in "${TT_FILES[@]}"; do
    echo "  Copying ${f}..."
    docker cp "${GALAXY_SRC}/${f}" "${CONTAINER_NAME}:${GALAXY_DST}/${f}"
done

# Demo file and long-ISL prompt data
DEMO_FILES=(
    "demo/demo_olmo_decode.py"
    "demo/sample_prompts/input_data_long_8k.json"
    "demo/sample_prompts/input_data_long_16k.json"
    "demo/sample_prompts/input_data_long_32k.json"
    "demo/sample_prompts/input_data_long_64k.json"
)

for f in "${DEMO_FILES[@]}"; do
    echo "  Copying ${f}..."
    docker cp "${GALAXY_SRC}/${f}" "${CONTAINER_NAME}:${GALAXY_DST}/${f}"
done

# 5. Commit to new image
echo "[5/5] Committing to ${NEW_IMAGE}..."
docker commit \
    --message "OLMo-3.1-32B fixes: prefill trace, QK-norm, CCL seqlens, model registration" \
    "${CONTAINER_NAME}" \
    "${NEW_IMAGE}"

docker rm "${CONTAINER_NAME}"

echo ""
echo "=== Done ==="
echo "New image: ${NEW_IMAGE}"
echo ""
echo "Run with:"
echo "  python run.py --model OLMo-3.1-32B-Think --workflow server --docker-server \\"
echo "    --no-auth --skip-system-sw-validation --tt-device galaxy \\"
echo "    --override-docker-image ${NEW_IMAGE}"
echo ""
echo "Or update model_spec.py to use this image by default (no --override-docker-image needed)."
