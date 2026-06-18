#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the Qwen3.6-27B VL (image+video) serving image as a PYTHON-ONLY overlay on the
# from-source dev base. All qwen3.6 VL changes since the base SHA are Python-only, so this
# layers files instead of a multi-hour from-source rebuild. Everything copied is committed
# tt-metal code (so the image is reproducible from source control).
#
# Usage: tt-inference-server/scripts/build_qwen36_vl_image.sh [TAG]
# Run from anywhere inside the tt-metal checkout.
set -euo pipefail

# Resolve paths from the script location (tt-inference-server is a NESTED git repo, so
# `git rev-parse --show-toplevel` is unreliable here). scripts/ -> TIS -> tt-metal root.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIS=$(cd "${SCRIPT_DIR}/.." && pwd)
TT_METAL_ROOT=$(cd "${TIS}/.." && pwd)
TAG="${1:-ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.15.0-qwen36-vl}"

ctx=$(mktemp -d /tmp/qwen36vl_img.XXXXXX)
trap 'rm -rf "$ctx"' EXIT

# Plugin (scoped tt_vllm_plugin layer).
cp -r "${TIS}/tt-vllm-plugin" "${ctx}/tt-vllm-plugin"

# tt-metal Python overlay, mirroring the tt-metal tree under ttm-overlay/.
mkdir -p "${ctx}/ttm-overlay/models/demos" \
         "${ctx}/ttm-overlay/models/tt_transformers/tt" \
         "${ctx}/ttm-overlay/models/tt_dit/parallel"
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
      "${TT_METAL_ROOT}/models/demos/qwen3_6_galaxy_v2/" \
      "${ctx}/ttm-overlay/models/demos/qwen3_6_galaxy_v2/"
# Changed shared tt_transformers files + qwen36 attention.
for f in common.py generator.py model.py model_config.py qwen36_full_attention.py qwen36_gdn_attention.py; do
    cp "${TT_METAL_ROOT}/models/tt_transformers/tt/${f}" "${ctx}/ttm-overlay/models/tt_transformers/tt/${f}"
done
# CCLManager (carries the vision-CCL ping-pong reset fix).
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
      "${TT_METAL_ROOT}/models/tt_dit/" "${ctx}/ttm-overlay/models/tt_dit/"

cp "${TIS}/Dockerfile.qwen36vl" "${ctx}/Dockerfile"

echo "Building ${TAG} (context $(du -sh "$ctx" | cut -f1)) ..."
docker build -t "${TAG}" "${ctx}"
echo "built ${TAG}"
