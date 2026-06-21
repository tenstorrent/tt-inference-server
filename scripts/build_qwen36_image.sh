#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the Qwen3.6-27B serving image: from-source dev base (tt-metal padfix baked from
# source) + the one scoped tt_vllm_plugin layer. No padfix COPY overlay.
#
# Prereq: the from-source base must exist, built from the pushed branch:
#   tt-inference-server/scripts/build_single_docker.sh --build \
#       --tt-metal-commit 09f1527c21f --vllm-commit 8f3691068 --ubuntu-version 22.04
#
# Stages a minimal build context (just the plugin dir) so the daemon does not tar the
# multi-GB tt-metal tree. Run from anywhere inside the tt-metal checkout.
#
# Usage: tt-inference-server/scripts/build_qwen36_image.sh [TAG]
set -euo pipefail

TT_METAL_ROOT=$(git rev-parse --show-toplevel)
TIS="${TT_METAL_ROOT}/tt-inference-server"
TAG="${1:-ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.15.0-qwen36}"

ctx=$(mktemp -d /tmp/qwen36_img.XXXXXX)
trap 'rm -rf "$ctx"' EXIT

cp -r "${TIS}/tt-vllm-plugin" "${ctx}/tt-vllm-plugin"
cp "${TIS}/Dockerfile.qwen36" "${ctx}/Dockerfile"

echo "Building ${TAG} (context $(du -sh "$ctx" | cut -f1)) ..."
docker build -t "${TAG}" "${ctx}"
echo "✅ built ${TAG}"
