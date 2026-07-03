#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Build the comfyui-media-server Docker image.
#
# Usage:
#   ./build_image.sh [image_tag] [tt_metal_sha]
#
# Defaults:
#   image_tag    = comfyui-media-server:dev
#   tt_metal_sha = the ARG default baked into the Dockerfile (the amalgamation SHA)
#
# This is LONG (builds tt-metal from source). For fast local iteration, use
# launch_server.sh directly instead — see LOCAL_TESTING.md (Path 1).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${1:-comfyui-media-server:dev}"
TT_METAL_SHA="${2:-}"

BUILD_ARGS=()
if [ -n "${TT_METAL_SHA}" ]; then
    BUILD_ARGS+=(--build-arg "TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_SHA}")
fi

echo "Building ${IMAGE_TAG} (context: ${SCRIPT_DIR})"
echo "This builds tt-metal from source and will take a long time."
exec docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${BUILD_ARGS[@]}" \
    -t "${IMAGE_TAG}" \
    "${SCRIPT_DIR}"
