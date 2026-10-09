#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# Build the DeepSeek-V4.1-Flash (Blackhole Galaxy) vLLM server image from LOCAL,
# possibly unpushed, checkouts of tt-metal, tt-inference-server and vllm-tt-plugin.
# Uses vllm-tt-metal/vllm.tt-metal.local.Dockerfile (COPY of git-archive exports instead
# of git clone). Needs network (PyPI, GitHub, ghcr.io, download.pytorch.org), no device.
#
# Usage: scripts/build_local_dsv41_image.sh [--skip-base]
# Env overrides: TT_METAL_DIR, TTIS_DIR, PLUGIN_DIR, CTX_ROOT, BASE_TAG, IMAGE_REPO
#
# Each repo is exported at the HEAD commit read at start (uncommitted changes are NOT
# included); the three shas are recorded in /home/container_app_user/BUILD_INFO.
set -euo pipefail

TT_METAL_DIR=${TT_METAL_DIR:-/mnt/tt-data/ssinghal/tests/tt-metal}
TTIS_DIR=${TTIS_DIR:-/mnt/tt-data/ssinghal/tests/tt-inference-server}
PLUGIN_DIR=${PLUGIN_DIR:-/mnt/tt-data/ssinghal/wt/plugin_sampled}
CTX_ROOT=${CTX_ROOT:-/mnt/tt-data/ssinghal/dsv4-docker-ctx}
IMAGE_REPO=${IMAGE_REPO:-dsv41-flash-local}
SKIP_BASE=0
[[ "${1:-}" == "--skip-base" ]] && SKIP_BASE=1

METAL_SHA=$(git -C "$TT_METAL_DIR" rev-parse HEAD)
TTIS_SHA=$(git -C "$TTIS_DIR" rev-parse HEAD)
PLUGIN_SHA=$(git -C "$PLUGIN_DIR" rev-parse HEAD)
TAG="${METAL_SHA:0:11}-${TTIS_SHA:0:8}-${PLUGIN_SHA:0:7}"
IMAGE="${IMAGE_REPO}:${TAG}"
BASE_TAG=${BASE_TAG:-tt-metalium-ci-build:local-${METAL_SHA:0:11}}
CTX="${CTX_ROOT}/${TAG}"
echo "tt-metal=$METAL_SHA tt-inference-server=$TTIS_SHA vllm-tt-plugin=$PLUGIN_SHA"
echo "image=$IMAGE base=$BASE_TAG ctx=$CTX"

# ---- 1. export sources (no .git, no untracked/ignored files, no build artifacts) ----
rm -rf "$CTX"; mkdir -p "$CTX/tt-metal" "$CTX/vllm-tt-plugin" "$CTX/tis"
git -C "$TT_METAL_DIR" archive "$METAL_SHA" | tar -x -C "$CTX/tt-metal"
# git archive skips submodules: export each at the commit recorded in the superproject
git -C "$TT_METAL_DIR" ls-tree -r "$METAL_SHA" | awk '$2=="commit"{print $3, $4}' | while read -r sha path; do
    mkdir -p "$CTX/tt-metal/$path"
    git -C "$TT_METAL_DIR/$path" archive "$sha" | tar -x -C "$CTX/tt-metal/$path"
done
git -C "$PLUGIN_DIR" archive "$PLUGIN_SHA" | tar -x -C "$CTX/vllm-tt-plugin"
git -C "$TTIS_DIR" archive "$TTIS_SHA" | tar -x -C "$CTX/tis"
# tt-inference-server pieces the image needs, plus the model catalog (dev env: the
# DeepSeek-V4.1-Flash entry lives in workflows/model_specs/dev)
mkdir -p "$CTX/vllm-tt-metal"
cp -a "$CTX/tis/vllm-tt-metal/src" "$CTX/vllm-tt-metal/src"
cp -a "$CTX/tis/vllm-tt-metal/requirements.txt" "$CTX/vllm-tt-metal/"
cp -a "$CTX/tis/utils" "$CTX/tis/VERSION" "$CTX/"
(cd "$CTX/tis" && MODEL_SPECS_ENV=dev python3 -c "
from pathlib import Path
from workflows.model_spec import MODEL_SPECS, export_model_specs_json
n = export_model_specs_json(MODEL_SPECS, Path('$CTX/model_spec.json'))
print('model_spec.json:', n, 'specs')")
cp "$CTX/tis/vllm-tt-metal/vllm.tt-metal.local.Dockerfile" "$CTX/Dockerfile"
rm -rf "$CTX/tis"
du -sh "$CTX"

# ---- 2. base (toolchain) image from the tt-metal export's own dockerfile/docker-bake.hcl ----
if [[ $SKIP_BASE == 0 ]] && ! docker image inspect "$BASE_TAG" >/dev/null 2>&1; then
    (cd "$CTX/tt-metal" && UBUNTU_VERSION=22.04 docker buildx bake -f dockerfile/docker-bake.hcl \
        --set "ci-build.tags=$BASE_TAG" --load ci-build)
fi

# ---- 3. the image ----
docker build -t "$IMAGE" -f "$CTX/Dockerfile" \
    --build-arg TT_METAL_DOCKERFILE_URL="$BASE_TAG" \
    --build-arg TT_METAL_COMMIT_SHA_OR_TAG="$METAL_SHA" \
    --build-arg TT_METAL_SCM_VERSION="0.0.0.dev0+g${METAL_SHA:0:11}" \
    --build-arg TT_INFERENCE_SERVER_COMMIT="$TTIS_SHA" \
    --build-arg VLLM_TT_PLUGIN_COMMIT="$PLUGIN_SHA" \
    "$CTX"
echo "BUILT $IMAGE"
