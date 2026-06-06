#!/bin/bash
# Gemma-4-31B-it as a Forge LLM on QB2 (2x P300 = p300x2, 4 chips),
# 512 seq len, concurrency 1, served via run.py + the tt-media-server forge image.
#
# The model spec lives in workflows/model_specs/prod/cnn.yaml (weights
# google/gemma-4-31b-it -> model_name "gemma-4-31b-it"). The tt-media-server
# already has built-in support for (VLLMForge_GEMMA4_31B, P300X2): it derives
# max_model_length=512, max_num_seqs=1, max_num_batched_tokens=2560 and mesh
# (1,4) from MODEL=gemma-4-31b-it + DEVICE=p300x2 -- no extra config needed.
#
# Requires:
#   - HF_TOKEN exported (or in $REPO/.env) -- gemma-4 is gated; weights are
#     pulled inside the container on first start.
#   - docker login ghcr.io with a PAT scoped read:packages on tenstorrent/tt-shield
#     (the override image is a tt-shield build that contains gemma-4-31b-it support).
#
# Iterate: edit, re-run. Watch container logs with:
#   docker logs -f $(docker ps -q --filter "publish=8013")
set -eu

PORT=8013
IMAGE="ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:e03b231b1de926cd8f9a0e1a2d39dd1df599f7a7_46a7c96_79778041239"

# Kill any prior server bound to this port.
docker ps -q --filter "publish=${PORT}" | xargs -r docker kill

python3 run.py \
  --model gemma-4-31b-it \
  --tt-device p300x2 \
  --engine forge \
  --impl forge-vllm-plugin \
  --workflow server \
  --docker-server \
  --no-auth \
  --service-port "${PORT}" \
  --override-docker-image "${IMAGE}"
