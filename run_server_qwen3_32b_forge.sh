#!/bin/bash
# Qwen3-32B as a Forge LLM (tensor-parallel) on QB2 (2x P300 = p300x2, 4 chips),
# 512 seq len, concurrency 1, mirroring run_server_gemma4_31b.sh.
#
# IMPORTANT: this needs a forge image whose tt-media-server includes the
# VLLMForge_QWEN_32B runner (added in this repo's tt-media-server). The pinned
# gemma image below does NOT contain it -- override IMAGE with a forge build that
# does (or dev-mount the updated tt-media-server). See HANDOFF_gemma4_31b_it_forge.md.
#
# Requires:
#   - HF_TOKEN exported (or in $REPO/.env)
#   - docker login ghcr.io with read:packages on tenstorrent/tt-shield
#
# Watch: docker logs -f $(docker ps -q --filter "publish=8014")
set -eu

PORT=8014
IMAGE="ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:e03b231b1de926cd8f9a0e1a2d39dd1df599f7a7_46a7c96_79778041239"

docker ps -q --filter "publish=${PORT}" | xargs -r docker kill

python3 run.py \
  --model Qwen3-32B \
  --tt-device p300x2 \
  --engine forge \
  --impl forge-vllm-plugin \
  --workflow server \
  --docker-server \
  --no-auth \
  --service-port "${PORT}" \
  --override-docker-image "${IMAGE}"
