#!/bin/bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SPEC_JSON_HOST=$(cd "$REPO" && "$REPO/venv/bin/python" gen_runtime_spec.py whisper-large-v3 p150 --device-id 2)
SPEC_JSON_CONTAINER="/home/container_app_user/model_specs/$(basename "$SPEC_JSON_HOST")"

docker rm -f whisper-large-v3 2>/dev/null || true

docker run -d \
  --name whisper-large-v3 \
  --env-file "$REPO/.env" \
  --ipc host \
  --publish 0.0.0.0:7001:8000 \
  --device /dev/tenstorrent/2:/dev/tenstorrent/2 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_whisper-whisper-large-v3:/home/container_app_user/cache_root \
  --mount "type=bind,src=${SPEC_JSON_HOST},dst=${SPEC_JSON_CONTAINER},readonly" \
  --mount "type=bind,src=${REPO}/benchmarking,dst=/home/container_app_user/app/benchmarking" \
  --mount "type=bind,src=${REPO}/evals,dst=/home/container_app_user/app/evals" \
  --mount "type=bind,src=${REPO}/utils,dst=/home/container_app_user/app/utils" \
  --mount "type=bind,src=${REPO}/tests,dst=/home/container_app_user/app/tests" \
  --mount "type=bind,src=${REPO}/tt-media-server,dst=/home/container_app_user/tt-metal/server" \
  -e CACHE_ROOT=/home/container_app_user/cache_root \
  -e MODEL=whisper-large-v3 \
  -e DEVICE=p150 \
  -e "RUNTIME_MODEL_SPEC_JSON_PATH=${SPEC_JSON_CONTAINER}" \
  ghcr.io/tenstorrent/tt-media-inference-server:qb2_launch-2508216

docker network connect tt_studio_network whisper-large-v3
echo "whisper-large-v3 started on host port 7001 (container port 8000)"
docker logs -f whisper-large-v3
