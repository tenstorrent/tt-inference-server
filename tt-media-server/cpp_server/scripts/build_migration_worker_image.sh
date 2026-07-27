#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CPP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly MEDIA_ROOT="$(cd "${CPP_ROOT}/.." && pwd)"
readonly REPO_ROOT="$(cd "${MEDIA_ROOT}/.." && pwd)"
readonly DOCKERFILE="${MEDIA_ROOT}/Dockerfile.migration-worker"
readonly ENGINE_SHA="$(git -C "${REPO_ROOT}" ls-tree HEAD \
  tt-media-server/cpp_server/tt-llm-engine | awk '{print $3}')"
readonly DEFAULT_ENGINE_IMAGE="ghcr.io/tenstorrent/tt-llm-engine/blaze:${ENGINE_SHA}"

ENGINE_IMAGE="${TT_LLM_ENGINE_IMAGE:-${DEFAULT_ENGINE_IMAGE}}"
IMAGE="${MIGRATION_WORKER_IMAGE:-tt-migration-worker:dev}"
IMAGE_WAS_SET="${MIGRATION_WORKER_IMAGE:+1}"
PUSH=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--engine-image IMAGE] [--image IMAGE] [--push]

Options:
  --engine-image IMAGE  Prebuilt tt-llm-engine dependency image
                        (default: ${DEFAULT_ENGINE_IMAGE})
  --image IMAGE         Output image (default: ${IMAGE})
  --push                Push directly to the configured registry
  -h, --help            Show this help

Environment alternatives:
  TT_LLM_ENGINE_IMAGE, MIGRATION_WORKER_IMAGE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine-image)
      ENGINE_IMAGE="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      IMAGE_WAS_SET=1
      shift 2
      ;;
    --push)
      PUSH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "${ENGINE_SHA}" && -n "${ENGINE_IMAGE}" ]] || {
  echo "ERROR: could not resolve the tt-llm-engine dependency image" >&2
  exit 2
}
if [[ "${PUSH}" -eq 1 && -z "${IMAGE_WAS_SET}" ]]; then
  echo "ERROR: --push requires --image or MIGRATION_WORKER_IMAGE" >&2
  exit 2
fi
command -v docker >/dev/null 2>&1 || {
  echo "ERROR: docker is not installed" >&2
  exit 1
}
docker buildx version >/dev/null

COMMIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
OUTPUT_ARGS=(--load)
[[ "${PUSH}" -eq 1 ]] && OUTPUT_ARGS=(--push)

docker buildx build \
  --file "${DOCKERFILE}" \
  --build-arg "TT_LLM_ENGINE_IMAGE=${ENGINE_IMAGE}" \
  --build-arg "TT_INFERENCE_SERVER_COMMIT_SHA=${COMMIT_SHA}" \
  --tag "${IMAGE}" \
  "${OUTPUT_ARGS[@]}" \
  "${REPO_ROOT}"
