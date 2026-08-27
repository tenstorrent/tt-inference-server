#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build_quetzal_dev_image.sh \
  --base-image IMAGE@sha256:DIGEST \
  --quetzal-commit FULL_COMMIT_SHA \
  --tag OUTPUT_TAG

Builds a uniquely tagged TTIS development image containing a non-editable,
commit-pinned Quetzal wheel. It does not install any model artifact or weights.
EOF
}

base_image=""
quetzal_commit=""
output_tag=""
while (($#)); do
    case "$1" in
        --base-image) base_image="${2:-}"; shift 2 ;;
        --quetzal-commit) quetzal_commit="${2:-}"; shift 2 ;;
        --tag) output_tag="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "${base_image}" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "--base-image must be pinned by an sha256 digest" >&2
    exit 2
fi
if [[ ! "${quetzal_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "--quetzal-commit must be a full 40-character commit SHA" >&2
    exit 2
fi
if [[ -z "${output_tag}" ]]; then
    echo "--tag is required" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec docker build \
    --file "${repo_root}/vllm-tt-metal/vllm.tt-metal.src.quetzal.Dockerfile" \
    --build-arg "TT_INFERENCE_SERVER_BASE_IMAGE=${base_image}" \
    --build-arg "TT_QUETZAL_COMMIT_SHA=${quetzal_commit}" \
    --tag "${output_tag}" \
    "${repo_root}"
