#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build_quetzal_dev_image.sh \
  --base-image IMAGE@sha256:DIGEST \
  --quetzal-source PATH_TO_CLEAN_GIT_CHECKOUT \
  --quetzal-commit FULL_COMMIT_SHA \
  --tag OUTPUT_TAG

Builds a uniquely tagged TTIS development image containing a non-editable,
commit-pinned Quetzal wheel. It does not install any model artifact or weights.
EOF
}

base_image=""
quetzal_commit=""
quetzal_source=""
output_tag=""
while (($#)); do
    case "$1" in
        --base-image) base_image="${2:-}"; shift 2 ;;
        --quetzal-source) quetzal_source="${2:-}"; shift 2 ;;
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
if [[ -z "${quetzal_source}" ]]; then
    echo "--quetzal-source is required" >&2
    exit 2
fi
if [[ -z "${output_tag}" ]]; then
    echo "--tag is required" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ttis_commit="$(git -C "${repo_root}" rev-parse HEAD)"
if [[ ! "${ttis_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "cannot resolve exact tt-inference-server commit" >&2
    exit 2
fi
quetzal_source="$(git -C "${quetzal_source}" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "--quetzal-source must be a local git checkout" >&2
    exit 2
}
if [[ -n "$(git -C "${quetzal_source}" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "--quetzal-source must have no tracked, staged, or untracked changes" >&2
    exit 2
fi
source_head="$(git -C "${quetzal_source}" rev-parse HEAD)"
if [[ "${source_head}" != "${quetzal_commit}" ]]; then
    echo "--quetzal-source HEAD ${source_head} does not match --quetzal-commit ${quetzal_commit}" >&2
    exit 2
fi
if git -C "${quetzal_source}" ls-tree -r --name-only "${quetzal_commit}" \
        | grep -Fxq '.tt-quetzal-commit'; then
    echo "Quetzal source reserves .tt-quetzal-commit for build identity" >&2
    exit 2
fi

export_root="$(mktemp -d)"
ttis_export_root="$(mktemp -d)"
trap 'rm -rf -- "${export_root}" "${ttis_export_root}"' EXIT
git -C "${quetzal_source}" archive --format=tar "${quetzal_commit}" \
    | tar -xf - -C "${export_root}"
printf '%s\n' "${quetzal_commit}" > "${export_root}/.tt-quetzal-commit"
git -C "${repo_root}" archive --format=tar "${ttis_commit}" \
    vllm-tt-metal/src/run_vllm_api_server.py \
    | tar -xf - -C "${ttis_export_root}"

docker buildx build --load \
    --file "${repo_root}/vllm-tt-metal/vllm.tt-metal.src.quetzal.Dockerfile" \
    --build-context "quetzal_src=${export_root}" \
    --build-context "ttis_src=${ttis_export_root}" \
    --build-arg "TT_INFERENCE_SERVER_BASE_IMAGE=${base_image}" \
    --build-arg "TT_INFERENCE_SERVER_COMMIT_SHA=${ttis_commit}" \
    --build-arg "TT_QUETZAL_COMMIT_SHA=${quetzal_commit}" \
    --tag "${output_tag}" \
    "${repo_root}"
