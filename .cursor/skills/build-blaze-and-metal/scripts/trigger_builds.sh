#!/usr/bin/env bash
# Trigger tt-shield workflows to build blaze media-server and tt-metal images.
set -euo pipefail

SHIELD_REPO="${SHIELD_REPO:-tenstorrent/tt-shield}"
INFERENCE_REF="${INFERENCE_REF:-}"
TT_METAL_REF="${TT_METAL_REF:-}"
BLAZE_WORKFLOW="${BLAZE_WORKFLOW:-on-dispatch-build-media-server.yml}"
METAL_WORKFLOW="${METAL_WORKFLOW:-on-dispatch-without-inference-server.yml}"

usage() {
  cat <<EOF
Usage: $0 [options]

Resolves refs from Dockerfile.blaze (unless overridden), then dispatches tt-shield builds.

Options:
  --inference-ref REF   tt-inference-server git ref (default: current branch or HEAD)
  --metal-ref REF       tt-metal git ref (default: submodule SHA from tt-llm-engine)
  --blaze-only          Only trigger blaze/media-server image build
  --metal-only          Only trigger tt-metal image build
  --dry-run             Print gh commands without running
  -h, --help            Show this help

Requires: gh auth login (read access to tt-shield)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DRY_RUN=0
BLAZE_ONLY=0
METAL_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --inference-ref) INFERENCE_REF="$2"; shift 2 ;;
    --metal-ref) TT_METAL_REF="$2"; shift 2 ;;
    --blaze-only) BLAZE_ONLY=1; shift ;;
    --metal-only) METAL_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI not found. Install: https://cli.github.com/" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh not authenticated. Run: gh auth login" >&2
  exit 1
fi

# Resolve refs from Dockerfile.blaze + tt-llm-engine submodule
eval "$("${SCRIPT_DIR}/resolve_blaze_refs.sh" | grep -E '^[a-z_]+=' | sed 's/^/export /')"

if [[ -z "${INFERENCE_REF}" ]]; then
  INFERENCE_REF="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
  if [[ "${INFERENCE_REF}" == "HEAD" ]]; then
    INFERENCE_REF="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  fi
fi

if [[ -z "${TT_METAL_REF}" || "${TT_METAL_REF}" == "unresolved" ]]; then
  if [[ "${tt_metal_sha:-}" != "unresolved" && -n "${tt_metal_sha:-}" ]]; then
    TT_METAL_REF="${tt_metal_sha}"
  else
    TT_METAL_REF="main"
    echo "WARN: tt-metal SHA unresolved; using ref: ${TT_METAL_REF}" >&2
  fi
fi

echo "=== Build refs ==="
echo "  tt-llm-engine branch (Dockerfile.blaze): ${tt_llm_engine_branch:-?}"
echo "  tt-llm-engine tip SHA:                    ${tt_llm_engine_sha:-?}"
echo "  tt-metal (submodule / workflow input):    ${TT_METAL_REF}"
echo "  tt-inference-server ref:                  ${INFERENCE_REF}"
echo ""

run_gh() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

# Blaze image: tt-media-inference-server-blaze (Dockerfile.blaze)
if [[ "${METAL_ONLY}" -eq 0 ]]; then
  echo "=== Blaze media-server image ==="
  echo "Workflow: ${SHIELD_REPO} / ${BLAZE_WORKFLOW}"
  echo "  inference-server-type: blaze-media-inference-server"
  echo "  inference-server-git-ref: ${INFERENCE_REF}"
  echo "  tt-metal-git-ref: ${TT_METAL_REF}"
  echo ""
  if ! gh workflow view "${BLAZE_WORKFLOW}" -R "${SHIELD_REPO}" >/dev/null 2>&1; then
    echo "WARN: ${BLAZE_WORKFLOW} not found. List workflows:" >&2
    gh workflow list -R "${SHIELD_REPO}" | grep -iE 'media|blaze|inference' || true
    echo "Adjust BLAZE_WORKFLOW=... if the file name differs on main." >&2
  fi
  run_gh gh workflow run "${BLAZE_WORKFLOW}" -R "${SHIELD_REPO}" \
    -f "inference-server-type=blaze-media-inference-server" \
    -f "inference-server-git-ref=${INFERENCE_REF}" \
    -f "tt-metal-git-ref=${TT_METAL_REF}"
  echo "Blaze build dispatched. Watch: https://github.com/${SHIELD_REPO}/actions/workflows/${BLAZE_WORKFLOW}"
  echo ""
fi

# Metal image: ghcr.io/tenstorrent/tt-shield/cnn-test:<sha>_<run_id>
if [[ "${BLAZE_ONLY}" -eq 0 ]]; then
  echo "=== tt-metal image (cnn-test) ==="
  echo "Workflow: ${SHIELD_REPO} / ${METAL_WORKFLOW}"
  echo "  tt-metal-git-ref: ${TT_METAL_REF}"
  echo "  (also runs a model test after build; pick a minimal model if prompted)"
  echo ""
  if ! gh workflow view "${METAL_WORKFLOW}" -R "${SHIELD_REPO}" >/dev/null 2>&1; then
    echo "WARN: ${METAL_WORKFLOW} not found. Try:" >&2
    echo "  gh workflow list -R ${SHIELD_REPO} | grep -i metal" >&2
    echo "  or set METAL_WORKFLOW=on-dispatch-build-metal.yml if that exists on main." >&2
  fi
  # Minimal inputs; workflow has many model/runner fields with defaults.
  run_gh gh workflow run "${METAL_WORKFLOW}" -R "${SHIELD_REPO}" \
    -f "tt-metal-git-ref=${TT_METAL_REF}" \
    -f "model=stable-diffusion-xl-base-1.0" \
    -f "runner-label=tt-ubuntu-2204-xlarge-stable" \
    -f "device-type=n150"
  echo "Metal build dispatched. Image tag pattern: ghcr.io/tenstorrent/tt-shield/cnn-test:<sha>_<run_id>"
  echo "Watch: https://github.com/${SHIELD_REPO}/actions/workflows/${METAL_WORKFLOW}"
fi
