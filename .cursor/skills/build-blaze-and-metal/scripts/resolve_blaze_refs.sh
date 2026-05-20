#!/usr/bin/env bash
# Resolve tt-llm-engine branch (from Dockerfile.blaze) and pinned tt-metal submodule SHA.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DOCKERFILE="${REPO_ROOT}/tt-media-server/Dockerfile.blaze"
TT_LLM_ENGINE_REPO="${TT_LLM_ENGINE_REPO:-https://github.com/tenstorrent/tt-llm-engine.git}"

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "ERROR: ${DOCKERFILE} not found" >&2
  exit 1
fi

# e.g. git clone --depth 1 --branch main https://.../tt-llm-engine.git
BRANCH="$(grep -oP 'git clone --depth 1 --branch \K\S+' "${DOCKERFILE}" | head -1 || true)"
BRANCH="${BRANCH:-main}"

echo "dockerfile=${DOCKERFILE}"
echo "tt_llm_engine_branch=${BRANCH}"

ENGINE_SHA=""
if command -v git >/dev/null 2>&1; then
  ENGINE_SHA="$(git ls-remote "${TT_LLM_ENGINE_REPO}" "refs/heads/${BRANCH}" 2>/dev/null | awk '{print $1; exit}' || true)"
fi

if [[ -z "${ENGINE_SHA}" ]]; then
  echo "tt_llm_engine_sha=unresolved"
  echo "tt_metal_sha=unresolved"
  echo "note=Could not ls-remote tt-llm-engine (private repo or no network). Use a local clone: cd tt-media-server/cpp_server/tt-llm-engine && git rev-parse HEAD && git -C tt-metal rev-parse HEAD"
  exit 0
fi

echo "tt_llm_engine_sha=${ENGINE_SHA}"

METAL_SHA=""
if command -v curl >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  METAL_SHA="$(curl -fsSL "https://api.github.com/repos/tenstorrent/tt-llm-engine/git/trees/${ENGINE_SHA}" 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for entry in data.get('tree', []):
    if entry.get('path') == 'tt-metal' and entry.get('type') == 'commit':
        print(entry['sha'])
        break
" 2>/dev/null || true)"
fi

if [[ -z "${METAL_SHA}" ]]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "${TMP}"' EXIT
  if git clone --depth 1 --branch "${BRANCH}" "${TT_LLM_ENGINE_REPO}" "${TMP}/tt-llm-engine" 2>/dev/null; then
    git -C "${TMP}/tt-llm-engine" submodule update --init tt-metal 2>/dev/null || true
    METAL_SHA="$(git -C "${TMP}/tt-llm-engine/tt-metal" rev-parse HEAD 2>/dev/null || true)"
  fi
fi

if [[ -n "${METAL_SHA}" ]]; then
  echo "tt_metal_sha=${METAL_SHA}"
else
  echo "tt_metal_sha=unresolved"
  echo "note=Resolve tt-metal manually from tt-llm-engine submodule at branch ${BRANCH}"
fi
