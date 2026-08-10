#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# End-to-end verification of the `--tools aiperf` benchmark path for a model
# "served" on a 32-chip Blackhole Galaxy (readiness item 7.1, llm-gauntlet#72).
#
# Real Galaxy hardware is not required: this starts a mock OpenAI-compatible
# server (llm-d-inference-sim) seeded with Blackhole-Galaxy-representative
# latency, builds the AIPerf venv from requirements/llm-aiperf.txt, and runs the
# real repo AIPerf path (AIPerfDriver -> AIPerfParser -> apply_target_checks ->
# report_module) against it via scripts/verify/aiperf_bh_galaxy_e2e.py.
#
# Usage:
#   scripts/verify/run_aiperf_bh_galaxy_e2e.sh
#
# Override any knob via environment variables, e.g.:
#   PORT=8100 TTFT=90ms ITL=30ms scripts/verify/run_aiperf_bh_galaxy_e2e.sh
#
# Seed the mock latency (TTFT/ITL) from the Appendix B AIPerf target sheet so a
# tier passes or fails realistically; pass real targets to the harness via
# --targets-json (see TARGETS_JSON below).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# ── Config (override via env) ────────────────────────────────────────────────
CONTAINER_ENGINE="${CONTAINER_ENGINE:-docker}"
IMAGE="${IMAGE:-ghcr.io/llm-d/llm-d-inference-sim:v0.9.0}"
CONTAINER_NAME="${CONTAINER_NAME:-bh-galaxy-sim}"
PORT="${PORT:-8000}"
# The mock stands in for the Blackhole Galaxy-served model.
SERVED_MODEL="${SERVED_MODEL:-tenstorrent/blackhole-galaxy-mock}"
# A small real HF tokenizer AIPerf uses for synthetic prompts / token counting.
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-0.5B-Instruct}"
# Blackhole-Galaxy-representative latency seeds (replace from the target sheet).
TTFT="${TTFT:-60ms}"
TTFT_STDDEV="${TTFT_STDDEV:-8ms}"
ITL="${ITL:-22ms}"
ITL_STDDEV="${ITL_STDDEV:-3ms}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv_aiperf_e2e}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/workflow_logs/aiperf_bh_galaxy_e2e}"
TARGETS_JSON="${TARGETS_JSON:-}"
PYTHON311="${PYTHON311:-python3.11}"
# Optional sweep overrides (comma-separated ISLs / concurrencies, OSL, requests).
ISLS="${ISLS:-}"
CONCURRENCIES="${CONCURRENCIES:-}"
OSL="${OSL:-}"
NUM_PROMPTS="${NUM_PROMPTS:-}"

log() { echo "[aiperf-e2e] $*"; }

cleanup() {
  log "Stopping mock server ${CONTAINER_NAME}"
  "${CONTAINER_ENGINE}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ── 1. AIPerf venv ───────────────────────────────────────────────────────────
if [[ ! -x "${VENV_DIR}/bin/aiperf" && ! -f "${VENV_DIR}/bin/python" ]]; then
  log "Creating AIPerf venv at ${VENV_DIR}"
  "${PYTHON311}" -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
fi
if ! "${VENV_DIR}/bin/python" -c "import aiperf" >/dev/null 2>&1; then
  log "Installing aiperf deps (requirements/llm-aiperf.txt)"
  # `-c constraints.txt` in the requirements file resolves relative to CWD.
  ( cd "${REPO_ROOT}/requirements" && "${VENV_DIR}/bin/pip" install -r llm-aiperf.txt )
fi
log "AIPerf venv ready: $("${VENV_DIR}/bin/python" -m aiperf --version 2>/dev/null || echo aiperf)"

# ── 2. Mock server ───────────────────────────────────────────────────────────
"${CONTAINER_ENGINE}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
log "Starting mock server ${IMAGE} on port ${PORT} (TTFT=${TTFT}, ITL=${ITL})"
"${CONTAINER_ENGINE}" run -d --name "${CONTAINER_NAME}" -p "${PORT}:${PORT}" "${IMAGE}" \
  --model "${SERVED_MODEL}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --time-to-first-token "${TTFT}" \
  --time-to-first-token-std-dev "${TTFT_STDDEV}" \
  --inter-token-latency "${ITL}" \
  --inter-token-latency-std-dev "${ITL_STDDEV}" \
  --mode random >/dev/null

log "Waiting for /health ..."
for _ in $(seq 1 30); do
  if curl -sS -m 3 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    log "Mock server healthy."
    break
  fi
  sleep 1
done
curl -sS -m 5 "http://localhost:${PORT}/health" >/dev/null || {
  log "ERROR: mock server did not become healthy"; exit 1;
}

# ── 3. Run the real AIPerf benchmark path against the mock ────────────────────
HARNESS_ARGS=(
  --base-url "http://localhost"
  --service-port "${PORT}"
  --served-model "${SERVED_MODEL}"
  --tokenizer "${TOKENIZER}"
  --device "BLACKHOLE_GALAXY"
  --venv-python "${VENV_DIR}/bin/python"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${TARGETS_JSON}" ]]; then
  HARNESS_ARGS+=(--targets-json "${TARGETS_JSON}")
fi
[[ -n "${ISLS}" ]] && HARNESS_ARGS+=(--isls "${ISLS}")
[[ -n "${CONCURRENCIES}" ]] && HARNESS_ARGS+=(--concurrencies "${CONCURRENCIES}")
[[ -n "${OSL}" ]] && HARNESS_ARGS+=(--osl "${OSL}")
[[ -n "${NUM_PROMPTS}" ]] && HARNESS_ARGS+=(--num-prompts "${NUM_PROMPTS}")

log "Running AIPerf E2E harness ..."
python3 scripts/verify/aiperf_bh_galaxy_e2e.py "${HARNESS_ARGS[@]}"
rc=$?

log "Done. Artifacts + report under ${OUTPUT_DIR}"
exit "${rc}"
