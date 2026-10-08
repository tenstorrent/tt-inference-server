#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# remote_runner.sh -- runs ON the Colab VM. colab_gpu_reference.sh uploads it
# next to gpuref.py and launches it detached (setsid), so a dropped CLI
# connection does not kill the evals. For each model it serves the HF repo
# with an upstream vLLM on :8000 under the HF repo id and runs TTIS's own
# `run.py --workflow evals --tt-device gpu` against it (full evals, no limits),
# exactly as docs/gpu_workflows.md describes for bring-your-own-server GPUs.
#
# State in WORK_DIR (polled by the driver):
#   phase                 one line, the current step
#   DONE | FAILED         terminal markers (FAILED: setup failed or a model failed)
#   runner.log            this script's log (the launcher redirects into it)
#   results/              what the driver downloads:
#     status.json         per-model status
#     <org>__<name>/      serve_plan.json, vllm_server.log, run_py.log, provenance.json
#     workflow_logs/      TTIS workflow_logs (eval reports, lm-eval outputs, run logs)
#
# The HF token is expected at ~/.cache/huggingface/token (mode 0600, uploaded
# by the driver); huggingface_hub, vLLM and lm-eval all read it from there.
# This script never reads, prints or exports it.

set -uo pipefail

# Pinned vLLM. 0.13.0 is the version TTIS already pins for its vLLM client
# venvs (requirements/llm-vllm.txt, benchmarks-vllm.txt), its CUDA 12 wheels
# run on Colab's H100 driver, and it serves all three first targets natively:
# SOLAR-10.7B and Llama-3.2-1B (LlamaForCausalLM, incl. llama3 rope scaling)
# and Qwen1.5-0.5B-Chat (Qwen2ForCausalLM). Its transformers<5 cap is fine for
# these tokenizers. Override with --vllm-version only to work around a VM issue.
VLLM_VERSION="0.13.0"
TTIS_REPO_URL="https://github.com/tenstorrent/tt-inference-server.git"
WORK_DIR="/content/gpuref"
TTIS_SHA=""
SERVER_TIMEOUT=2700   # seconds to wait for /v1/models (download + load + graphs)
PORT=8000
MODELS=()

usage() {
    cat <<EOF
Usage: $0 --ttis-sha SHA [--work-dir DIR] [--vllm-version V] [--server-timeout S] MODEL...
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ttis-sha) TTIS_SHA="$2"; shift 2 ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --vllm-version) VLLM_VERSION="$2"; shift 2 ;;
        --server-timeout) SERVER_TIMEOUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) MODELS+=("$1"); shift ;;
    esac
done
[[ -n "${TTIS_SHA}" && ${#MODELS[@]} -gt 0 ]] || { usage >&2; exit 2; }
[[ "${TTIS_SHA}" =~ ^[0-9a-f]{40}$ ]] || { echo "--ttis-sha must be a full 40-hex sha" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUREF="${SCRIPT_DIR}/gpuref.py"
TTIS_DIR="${WORK_DIR}/tt-inference-server"
VENV="${WORK_DIR}/vllm-venv"
RESULTS="${WORK_DIR}/results"
SERVER_PID=""

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
phase() { printf '%s\n' "$*" > "${WORK_DIR}/phase"; log "PHASE: $*"; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

finish() {
    local marker="$1" reason="$2"
    stop_server
    if [[ -d "${TTIS_DIR}/workflow_logs" ]]; then
        rm -rf "${RESULTS}/workflow_logs"
        cp -a "${TTIS_DIR}/workflow_logs" "${RESULTS}/workflow_logs"
    fi
    cp -f "${WORK_DIR}/runner.log" "${RESULTS}/runner.log" 2>/dev/null || true
    phase "${marker}: ${reason}"
    printf '%s\n' "${reason}" > "${WORK_DIR}/${marker}"
}

on_exit() {
    local rc=$?
    if [[ ! -e "${WORK_DIR}/DONE" && ! -e "${WORK_DIR}/FAILED" ]]; then
        finish FAILED "runner exited unexpectedly (rc=${rc})"
    fi
}
trap on_exit EXIT
trap 'exit 143' TERM INT

stop_server() {
    [[ -n "${SERVER_PID}" ]] || return 0
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "stopping vLLM (pgid ${SERVER_PID})"
        kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
    sleep 5  # let the CUDA context and port go before the next model
}

# start_server LOG ARGS...: vllm serve in its own process group (SERVER_PID).
start_server() {
    local logfile="$1"; shift
    log "${VENV}/bin/vllm $*"
    setsid "${VENV}/bin/vllm" "$@" > "${logfile}" 2>&1 < /dev/null &
    SERVER_PID=$!
}

# Succeeds once /v1/models lists the served HF repo id.
server_ready() {
    local model="$1"
    curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
        | python3 -c 'import json,sys; ids=[m["id"] for m in json.load(sys.stdin).get("data",[])]; sys.exit(0 if sys.argv[1] in ids else 1)' "${model}"
}

wait_for_server() {
    local model="$1" waited=0
    while (( waited < SERVER_TIMEOUT )); do
        if server_ready "${model}"; then
            log "vLLM ready after ${waited}s"
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "vLLM exited during startup"
            return 1
        fi
        sleep 10
        waited=$((waited + 10))
    done
    log "vLLM not ready after ${SERVER_TIMEOUT}s"
    return 1
}

setup() {
    phase "setup: gpu"
    nvidia-smi || return 1

    phase "setup: clone tt-inference-server@${TTIS_SHA}"
    rm -rf "${TTIS_DIR}"
    git init -q "${TTIS_DIR}" \
        && git -C "${TTIS_DIR}" remote add origin "${TTIS_REPO_URL}" \
        && git -C "${TTIS_DIR}" fetch -q --depth 1 origin "${TTIS_SHA}" \
        && git -C "${TTIS_DIR}" checkout -q --detach FETCH_HEAD || return 1
    [[ "$(git -C "${TTIS_DIR}" rev-parse HEAD)" == "${TTIS_SHA}" ]] || return 1

    # docs/workflows_user_guide.md: run.py bootstraps its own venvs with venv
    # and uv; the host only needs python3 with venv/ensurepip (plus PyYAML,
    # which run.py imports before any venv exists).
    phase "setup: host python"
    local probe pyver
    probe="$(mktemp -d)"
    if ! python3 -m venv "${probe}/v" >/dev/null 2>&1; then
        pyver="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
        log "python3 -m venv unavailable; installing python${pyver}-venv"
        apt-get update -qq && apt-get install -y -qq "python${pyver}-venv" python3-venv || return 1
    fi
    rm -rf "${probe}"
    python3 -c 'import yaml' 2>/dev/null || python3 -m pip install -q pyyaml || return 1
    command -v uv >/dev/null 2>&1 || python3 -m pip install -q uv || return 1

    phase "setup: vllm==${VLLM_VERSION}"
    # A dedicated venv keeps vLLM's torch away from Colab's preinstalled one.
    uv venv -q --python 3.12 "${VENV}" \
        && uv pip install -q --python "${VENV}/bin/python" "vllm==${VLLM_VERSION}" || return 1
    "${VENV}/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())' || return 1
}

run_model() {
    local model="$1" idx="$2" total="$3"
    local slug="${model//\//__}"
    local mdir="${RESULTS}/${slug}"
    local started finished evals_rc="" status note="" cap=""
    mkdir -p "${mdir}"
    started="$(now)"

    phase "model ${idx}/${total} ${model}: plan"
    if ! (cd "${TTIS_DIR}" && python3 "${GPUREF}" serve-plan --ttis-dir "${TTIS_DIR}" --model "${model}") \
            > "${mdir}/serve_plan.json" 2> "${mdir}/serve_plan.err"; then
        status="failed"; note="GPU spec did not resolve (serve_plan.err)"
    else
        local -a serve_args=()
        mapfile -t serve_args < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["vllm_serve_args"]))' "${mdir}/serve_plan.json")
        phase "model ${idx}/${total} ${model}: vllm serve"
        start_server "${mdir}/vllm_server.log" "${serve_args[@]}"
        local ready=0
        if wait_for_server "${model}"; then
            ready=1
        else
            # If this GPU cannot hold one max_context sequence, vLLM refuses
            # and prints its own estimate. Retry once at that length and record
            # the cap: lm-eval still sizes prompts to max_context, so a capped
            # run is flagged in status.json and provenance.json for review.
            cap="$(grep -oE 'estimated maximum model length is [0-9]+' "${mdir}/vllm_server.log" | grep -oE '[0-9]+$' | tail -1)"
            stop_server
            if [[ -n "${cap}" ]]; then
                log "vLLM refused max_context on this GPU; retrying with --max-model-len ${cap}"
                local i
                for i in "${!serve_args[@]}"; do
                    if [[ "${serve_args[$i]}" == "--max-model-len" ]]; then serve_args[i+1]="${cap}"; fi
                done
                mv "${mdir}/vllm_server.log" "${mdir}/vllm_server.refused.log"
                start_server "${mdir}/vllm_server.log" "${serve_args[@]}"
                if wait_for_server "${model}"; then ready=1; fi
            fi
        fi
        if [[ "${ready}" -eq 0 ]]; then
            status="failed"; note="vLLM did not become ready (vllm_server.log)"
        else
            phase "model ${idx}/${total} ${model}: evals"
            (cd "${TTIS_DIR}" && python3 run.py --workflow evals --tt-device gpu --model "${model}" --dev-mode) \
                > "${mdir}/run_py.log" 2>&1 < /dev/null
            evals_rc=$?
            if [[ "${evals_rc}" -eq 0 ]]; then status="ok"; else status="failed"; note="run.py exited ${evals_rc} (run_py.log)"; fi
            if [[ -n "${cap}" ]]; then note="${note:+${note}; }max_model_len capped to ${cap} on this GPU (spec max_context not servable)"; fi
        fi
        stop_server
    fi
    finished="$(now)"
    log "${model}: ${status}${note:+ -- ${note}}"

    python3 "${GPUREF}" provenance --out "${mdir}/provenance.json" --model "${model}" \
        --ttis-dir "${TTIS_DIR}" --plan "${mdir}/serve_plan.json" \
        --vllm-python "${VENV}/bin/python" --vllm-bin "${VENV}/bin/vllm" \
        --status "${status}" ${evals_rc:+--evals-rc "${evals_rc}"} \
        --started-at "${started}" --finished-at "${finished}" ${note:+--note "${note}"} \
        ${cap:+--max-model-len-cap "${cap}"} \
        || log "WARNING: provenance.json not written for ${model}"

    python3 - "${RESULTS}/status.json" "${model}" "${status}" "${note}" <<'PY'
import json, sys
path, model, status, note = sys.argv[1:5]
try:
    data = json.load(open(path))
except (OSError, ValueError):
    data = {}
data[model] = {"status": status, "note": note or None}
json.dump(data, open(path, "w"), indent=2)
PY
    [[ "${status}" == "ok" ]]
}

main() {
    mkdir -p "${RESULTS}"
    rm -f "${WORK_DIR}/DONE" "${WORK_DIR}/FAILED"
    log "models: ${MODELS[*]}; ttis ${TTIS_SHA}; vllm ${VLLM_VERSION}"
    if ! setup; then
        finish FAILED "setup failed at: $(cat "${WORK_DIR}/phase")"
        exit 1
    fi
    local failed=0 i=0
    for model in "${MODELS[@]}"; do
        i=$((i + 1))
        run_model "${model}" "${i}" "${#MODELS[@]}" || failed=$((failed + 1))
    done
    if (( failed > 0 )); then
        finish FAILED "${failed}/${#MODELS[@]} model(s) failed; see results/status.json"
        exit 1
    fi
    finish DONE "${#MODELS[@]} model(s) ok"
}

main
