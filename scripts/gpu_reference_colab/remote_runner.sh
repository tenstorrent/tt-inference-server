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
#
# --serve (agentic tasks, one model): no evals here. Serve the model with
# vLLM's API key (VLLM_API_KEY, from ${WORK_DIR}/secrets/endpoint_api_key),
# publish it through an ngrok tunnel (NGROK_AUTHTOKEN from secrets/), write the
# public URL to ${WORK_DIR}/endpoint and the SERVING marker, and keep both up
# until the driver writes STOP (Shield's endpoint tests run the evals). Neither
# secret is ever on a command line or in a log.

set -uo pipefail

# Serving stack per model (vLLM, transformers pin, server env): gpuref.py
# serving_stack, from the model's architecture (default vLLM 0.13.0, TTIS's
# own client pin). Each model's serve plan carries its stack; the runner keeps
# one venv per stack under ${WORK_DIR}/venvs, built on first use, so a sweep
# with mixed stacks runs in one session. --vllm-version / --transformers-version
# override every model's stack (explicit runs only).
VLLM_OVERRIDE=""
TRANSFORMERS_OVERRIDE=""
TTIS_REPO_URL="https://github.com/tenstorrent/tt-inference-server.git"
WORK_DIR="/content/gpuref"
TTIS_SHA=""
SERVER_TIMEOUT=2700   # seconds to wait for /v1/models (download + load + graphs)
PORT=8000
MODELS=()
SERVE=0 TUNNEL_DOMAIN=""   # --serve [--tunnel-domain D]: see above
NGROK_URL="https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
declare -A TASKS=()   # model -> comma-separated tasks (--task); unset = all its tasks

usage() {
    cat <<EOF
Usage: $0 --ttis-sha SHA [--work-dir DIR] [--vllm-version V] [--transformers-version V] [--server-timeout S] [--task MODEL=T[,T]]... [--serve [--tunnel-domain D]] MODEL...
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ttis-sha) TTIS_SHA="$2"; shift 2 ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --vllm-version) VLLM_OVERRIDE="$2"; shift 2 ;;
        --transformers-version) TRANSFORMERS_OVERRIDE="$2"; shift 2 ;;
        --server-timeout) SERVER_TIMEOUT="$2"; shift 2 ;;
        --task) TASKS["${2%%=*}"]="${2#*=}"; shift 2 ;;
        --serve) SERVE=1; shift ;;
        --tunnel-domain) TUNNEL_DOMAIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) MODELS+=("$1"); shift ;;
    esac
done
[[ -n "${TTIS_SHA}" && ${#MODELS[@]} -gt 0 ]] || { usage >&2; exit 2; }
[[ "${TTIS_SHA}" =~ ^[0-9a-f]{40}$ ]] || { echo "--ttis-sha must be a full 40-hex sha" >&2; exit 2; }
[[ "${SERVE}" -eq 0 || ${#MODELS[@]} -eq 1 ]] || { echo "--serve serves exactly one model" >&2; exit 2; }
[[ -z "${TUNNEL_DOMAIN}" || "${TUNNEL_DOMAIN}" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "bad --tunnel-domain" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUREF="${SCRIPT_DIR}/gpuref.py"
TTIS_DIR="${WORK_DIR}/tt-inference-server"
VENV=""            # this model's stack venv (ensure_stack)
SERVER_ENV=()      # this model's stack env, KEY=VALUE
RESULTS="${WORK_DIR}/results"
SERVER_PID=""
SECRETS="${WORK_DIR}/secrets"   # --serve: endpoint_api_key, auth_header, ngrok_authtoken (0600)
TUNNEL_PID=""

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
phase() { printf '%s\n' "$*" > "${WORK_DIR}/phase"; log "PHASE: $*"; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

finish() {
    local marker="$1" reason="$2"
    stop_tunnel
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
    # The stack's server env, and its venv's bin (tools vLLM may shell out to) on PATH.
    # --serve: vLLM reads the API key from VLLM_API_KEY, so it is in no argv.
    if [[ "${SERVE}" -eq 1 ]]; then
        VLLM_API_KEY="$(< "${SECRETS}/endpoint_api_key")" PATH="${VENV}/bin:${PATH}" \
            setsid env ${SERVER_ENV[@]+"${SERVER_ENV[@]}"} \
            "${VENV}/bin/vllm" "$@" > "${logfile}" 2>&1 < /dev/null &
    else
        PATH="${VENV}/bin:${PATH}" setsid env ${SERVER_ENV[@]+"${SERVER_ENV[@]}"} \
            "${VENV}/bin/vllm" "$@" > "${logfile}" 2>&1 < /dev/null &
    fi
    SERVER_PID=$!
}

# Authorization header for --serve requests, read by curl from a 0600 file.
auth() { [[ "${SERVE}" -eq 1 ]] && printf '%s\n' -H "@${SECRETS}/auth_header"; }

stop_tunnel() {
    [[ -n "${TUNNEL_PID}" ]] || return 0
    kill -TERM -- "-${TUNNEL_PID}" 2>/dev/null || true
    wait "${TUNNEL_PID}" 2>/dev/null || true
    TUNNEL_PID=""
}

# start_tunnel: ngrok in its own process group -> ${WORK_DIR}/endpoint (0600).
start_tunnel() {
    local url="" i
    if [[ ! -x "${WORK_DIR}/bin/ngrok" ]]; then
        mkdir -p "${WORK_DIR}/bin"
        curl -sfL "${NGROK_URL}" | tar -xz -C "${WORK_DIR}/bin" ngrok || return 1
    fi
    NGROK_AUTHTOKEN="$(< "${SECRETS}/ngrok_authtoken")" setsid "${WORK_DIR}/bin/ngrok" http "${PORT}" \
        --log stdout --log-format json ${TUNNEL_DOMAIN:+--url "https://${TUNNEL_DOMAIN}"} \
        >> "${RESULTS}/tunnel.log" 2>&1 < /dev/null &
    TUNNEL_PID=$!
    for i in $(seq 1 30); do
        url="$(curl -sf http://127.0.0.1:4040/api/tunnels 2>/dev/null \
            | python3 -c 'import json,sys; print(next((t["public_url"] for t in json.load(sys.stdin)["tunnels"] if t["public_url"].startswith("https://")), ""))' 2>/dev/null)"
        [[ -n "${url}" ]] && break
        kill -0 "${TUNNEL_PID}" 2>/dev/null || return 1
        sleep 2
    done
    [[ -n "${url}" ]] || return 1
    ( umask 077; printf '%s\n' "${url}" > "${WORK_DIR}/endpoint" )
    "${WORK_DIR}/bin/ngrok" version >> "${RESULTS}/tunnel.log" 2>&1 || true
}

# The tunnel answers like the local server: /v1/models through it, with the key.
tunnel_ready() {
    local model="$1" url body; local -a hdr=()
    url="$(cat "${WORK_DIR}/endpoint" 2>/dev/null)" || return 1
    mapfile -t hdr < <(auth)
    body="$(curl -sf --max-time 30 "${hdr[@]}" "${url%/}/v1/models" 2>/dev/null)" || return 1
    printf '%s' "${body}" | python3 "${GPUREF}" served "${model}"
}

# serve_until_stop MODEL: keep vLLM and the tunnel up until the driver writes
# STOP. A dead tunnel is restarted (a reserved --tunnel-domain keeps its URL);
# a dead vLLM ends the run (the driver rolls over to a new session).
serve_until_stop() {
    local model="$1" restarts=0
    local up=0 i
    if start_tunnel; then
        for i in $(seq 1 10); do tunnel_ready "${model}" && { up=1; break; }; sleep 6; done
    fi
    [[ "${up}" -eq 1 ]] || { log "tunnel did not come up after ${i:-0} checks (tunnel.log)"; return 1; }
    # This VM's serving snapshot (GPU, stack, serve command, cap), before it
    # serves: provenance.json is only written after STOP, and a VM lost before
    # then takes it along. The driver binds each Shield score to the snapshot
    # of the VM that served it. (mdir, started, cap: the caller's locals.)
    local -a agentic=()
    [[ -n "${TASKS[${model}]:-}" ]] && agentic=(--agentic-tasks "${TASKS[${model}]}")
    python3 "${GPUREF}" provenance --out "${mdir}/serving_provenance.json" --model "${model}" \
        --ttis-dir "${TTIS_DIR}" --plan "${mdir}/serve_plan.json" \
        --vllm-python "${VENV}/bin/python" --vllm-bin "${VENV}/bin/vllm" \
        --status serving --started-at "${started}" --finished-at "$(now)" \
        ${cap:+--max-model-len-cap "${cap}"} ${agentic[@]+"${agentic[@]}"} \
        --endpoint "$(python3 "${GPUREF}" redact-url --url-file "${WORK_DIR}/endpoint")" \
        || { log "serving_provenance.json not written; not serving without it"; return 1; }
    touch "${WORK_DIR}/SERVING"
    log "serving ${model} through the tunnel"
    while [[ ! -e "${WORK_DIR}/STOP" ]]; do
        sleep 30
        kill -0 "${SERVER_PID}" 2>/dev/null || { log "vLLM exited while serving"; rm -f "${WORK_DIR}/SERVING"; return 1; }
        if ! kill -0 "${TUNNEL_PID}" 2>/dev/null; then
            restarts=$((restarts + 1)); log "tunnel exited; restart ${restarts}"
            stop_tunnel; start_tunnel || { rm -f "${WORK_DIR}/SERVING"; return 1; }
            printf '%s\n' "${restarts}" > "${WORK_DIR}/tunnel_restarts"
        fi
    done
    rm -f "${WORK_DIR}/SERVING"
    log "STOP requested; ${restarts} tunnel restart(s)"
}

# Succeeds once /v1/models lists the served HF repo id.
server_ready() {
    local model="$1" body; local -a hdr=()
    mapfile -t hdr < <(auth)
    body="$(curl -sf ${hdr[@]+"${hdr[@]}"} "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)" || return 1
    printf '%s' "${body}" | python3 "${GPUREF}" served "${model}"
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

}

# ensure_stack VLLM [TRANSFORMERS]: sets VENV to that stack's venv, building it
# on first use. A dedicated venv keeps vLLM's torch away from Colab's own.
ensure_stack() {
    local vllm="$1" tf="${2:-}"
    VENV="${WORK_DIR}/venvs/vllm-${vllm}${tf:+-transformers-${tf}}"
    [[ -x "${VENV}/bin/vllm" && -e "${VENV}/.ready" ]] && return 0
    phase "setup: vllm==${vllm}${tf:+ transformers==${tf}}"
    rm -rf "${VENV}"
    uv venv -q --python 3.12 "${VENV}" \
        && uv pip install -q --python "${VENV}/bin/python" "vllm==${vllm}" || return 1
    if [[ -n "${tf}" ]]; then
        uv pip install -q --python "${VENV}/bin/python" "transformers==${tf}" || return 1
    fi
    "${VENV}/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())' || return 1
    touch "${VENV}/.ready"
}

run_model() {
    local model="$1" idx="$2" total="$3"
    local slug="${model//\//__}"
    local mdir="${RESULTS}/${slug}"
    local started finished evals_rc="" status="" note="" cap="" eval_samples=""
    mkdir -p "${mdir}"
    started="$(now)"

    phase "model ${idx}/${total} ${model}: plan"
    local -a overrides=()
    [[ -n "${VLLM_OVERRIDE}" ]] && overrides+=(--vllm-version "${VLLM_OVERRIDE}")
    [[ -n "${TRANSFORMERS_OVERRIDE}" ]] && overrides+=(--transformers-version "${TRANSFORMERS_OVERRIDE}")
    local stack_vllm="" stack_tf=""
    # No GPU spec at the pinned commit: derive one from its TT row into this
    # clone (recorded as derived_gpu_spec.json and in provenance).
    if ! (cd "${TTIS_DIR}" && python3 "${GPUREF}" ensure-gpu-spec --ttis-dir "${TTIS_DIR}" --model "${model}" --out "${mdir}/derived_gpu_spec.json") \
            2> "${mdir}/serve_plan.err"; then
        status="failed"; note="no GPU spec and none derivable from a TT row (serve_plan.err)"
    elif ! (cd "${TTIS_DIR}" && python3 "${GPUREF}" serve-plan --ttis-dir "${TTIS_DIR}" --model "${model}" ${overrides[@]+"${overrides[@]}"}) \
            > "${mdir}/serve_plan.json" 2>> "${mdir}/serve_plan.err"; then
        status="failed"; note="GPU spec did not resolve (serve_plan.err)"
    elif ! python3 "${GPUREF}" materialize-hf-config "${mdir}/serve_plan.json" 2>> "${mdir}/serve_plan.err"; then
        status="failed"; note="could not write the edited config.json (serve_plan.err)"
    elif ! read -r stack_vllm stack_tf < <(python3 "${GPUREF}" plan-stack "${mdir}/serve_plan.json") \
            || ! ensure_stack "${stack_vllm}" "${stack_tf}"; then
        status="failed"; note="serving stack vllm==${stack_vllm} ${stack_tf:+transformers==${stack_tf} }did not install (runner.log)"
    else
        mapfile -t SERVER_ENV < <(python3 "${GPUREF}" plan-stack --env "${mdir}/serve_plan.json")
        local -a serve_args=()
        mapfile -t serve_args < <(python3 "${GPUREF}" plan-argv "${mdir}/serve_plan.json")
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
        elif [[ "${SERVE}" -eq 1 ]]; then
            phase "model ${idx}/${total} ${model}: serving"
            if serve_until_stop "${model}"; then status="ok"; note="served for Shield's endpoint tests"
            else status="failed"; note="serving ended before STOP (runner.log, tunnel.log, vllm_server.log)"; fi
            if [[ -n "${cap}" ]]; then note="${note}; max_model_len capped to ${cap} on this GPU"; fi
        else
            phase "model ${idx}/${total} ${model}: evals"
            # The eval client sends as many concurrent requests as vLLM was
            # given sequences (gpuref.py gpu_concurrency; recorded in provenance).
            local client_conc
            client_conc="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("client_concurrency") or "")' "${mdir}/serve_plan.json")"
            # --task: only these tasks, each in full (gpuref.py task_selection).
            local -a select=()
            if [[ -n "${TASKS[${model}]:-}" ]]; then
                if eval_samples="$(python3 "${GPUREF}" task-selection "${model}" "${TASKS[${model}]}" \
                        --ttis-dir "${TTIS_DIR}" 2>> "${mdir}/serve_plan.err")"; then
                    select=(--eval-samples "${eval_samples}")
                else
                    eval_samples=""
                fi
            fi
            if [[ -n "${TASKS[${model}]:-}" && ${#select[@]} -eq 0 ]]; then
                evals_rc=2; status="failed"; note="--task ${TASKS[${model}]} is not this model's lm-eval tasks (serve_plan.err)"
            else
                (cd "${TTIS_DIR}" && TT_EVAL_CLIENT_CONCURRENCY="${client_conc}" \
                    python3 run.py --workflow evals --tt-device gpu --model "${model}" --dev-mode \
                    ${select[@]+"${select[@]}"}) > "${mdir}/run_py.log" 2>&1 < /dev/null
                evals_rc=$?
            fi
            if [[ "${status}" == "failed" ]]; then :
            elif [[ "${evals_rc}" -ne 0 ]]; then status="failed"; note="run.py exited ${evals_rc} (run_py.log)"
            # run.py exits 0 even when one task's lm-eval subprocess failed.
            elif grep -q "command failed with return code" "${mdir}/run_py.log"; then status="failed"; note="an eval task failed (run_py.log)"
            else status="ok"; fi
            if [[ -n "${cap}" ]]; then note="${note:+${note}; }max_model_len capped to ${cap} on this GPU (spec max_context not servable)"; fi
        fi
        stop_server
    fi
    # Evidence that a cap did not bind: longest prompt+generation per task and
    # any rejected/truncated-request log lines. A capped run that came close is
    # not usable as a reference.
    local evidence="${mdir}/request_evidence.json"
    if [[ -n "${evals_rc}" && "${SERVE}" -eq 0 ]]; then
        local -a logs=(--log "${mdir}/run_py.log" --log "${mdir}/vllm_server.log")
        local f; for f in "${TTIS_DIR}"/workflow_logs/run_logs/*"${model##*/}"*; do [[ -e "$f" ]] && logs+=(--log "$f"); done
        "${VENV}/bin/python" "${GPUREF}" request-evidence --model "${model}" \
            --revision "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("revision") or "main")' "${mdir}/serve_plan.json")" \
            --samples-root "${TTIS_DIR}/workflow_logs/reports_output/evals/${slug}_gpu_evals" \
            "${logs[@]}" --out "${evidence}" || log "WARNING: request evidence not computed for ${model}"
        if [[ -n "${cap}" ]] && python3 "${GPUREF}" cap-bound "${evidence}" "${cap}"; then
            status="failed"; note="${note:+${note}; }a request reached the max_model_len cap or was rejected: not a usable reference"
        fi
    fi
    finished="$(now)"
    log "${model}: ${status}${note:+ -- ${note}}"
    # --serve: the agentic tasks Shield runs against this server, and its URL
    # (redacted: the full URL stays in the 0600 endpoint file).
    local -a served=()
    if [[ "${SERVE}" -eq 1 ]]; then
        [[ -n "${TASKS[${model}]:-}" ]] && served+=(--agentic-tasks "${TASKS[${model}]}")
        [[ -s "${WORK_DIR}/endpoint" ]] && served+=(--endpoint "$(python3 "${GPUREF}" redact-url --url-file "${WORK_DIR}/endpoint")")
    fi

    python3 "${GPUREF}" provenance --out "${mdir}/provenance.json" --model "${model}" \
        --ttis-dir "${TTIS_DIR}" --plan "${mdir}/serve_plan.json" \
        --vllm-python "${VENV}/bin/python" --vllm-bin "${VENV}/bin/vllm" \
        --status "${status}" ${evals_rc:+--evals-rc "${evals_rc}"} \
        --started-at "${started}" --finished-at "${finished}" ${note:+--note "${note}"} \
        ${cap:+--max-model-len-cap "${cap}"} --evidence "${evidence}" \
        ${eval_samples:+--eval-samples "${eval_samples}"} \
        ${served[@]+"${served[@]}"} \
        || log "WARNING: provenance.json not written for ${model}"

    python3 "${GPUREF}" record-status "${RESULTS}/status.json" "${model}" "${status}" "${note}"
    [[ "${status}" == "ok" ]]
}

main() {
    mkdir -p "${RESULTS}"
    rm -f "${WORK_DIR}/DONE" "${WORK_DIR}/FAILED"
    log "models: ${MODELS[*]}; ttis ${TTIS_SHA}; stack per model${VLLM_OVERRIDE:+ (override vllm ${VLLM_OVERRIDE})}${TRANSFORMERS_OVERRIDE:+ (override transformers ${TRANSFORMERS_OVERRIDE})}"
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
