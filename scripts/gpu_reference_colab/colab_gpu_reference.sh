#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# colab_gpu_reference.sh -- GPU reference eval scores on a Google Colab VM (README.md).
#
#   1 preflight   2 start VM   3 upload token   4 launch runner
#   5 poll        6 download + summary          7 stop (EXIT trap, unless --keep)
#
# Only the `colab` calls and this flow live here: checks and parsers are in
# gpuref.py, VM code in snippets/. The HF token goes up as a 0600 temp file through
# `colab upload`: never on a command line, in a log, or in code sent with exec.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REMOTE_REL="content/gpuref"   # /content/gpuref on the VM, for upload/download
GPU="H100" HIGH_MEM=0 TTIS_REF="HEAD" OUT_DIR="" KEEP=0 SESSION="" POLL_MINUTES=5
VLLM_VERSION="" MIN_BALANCE=0 DRY_RUN=0 MODELS=()
NEW_ATTEMPTS=3 NEW_RETRY_SECONDS=120   # per GPU type, on 503 (no capacity)

usage() {
    cat <<'EOF'
Usage: colab_gpu_reference.sh [options] MODEL [MODEL...]
  --gpu G[,G...]      GPU types in preference order (default H100), e.g. H100,A100
  --high-mem          high-RAM machine shape
  --ttis-ref REF      pushed TTIS commit the VM checks out (default HEAD)
  --out DIR           results dir (default workflow_logs/gpu_reference_colab/<session>)
  --keep              do not stop the VM on exit
  --session NAME      session name (default gpuref-<sha[:8]>); same name re-attaches
  --poll-minutes N    minutes between polls (default 5)
  --min-balance CU    stop (keeping partial results) below this compute-unit balance
  --vllm-version V    override the runner's vLLM pin
  --dry-run           print every colab command instead of running it
EOF
}
log() { printf '[gpuref] %s\n' "$*" >&2; }
die() { printf '[gpuref] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;            --high-mem) HIGH_MEM=1; shift ;;
        --ttis-ref) TTIS_REF="$2"; shift 2 ;;  --out) OUT_DIR="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;               --session) SESSION="$2"; shift 2 ;;
        --poll-minutes) POLL_MINUTES="$2"; shift 2 ;;
        --min-balance) MIN_BALANCE="$2"; shift 2 ;;
        --vllm-version) VLLM_VERSION="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;         -h|--help) usage; exit 0 ;;
        -*) usage >&2; die "unknown option: $1" ;;
        *) MODELS+=("$1"); shift ;;
    esac
done
[[ ${#MODELS[@]} -gt 0 ]] || { usage >&2; die "at least one MODEL (HF repo id) is required"; }
for m in "${MODELS[@]}"; do [[ "$m" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || die "not an HF repo id: $m"; done
[[ "${GPU}" =~ ^((T4|L4|G4|H100|A100),)*(T4|L4|G4|H100|A100)$ ]] || die "bad --gpu: ${GPU}"
[[ "${POLL_MINUTES}" =~ ^[1-9][0-9]*$ ]] || die "--poll-minutes must be a positive integer"
[[ "${MIN_BALANCE}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--min-balance must be a number"
[[ -z "${VLLM_VERSION}" || "${VLLM_VERSION}" =~ ^[0-9A-Za-z.+-]+$ ]] || die "bad --vllm-version"

LOCAL_TMP="$(mktemp -d "${TMPDIR:-/tmp}/gpuref.XXXXXX")"; chmod 700 "${LOCAL_TMP}"
SESSION_ACTIVE=0
cleanup() {   # step 7: always runs
    local rc=$?
    rm -rf "${LOCAL_TMP}"
    if [[ "${SESSION_ACTIVE}" -eq 1 && "${KEEP}" -eq 1 ]]; then
        log "--keep: '${SESSION}' left running; re-attach with the same command, stop with: colab stop -s ${SESSION}"
    elif [[ "${SESSION_ACTIVE}" -eq 1 ]]; then
        log "stopping session '${SESSION}'"
        colab_cmd stop -s "${SESSION}" || log "WARNING: 'colab stop -s ${SESSION}' failed; check 'colab sessions'"
    fi
    exit "${rc}"
}
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM

colab_cmd() {   # every colab call; --dry-run prints it to stderr instead
    if [[ "${DRY_RUN}" -eq 1 ]]; then { printf '+ colab'; printf ' %q' "$@"; printf '\n'; } >&2; return 0; fi
    colab "$@" < /dev/null
}
if python3 -c 'import yaml' 2>/dev/null; then PYRUN=(python3)
elif command -v uv >/dev/null 2>&1; then PYRUN=(uv run --no-project --quiet --with pyyaml python)
else die "need python3 with PyYAML, or uv"; fi
gpuref() { "${PYRUN[@]}" "${SCRIPT_DIR}/gpuref.py" "$@"; }
remote_exec() {   # NAME TIMEOUT [RUNNER ARGS...]: run snippets/NAME.py on the VM
    local name="$1" timeout="$2" arg; shift 2
    if [[ $# -gt 0 ]]; then   # launch: prepend the runner's argv (validated values only)
        printf 'ARGS = [' > "${LOCAL_TMP}/${name}.py"
        for arg in "$@"; do printf "'%s', " "${arg}" >> "${LOCAL_TMP}/${name}.py"; done
        printf ']\n' >> "${LOCAL_TMP}/${name}.py"
    else : > "${LOCAL_TMP}/${name}.py"; fi
    cat "${SCRIPT_DIR}/snippets/${name}.py" >> "${LOCAL_TMP}/${name}.py"
    [[ "${DRY_RUN}" -eq 1 ]] && { colab_cmd exec -s "${SESSION}" -f "${LOCAL_TMP}/${name}.py" --timeout "${timeout}"; return; }
    # Hard deadline: a lost kernel websocket can hang `colab exec` past --timeout.
    perl -e 'alarm shift; exec @ARGV' $((timeout + 120)) \
        colab exec -s "${SESSION}" -f "${LOCAL_TMP}/${name}.py" --timeout "${timeout}" < /dev/null
}
remote_step() {   # NAME TIMEOUT MARKER [ARGS...]: `colab exec` exits 0 even if the code raised
    local name="$1" timeout="$2" marker="$3" out; shift 3
    if [[ "${DRY_RUN}" -eq 1 ]]; then remote_exec "${name}" "${timeout}" "$@"; return 0; fi
    out="$(remote_exec "${name}" "${timeout}" "$@" 2>&1)" || true
    printf '%s\n' "${out}" | grep "^GPUREF_" >&2 || true
    printf '%s\n' "${out}" | grep -q "^${marker}" || die "VM step '${name}' failed:"$'\n'"${out}"
}
has_session() { awk -v want="[${SESSION}]" '$1 == want { f = 1 } END { exit !f }'; }
remote_state() {   # sets STATE; fails when the status cannot be read
    local out; out="$(remote_exec status 120 2>&1)" || return 1
    STATE="$(printf '%s\n' "${out}" | gpuref status-view)"
}

# 1 preflight ---------------------------------------------------------------
TTIS_SHA="$(git -C "${REPO_ROOT}" rev-parse --verify "${TTIS_REF}^{commit}" 2>/dev/null)" || die "--ttis-ref ${TTIS_REF} is not a commit"
git -C "${REPO_ROOT}" fetch -q origin 2>/dev/null || log "WARNING: git fetch origin failed"
if [[ -z "$(git -C "${REPO_ROOT}" branch -r --contains "${TTIS_SHA}" 2>/dev/null)" ]]; then
    msg="TTIS ${TTIS_SHA} is not on any origin branch; push it first (the VM clones that exact sha)"
    if [[ "${DRY_RUN}" -eq 1 ]]; then log "WARNING: ${msg}"; else die "${msg}"; fi
fi
git -C "${REPO_ROOT}" diff --quiet "${TTIS_SHA}" -- workflows/model_specs reference_config/evals \
    || log "WARNING: local specs/eval configs differ from ${TTIS_SHA}; the preflight may not match the VM"
SESSION="${SESSION:-gpuref-${TTIS_SHA:0:8}}"
[[ "${SESSION}" =~ ^[A-Za-z0-9._-]+$ ]] || die "bad --session: ${SESSION}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/workflow_logs/gpu_reference_colab/${SESSION}}"
log "models: ${MODELS[*]}; ttis ${TTIS_SHA}; session ${SESSION}; out ${OUT_DIR}"
usable="$(gpuref preflight --gpus "${GPU}" "${MODELS[@]}")" || die "preflight failed (see above)"
read -r -a GPUS <<< "${usable}"
[[ ${#GPUS[@]} -gt 0 ]] || die "no usable GPU type"
HAVE_TOKEN=0; [[ -n "${HF_TOKEN:-}" || -s "${HOME}/.cache/huggingface/token" ]] && HAVE_TOKEN=1
if [[ "${DRY_RUN}" -eq 0 ]]; then
    command -v colab >/dev/null 2>&1 || die "install the Colab CLI: uv tool install google-colab-cli"
    # Without a cached token colab would start an interactive sign-in.
    [[ -s "${HOME}/.config/colab-cli/token.json" ]] || die "Colab CLI not signed in: run 'colab sessions' once in a terminal"
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then colab_cmd sessions; SESSIONS_OUT=""
else SESSIONS_OUT="$(colab_cmd sessions 2>&1)" || die "'colab sessions' failed: ${SESSIONS_OUT}"; fi

# 2 start VM (or re-attach) ---------------------------------------------------
STATE="ABSENT"
if [[ "${DRY_RUN}" -eq 0 ]] && printf '%s\n' "${SESSIONS_OUT}" | has_session; then
    SESSION_ACTIVE=1
    remote_state || die "could not read the runner state on '${SESSION}'"
    log "session '${SESSION}' exists; runner ${STATE}"
else
    for gpu in "${GPUS[@]}"; do
        new_args=(new -s "${SESSION}" --gpu "${gpu}"); [[ "${HIGH_MEM}" -eq 1 ]] && new_args+=(--high-mem)
        for ((attempt = 1, delay = NEW_RETRY_SECONDS; attempt <= NEW_ATTEMPTS; attempt++, delay *= 2)); do
            log "creating '${SESSION}' on ${gpu} (attempt ${attempt}/${NEW_ATTEMPTS})"
            if colab_cmd "${new_args[@]}" > "${LOCAL_TMP}/new.out" 2>&1; then cat "${LOCAL_TMP}/new.out" >&2; SESSION_ACTIVE=1; break 2; fi
            tail -3 "${LOCAL_TMP}/new.out" >&2
            if colab sessions < /dev/null 2>&1 | has_session; then SESSION_ACTIVE=1; die "'colab new' failed but the session exists"; fi
            if ! grep -q "Service Unavailable" "${LOCAL_TMP}/new.out" || (( attempt == NEW_ATTEMPTS )); then break; fi
            log "no ${gpu} capacity (503); retrying in ${delay}s"; sleep "${delay}"
        done
    done
    [[ "${SESSION_ACTIVE}" -eq 1 ]] || die "'colab new' failed on every GPU in --gpu (${GPUS[*]})"
    log "allocated ${gpu} (the exact GPU is recorded in provenance.json)"
fi

if [[ "${STATE}" == "ABSENT" || "${STATE}" == "DIED" ]]; then
    # 3 upload token --------------------------------------------------------
    remote_step prepare 60 GPUREF_PREPARED
    if [[ "${HAVE_TOKEN}" -eq 1 ]]; then
        tok="${LOCAL_TMP}/hf_token"
        ( umask 077; if [[ -n "${HF_TOKEN:-}" ]]; then printf '%s' "${HF_TOKEN}" > "${tok}"; else cp "${HOME}/.cache/huggingface/token" "${tok}"; fi )
        colab_cmd upload -s "${SESSION}" "${tok}" "${REMOTE_REL}/hf_token.upload" > /dev/null || { rm -f "${tok}"; die "HF token upload failed"; }
        rm -f "${tok}"
        remote_step install_token 60 GPUREF_TOKEN_INSTALLED
    fi
    # 4 launch the runner detached ------------------------------------------
    colab_cmd upload -s "${SESSION}" "${SCRIPT_DIR}/remote_runner.sh" "${REMOTE_REL}/remote_runner.sh"
    colab_cmd upload -s "${SESSION}" "${SCRIPT_DIR}/gpuref.py" "${REMOTE_REL}/gpuref.py"
    runner_args=(--ttis-sha "${TTIS_SHA}" --work-dir "/${REMOTE_REL}")
    [[ -n "${VLLM_VERSION}" ]] && runner_args+=(--vllm-version "${VLLM_VERSION}")
    remote_step launch 60 GPUREF_LAUNCHED "${runner_args[@]}" "${MODELS[@]}"
    STATE="RUNNING"
fi

# 5 poll (each poll is a kernel execution, which also keeps the VM alive) -----
if [[ "${DRY_RUN}" -eq 1 ]]; then log "poll every ${POLL_MINUTES} min until DONE/FAILED:"; remote_exec status 120; STATE="DONE"; fi
failures=0
while [[ "${STATE}" == "RUNNING" ]]; do
    sleep $((POLL_MINUTES * 60))
    if ! remote_state; then
        failures=$((failures + 1)); log "status poll failed (${failures}/6)"
        if (( failures >= 6 )); then
            colab sessions < /dev/null 2>&1 | has_session || { SESSION_ACTIVE=0; die "session '${SESSION}' is gone; results on the VM are lost"; }
            failures=0
        fi
        STATE="RUNNING"; continue
    fi
    failures=0; log "$(date '+%H:%M:%S') runner ${STATE}"
    [[ "${STATE}" == "DIED" ]] && STATE="FAILED"
    if [[ "${MIN_BALANCE}" != "0" && "${STATE}" == "RUNNING" ]] \
        && balance="$(colab usage < /dev/null 2>/dev/null | gpuref balance-below "${MIN_BALANCE}")"; then
        log "balance ${balance} CU is below --min-balance ${MIN_BALANCE}; collecting partial results"; STATE="FAILED"
    fi
done

# 6 download + summary ----------------------------------------------------------
remote_step pack 900 GPUREF_PACKED
mkdir -p "${OUT_DIR}"
colab_cmd download -s "${SESSION}" "${REMOTE_REL}/gpuref-results.tar.gz" "${OUT_DIR}/gpuref-results.tar.gz"
if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '+ tar -xzf %q -C %q\n+ gpuref.py summary %q\n' "${OUT_DIR}/gpuref-results.tar.gz" "${OUT_DIR}" "${OUT_DIR}/results" >&2
else
    tar -xzf "${OUT_DIR}/gpuref-results.tar.gz" -C "${OUT_DIR}"
    gpuref summary "${OUT_DIR}/results" | tee "${OUT_DIR}/summary.txt" || log "no TTIS eval reports in the results"
fi
[[ "${STATE}" == "DONE" ]] || { log "runner finished ${STATE}; partial results in ${OUT_DIR}"; exit 1; }
log "done: ${OUT_DIR}"
