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
# Unattended: --detach runs it in the background; every state change goes to
# <out>/driver_status.json and one line of $RESULTS_ROOT/gpuref_status.log.
# Exit status 75 means no GPU capacity (retry later); 1 any other failure.
#
# Only the `colab` calls and this flow live here: checks and parsers are in
# gpuref.py, VM code in snippets/. The HF token goes up as a 0600 temp file through
# `colab upload`: never on a command line, in a log, or in code sent with exec.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Where results land; set GPUREF_RESULTS_ROOT to keep them outside the checkout
# (e.g. when it lives under a /tmp that is cleared on reboot).
RESULTS_ROOT="${GPUREF_RESULTS_ROOT:-${REPO_ROOT}/workflow_logs/gpu_reference_colab}"
REMOTE_REL="content/gpuref"   # /content/gpuref on the VM, for upload/download
GPU="H100" HIGH_MEM=0 TTIS_REF="HEAD" OUT_DIR="" KEEP=0 SESSION="" POLL_MINUTES=5
VLLM_VERSION="" TRANSFORMERS_VERSION="" MIN_BALANCE=0 DRY_RUN=0 MODELS=() TASK_ARGS=()
SWEEP=0 REFRESH=0 WRITE_SPECS=0 PRUNE=0 TT_REPORTS="" AUDIT_REFS="" ORIG_ARGS=("$@")
NEW_ATTEMPTS=3 NEW_RETRY_SECONDS="${GPUREF_NEW_RETRY_SECONDS:-120}"   # per GPU type, on 503 (no capacity)
RETRY_CAPACITY="" DETACH=0             # --retry-capacity MINUTES,HOURS; --detach
SERVE=0 AGENTIC_TASKS="" TUNNEL_DOMAIN="" # --serve mode (agentic --task): set from the tasks
# --serve credentials (0600 files; never on a command line or in a log): the
# endpoint API key (the same value is the tt-shield secret SHIELD_SECRET) and
# the ngrok authtoken.
ENDPOINT_KEY_FILE="${GPUREF_ENDPOINT_KEY_FILE:-${HOME}/.config/gpuref/endpoint_api_key}"
NGROK_TOKEN_FILE="${GPUREF_NGROK_TOKEN_FILE:-${HOME}/.config/gpuref/ngrok_authtoken}"
SHIELD_SECRET="${GPUREF_SHIELD_SECRET:-TEST_GPUREF_ENDPOINT_API_KEY}"
SHIELD_REF="${GPUREF_SHIELD_REF:-main}"   # tt-shield ref whose workflow file is dispatched
EX_NO_CAPACITY=75

usage() {
    cat <<'EOF'
Usage: colab_gpu_reference.sh [options] MODEL [MODEL...]
       colab_gpu_reference.sh --sweep [options]   (targets picked from the catalog)
  --gpu G[,G...]      GPU types in preference order (default H100), e.g. H100,A100
  --high-mem          high-RAM machine shape
  --ttis-ref REF      pushed TTIS commit the VM checks out (default HEAD)
  --out DIR           results dir (default $GPUREF_RESULTS_ROOT or workflow_logs/gpu_reference_colab, /<session>)
  --keep              do not stop the VM on exit
  --session NAME      session name (default gpuref-<sha[:8]>); same name re-attaches
  --poll-minutes N    minutes between polls (default 5)
  --min-balance CU    stop (keeping partial results) below this compute-unit balance
  --vllm-version V    override every model's serving-stack vLLM (default: per model)
  --transformers-version V  override every model's transformers pin
  --task MODEL=T[,T]  run only these tasks of MODEL (each in full); repeatable,
                      explicit runs only. Default: all of the model's lm-eval tasks.
                      Agentic tasks (one model): --serve mode, see README "Agentic tasks"
  --tunnel-domain D   --serve: reserved ngrok domain, so the URL survives a rollover
  --retry-capacity M,H  when no GPU is free (503) or the concurrent-session limit
                      is hit (412), retry every M minutes for up to H hours, then
                      exit 75; a sweep moves on to other groups and retries later
  --detach            run in the background (log: $GPUREF_RESULTS_ROOT/gpuref_run.log)
  --dry-run           print every colab command instead of running it
 sweep mode:
  --sweep             measure every ungated or gpu_reference_requested eval task
  --refresh           also re-measure tasks that already have a GPU reference
  --refresh-quetzal R[,R]  audit every task of every model with a Quetzal P300X2
                      row here or on git refs R (e.g. origin/main,PR heads); "." = here only
  --write-specs       merge derived GPU specs into llm.yaml (then review, commit, push)
  --prune             with --write-specs: drop generated specs of unselected models
  --emit-patch        accepted for old commands; the proposed patch is always written
  --tt-reports DIR    Shield eval report JSONs, for TT scores next to GPU ones
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
        --transformers-version) TRANSFORMERS_VERSION="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;         -h|--help) usage; exit 0 ;;
        --retry-capacity) RETRY_CAPACITY="$2"; shift 2 ;;
        --detach) DETACH=1; shift ;;
        --task) TASK_ARGS+=("$2"); shift 2 ;;
        --tunnel-domain) TUNNEL_DOMAIN="$2"; shift 2 ;;
        --sweep) SWEEP=1; shift ;;             --refresh) REFRESH=1; shift ;;
        --refresh-quetzal) AUDIT_REFS="$2"; shift 2 ;;
        --write-specs) WRITE_SPECS=1; shift ;; --emit-patch) shift ;;  # always written now
        --prune) PRUNE=1; shift ;;
        --tt-reports) TT_REPORTS="$2"; shift 2 ;;
        -*) usage >&2; die "unknown option: $1" ;;
        *) MODELS+=("$1"); shift ;;
    esac
done
[[ ${#MODELS[@]} -gt 0 || "${SWEEP}" -eq 1 ]] || { usage >&2; die "at least one MODEL (HF repo id) is required"; }
[[ ${#MODELS[@]} -eq 0 || "${SWEEP}" -eq 0 ]] || die "--sweep picks its own models from the catalog"
for m in ${MODELS[@]+"${MODELS[@]}"}; do [[ "$m" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || die "not an HF repo id: $m"; done
[[ "${GPU}" =~ ^((T4|L4|G4|H100|A100),)*(T4|L4|G4|H100|A100)$ ]] || die "bad --gpu: ${GPU}"
[[ "${POLL_MINUTES}" =~ ^[1-9][0-9]*$ ]] || die "--poll-minutes must be a positive integer"
[[ "${MIN_BALANCE}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--min-balance must be a number"
[[ -z "${VLLM_VERSION}" || "${VLLM_VERSION}" =~ ^[0-9A-Za-z.+-]+$ ]] || die "bad --vllm-version"
[[ -z "${TRANSFORMERS_VERSION}" || "${TRANSFORMERS_VERSION}" =~ ^[0-9A-Za-z.+-]+$ ]] || die "bad --transformers-version"
[[ ${#TASK_ARGS[@]} -eq 0 || "${SWEEP}" -eq 0 ]] || die "--task is for explicit runs; a sweep picks its own tasks"
for t in ${TASK_ARGS[@]+"${TASK_ARGS[@]}"}; do
    [[ "$t" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+=[A-Za-z0-9_]+(,[A-Za-z0-9_]+)*$ ]] || die "--task is MODEL=TASK[,TASK]: $t"
    [[ " ${MODELS[*]} " == *" ${t%%=*} "* ]] || die "--task names a model that is not being run: ${t%%=*}"
done
[[ -z "${RETRY_CAPACITY}" || "${RETRY_CAPACITY}" =~ ^[1-9][0-9]*,[0-9]+([.][0-9]+)?$ ]] || die "--retry-capacity is MINUTES,HOURS (e.g. 15,8)"
[[ -z "${TUNNEL_DOMAIN}" || "${TUNNEL_DOMAIN}" =~ ^[A-Za-z0-9.-]+$ ]] || die "bad --tunnel-domain"
[[ "${SHIELD_SECRET}" =~ ^TEST_[A-Z0-9_]+$ ]] || die "GPUREF_SHIELD_SECRET must be a TEST_* tt-shield secret name"
[[ "${SHIELD_REF}" =~ ^[A-Za-z0-9._/-]+$ ]] || die "bad GPUREF_SHIELD_REF"
if [[ "${DETACH}" -eq 1 && "${DRY_RUN}" -eq 0 ]]; then   # re-run in the background, own session
    mkdir -p "${RESULTS_ROOT}"
    args=(); for a in "${ORIG_ARGS[@]}"; do [[ "$a" == "--detach" ]] || args+=("$a"); done
    nohup perl -e 'use POSIX; POSIX::setsid(); exec @ARGV' "$0" "${args[@]}" \
        >> "${RESULTS_ROOT}/gpuref_run.log" 2>&1 < /dev/null &
    log "detached (pid $!); log ${RESULTS_ROOT}/gpuref_run.log, status ${RESULTS_ROOT}/gpuref_status.log"
    exit 0
fi

LOCAL_TMP="$(mktemp -d "${TMPDIR:-/tmp}/gpuref.XXXXXX")"; chmod 700 "${LOCAL_TMP}"
SESSION_ACTIVE=0
cleanup() {   # step 7: always runs
    local rc=$?
    # Any other exit (die, signal) still leaves a final state for monitors.
    if [[ "${rc}" -ne 0 && -z "${TERMINAL:-}" ]] && declare -F status > /dev/null; then status FAILED "rc=${rc}"; fi
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
status() {   # STATE [KEY=VALUE...]: driver_status.json + one line of gpuref_status.log
    [[ "${DRY_RUN}" -eq 1 || -z "${STATUS_FILE:-}" ]] && return 0
    case "$1" in DONE|FAILED|NO_CAPACITY|SWEEP_DONE|SWEEP_INCOMPLETE) TERMINAL=1 ;; esac
    gpuref driver-status --log "${RESULTS_ROOT}/gpuref_status.log" "${STATUS_FILE}" \
        "state=$1" "session=${SESSION:-sweep}" "ttis=${TTIS_SHA:0:8}" "${@:2}" || true
}
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
    local name="$1" timeout="$2" marker="$3" out attempt; shift 3
    if [[ "${DRY_RUN}" -eq 1 ]]; then remote_exec "${name}" "${timeout}" "$@"; return 0; fi
    for attempt in 1 2 3; do
        out="$(remote_exec "${name}" "${timeout}" "$@" 2>&1)" || true
        printf '%s\n' "${out}" | grep "^GPUREF_" >&2 || true
        printf '%s\n' "${out}" | grep -q "^${marker}" && return 0
        # A lost kernel websocket is transient. Every step but launch is
        # idempotent; launch may have started the runner, so check first.
        printf '%s\n' "${out}" | grep -q "Connection was lost" || break
        if [[ "${name}" == "launch" ]] && remote_state && [[ "${STATE}" == "RUNNING" || "${STATE}" == "SERVING" ]]; then return 0; fi
        log "VM step '${name}': connection lost (attempt ${attempt}/3); retrying in 30s"; sleep 30
    done
    die "VM step '${name}' failed:"$'\n'"${out}"
}
has_session() { awk -v want="[${SESSION}]" '$1 == want { f = 1 } END { exit !f }'; }
remote_state() {   # sets STATE and NDONE (models finished); fails when unreadable
    local out view; out="$(remote_exec status 120 2>&1)" || return 1
    view="$(printf '%s\n' "${out}" | gpuref status-view)" || return 1
    read -r STATE NDONE PHASE <<< "${view}"
}
fetch_results() {   # pack on the VM, download, extract into OUT_DIR
    remote_step pack 900 GPUREF_PACKED
    mkdir -p "${OUT_DIR}"
    colab_cmd download -s "${SESSION}" "${REMOTE_REL}/gpuref-results.tar.gz" "${OUT_DIR}/gpuref-results.tar.gz"
    if [[ "${DRY_RUN}" -eq 1 ]]; then printf '+ tar -xzf %q -C %q\n' "${OUT_DIR}/gpuref-results.tar.gz" "${OUT_DIR}" >&2
    else tar -xzf "${OUT_DIR}/gpuref-results.tar.gz" -C "${OUT_DIR}"; fi
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

if [[ "${SWEEP}" -eq 1 ]]; then   # plan from the catalog, then one session per GPU group
    ROOT="${RESULTS_ROOT}"; SWEEP_DIR="${ROOT}/sweep-${TTIS_SHA:0:8}"
    mkdir -p "${SWEEP_DIR}"; STATUS_FILE="${SWEEP_DIR}/driver_status.json"; status SWEEP_PLANNING
    plan_args=(sweep plan --sha "${TTIS_SHA}" --plan "${SWEEP_DIR}/sweep_plan.json" --results-root "${ROOT}")
    if [[ "${REFRESH}" -eq 1 ]]; then plan_args+=(--refresh); fi
    if [[ -n "${AUDIT_REFS}" ]]; then
        IFS=',' read -r -a audit_refs <<< "${AUDIT_REFS}"
        plan_args+=(--refresh-quetzal)
        for ref in "${audit_refs[@]}"; do [[ "${ref}" == "." ]] || plan_args+=("${ref}"); done
    fi
    if [[ "${WRITE_SPECS}" -eq 1 ]]; then plan_args+=(--write-specs); fi
    if [[ "${PRUNE}" -eq 1 ]]; then plan_args+=(--prune); fi
    # The eval fingerprint covers the whole stack, so plan with the one the runner gets.
    if [[ -n "${VLLM_VERSION}" ]]; then plan_args+=(--vllm-version "${VLLM_VERSION}"); fi
    if [[ -n "${TRANSFORMERS_VERSION}" ]]; then plan_args+=(--transformers-version "${TRANSFORMERS_VERSION}"); fi
    rc=0; gpuref "${plan_args[@]}" > "${SWEEP_DIR}/groups.tsv" || rc=$?
    [[ "${rc}" -ne 2 ]] || die "some targets need derived GPU specs: re-run with --write-specs, review and commit the llm.yaml change, push, and run again"
    [[ "${rc}" -eq 0 ]] || die "sweep plan failed"
    printf 'Sweep at TTIS %s, started %s. Plan: %s\nResume (finished models are skipped, running sessions re-attached):\n  %q' \
        "${TTIS_SHA}" "$(date)" "${SWEEP_DIR}/sweep_plan.json" "$0" > "${ROOT}/RESUME.txt"
    printf ' %q' ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"} >> "${ROOT}/RESUME.txt"
    # Pin the sha: "finished" means finished at this sha, so a resume after new
    # commits must still run (and skip) against the same one.
    [[ " ${ORIG_ARGS[*]-} " == *" --ttis-ref "* ]] || printf ' --ttis-ref %s' "${TTIS_SHA}" >> "${ROOT}/RESUME.txt"
    printf '\n' >> "${ROOT}/RESUME.txt"
    child=(--ttis-ref "${TTIS_SHA}" --poll-minutes "${POLL_MINUTES}" --min-balance "${MIN_BALANCE}")
    if [[ -n "${VLLM_VERSION}" ]]; then child+=(--vllm-version "${VLLM_VERSION}"); fi
    if [[ -n "${TRANSFORMERS_VERSION}" ]]; then child+=(--transformers-version "${TRANSFORMERS_VERSION}"); fi
    if [[ "${KEEP}" -eq 1 ]]; then child+=(--keep); fi
    if [[ "${HIGH_MEM}" -eq 1 ]]; then child+=(--high-mem); fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then child+=(--dry-run); fi
    # Group fallback: a group with no capacity (exit 75) is deferred so the
    # others run; deferred groups are retried at the end, with --retry-capacity.
    run_group() {   # GPUS MODELS RETRY -> the child's exit status
        local gpus="$1" models="$2" retry="$3" label; local -a group_models extra=()
        read -r -a group_models <<< "${models}"
        label="$( [[ "${gpus}" == *A100* ]] && echo a100 || echo h100 )"
        [[ -n "${retry}" ]] && extra=(--retry-capacity "${retry}")
        log "sweep group ${gpus}: ${models}"; status SWEEP_GROUP "group=${gpus}"
        "$0" "${child[@]}" ${extra[@]+"${extra[@]}"} --gpu "${gpus}" \
            --session "gpuref-sweep-${label}-${TTIS_SHA:0:8}" "${group_models[@]}"
    }
    balance_low() {
        [[ "${MIN_BALANCE}" != "0" && "${DRY_RUN}" -eq 0 ]] \
            && colab usage < /dev/null 2>/dev/null | gpuref balance-below "${MIN_BALANCE}" > /dev/null
    }
    rc=0; deferred=()
    while IFS=$'\t' read -r gpus models; do
        [[ -n "${models}" ]] || continue
        if balance_low; then log "balance below --min-balance ${MIN_BALANCE}; not starting the ${gpus} group"; status LOW_BALANCE "group=${gpus}"; rc=1; continue; fi
        grc=0; run_group "${gpus}" "${models}" "" || grc=$?
        if [[ "${grc}" -eq "${EX_NO_CAPACITY}" ]]; then log "no ${gpus} capacity; deferring the group"; status DEFERRED "group=${gpus}"; deferred+=("${gpus}"$'\t'"${models}")
        elif [[ "${grc}" -ne 0 ]]; then rc=1; fi
    done < "${SWEEP_DIR}/groups.tsv"
    for entry in ${deferred[@]+"${deferred[@]}"}; do
        gpus="${entry%%$'\t'*}"; models="${entry#*$'\t'}"
        if balance_low; then status LOW_BALANCE "group=${gpus}"; rc=1; continue; fi
        log "retrying deferred group ${gpus}${RETRY_CAPACITY:+ (every ${RETRY_CAPACITY%,*} min for up to ${RETRY_CAPACITY#*,} h)}"
        run_group "${gpus}" "${models}" "${RETRY_CAPACITY}" || rc=1
    done
    publish=(publish --plan "${SWEEP_DIR}/sweep_plan.json" --out-dir "${SWEEP_DIR}" --results-root "${ROOT}")
    if [[ -n "${TT_REPORTS}" ]]; then publish+=(--tt-reports "${TT_REPORTS}"); fi
    gpuref "${publish[@]}" > /dev/null || rc=1
    status "$([[ ${rc} -eq 0 ]] && echo SWEEP_DONE || echo SWEEP_INCOMPLETE)" "outputs=${SWEEP_DIR}"
    log "sweep outputs: ${SWEEP_DIR}/{sweep_summary.json,sweep_5353.md,eval_config.patch} (rendered only; post with gpuref.py publish --post after approval)"
    exit "${rc}"
fi
SESSION="${SESSION:-gpuref-${TTIS_SHA:0:8}}"
[[ "${SESSION}" =~ ^[A-Za-z0-9._-]+$ ]] || die "bad --session: ${SESSION}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/${SESSION}}"
STATUS_FILE="${OUT_DIR}/driver_status.json"
log "models: ${MODELS[*]}; ttis ${TTIS_SHA}; session ${SESSION}; out ${OUT_DIR}"
status STARTING "models=${#MODELS[@]}"
usable="$(gpuref preflight --gpus "${GPU}" "${MODELS[@]}")" || die "preflight failed (see above)"
# --task: fail fast against this checkout's EvalConfigs (the runner re-checks at --ttis-ref).
for t in ${TASK_ARGS[@]+"${TASK_ARGS[@]}"}; do
    kind="$(gpuref task-kind "${t%%=*}" "${t#*=}")" || die "--task ${t} (see above)"
    if [[ "${kind}" == "agentic" ]]; then
        SERVE=1; AGENTIC_TASKS="${t#*=}"
        log "${t%%=*}: agentic ${AGENTIC_TASKS}: --serve (Shield runs each task against this VM's server)"
        continue
    fi
    sel="$(gpuref task-selection "${t%%=*}" "${t#*=}")" || die "--task ${t} (see above)"
    log "${t%%=*}: only ${sel} (each task in full)"
done
if [[ "${SERVE}" -eq 1 ]]; then
    [[ ${#MODELS[@]} -eq 1 && ${#TASK_ARGS[@]} -eq 1 ]] || die "--serve (agentic tasks) runs one model with one --task"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        for f in "${ENDPOINT_KEY_FILE}" "${NGROK_TOKEN_FILE}"; do
            [[ -s "${f}" ]] || die "--serve needs ${f} (README: Agentic tasks)"
            [[ "$(stat -c %a "${f}" 2>/dev/null || stat -f %Lp "${f}")" == "600" ]] || die "${f} must be mode 0600"
        done
        command -v gh >/dev/null 2>&1 || die "--serve dispatches Shield's endpoint tests with gh"
    fi
fi
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
start_session() {   # sets STATE (ABSENT for a new VM), SESSION_ACTIVE
    STATE="ABSENT"
    if [[ "${DRY_RUN}" -eq 0 ]] && printf '%s\n' "${SESSIONS_OUT}" | has_session; then
        SESSION_ACTIVE=1
        remote_state || die "could not read the runner state on '${SESSION}'"
        log "session '${SESSION}' exists; runner ${STATE}"
        return 0
    fi
    # Capacity: each pass tries every GPU in --gpu (3 attempts on 503 each).
    # With --retry-capacity M,H, passes repeat every M minutes for up to H hours
    # on 503 (no capacity) or 412 (concurrent-session limit); then exit 75.
    local interval_min=0 deadline=0 capacity_only attempt delay
    if [[ -n "${RETRY_CAPACITY}" ]]; then
        interval_min="${RETRY_CAPACITY%,*}"
        deadline=$(( $(date +%s) + $(awk -v h="${RETRY_CAPACITY#*,}" 'BEGIN { printf "%d", h * 3600 }') ))
    fi
    while :; do
        capacity_only=1
        for gpu in "${GPUS[@]}"; do
            new_args=(new -s "${SESSION}" --gpu "${gpu}"); [[ "${HIGH_MEM}" -eq 1 ]] && new_args+=(--high-mem)
            for ((attempt = 1, delay = NEW_RETRY_SECONDS; attempt <= NEW_ATTEMPTS; attempt++, delay *= 2)); do
                log "creating '${SESSION}' on ${gpu} (attempt ${attempt}/${NEW_ATTEMPTS})"
                if colab_cmd "${new_args[@]}" > "${LOCAL_TMP}/new.out" 2>&1; then cat "${LOCAL_TMP}/new.out" >&2; SESSION_ACTIVE=1; break 3; fi
                tail -3 "${LOCAL_TMP}/new.out" >&2
                if colab sessions < /dev/null 2>&1 | has_session; then SESSION_ACTIVE=1; die "'colab new' failed but the session exists"; fi
                if grep -qiE "Precondition Failed|\b412\b" "${LOCAL_TMP}/new.out"; then log "concurrent-session limit (412)"; break; fi
                if ! grep -q "Service Unavailable" "${LOCAL_TMP}/new.out"; then capacity_only=0; break; fi
                (( attempt == NEW_ATTEMPTS )) && break
                log "no ${gpu} capacity (503); retrying in ${delay}s"; sleep "${delay}"
            done
        done
        [[ "${capacity_only}" -eq 1 ]] || die "'colab new' failed on every GPU in --gpu (${GPUS[*]}), not for capacity (see above)"
        if (( $(date +%s) + interval_min * 60 > deadline )); then
            status NO_CAPACITY "gpus=${GPU}"
            log "ERROR: no capacity on ${GPUS[*]}${RETRY_CAPACITY:+ within ${RETRY_CAPACITY#*,} h}; exit ${EX_NO_CAPACITY} (retry later)"
            exit "${EX_NO_CAPACITY}"
        fi
        status WAITING_CAPACITY "gpus=${GPU}"
        log "no capacity on ${GPUS[*]}; next pass in ${interval_min} min"; sleep $(( interval_min * 60 ))
    done
    log "allocated ${gpu} (the exact GPU is recorded in provenance.json)"
}

upload_secret() {   # LOCAL_FILE REMOTE_NAME: a 0600 copy, through `colab upload`
    local tmp="${LOCAL_TMP}/$2"
    if [[ "${DRY_RUN}" -eq 1 ]]; then colab_cmd upload -s "${SESSION}" "$1" "${REMOTE_REL}/$2"; return 0; fi
    ( umask 077; cp "$1" "${tmp}" )
    colab_cmd upload -s "${SESSION}" "${tmp}" "${REMOTE_REL}/$2" > /dev/null || { rm -f "${tmp}"; die "upload of $2 failed"; }
    rm -f "${tmp}"
}

launch_runner() {   # steps 3-4 on a fresh (or dead-runner) VM
    # 3 upload token --------------------------------------------------------
    remote_step prepare 60 GPUREF_PREPARED
    if [[ "${HAVE_TOKEN}" -eq 1 ]]; then
        tok="${LOCAL_TMP}/hf_token"
        ( umask 077; if [[ -n "${HF_TOKEN:-}" ]]; then printf '%s' "${HF_TOKEN}" > "${tok}"; else cp "${HOME}/.cache/huggingface/token" "${tok}"; fi )
        colab_cmd upload -s "${SESSION}" "${tok}" "${REMOTE_REL}/hf_token.upload" > /dev/null || { rm -f "${tok}"; die "HF token upload failed"; }
        rm -f "${tok}"
        remote_step install_token 60 GPUREF_TOKEN_INSTALLED
    fi
    if [[ "${SERVE}" -eq 1 ]]; then
        upload_secret "${ENDPOINT_KEY_FILE}" endpoint_api_key.upload
        upload_secret "${NGROK_TOKEN_FILE}" ngrok_authtoken.upload
        remote_step install_serve_secrets 60 GPUREF_SERVE_SECRETS_INSTALLED
    fi
    # 4 launch the runner detached ------------------------------------------
    colab_cmd upload -s "${SESSION}" "${SCRIPT_DIR}/remote_runner.sh" "${REMOTE_REL}/remote_runner.sh"
    colab_cmd upload -s "${SESSION}" "${SCRIPT_DIR}/gpuref.py" "${REMOTE_REL}/gpuref.py"
    runner_args=(--ttis-sha "${TTIS_SHA}" --work-dir "/${REMOTE_REL}")
    [[ -n "${VLLM_VERSION}" ]] && runner_args+=(--vllm-version "${VLLM_VERSION}")
    [[ -n "${TRANSFORMERS_VERSION}" ]] && runner_args+=(--transformers-version "${TRANSFORMERS_VERSION}")
    for t in ${TASK_ARGS[@]+"${TASK_ARGS[@]}"}; do runner_args+=(--task "$t"); done
    if [[ "${SERVE}" -eq 1 ]]; then
        runner_args+=(--serve); [[ -n "${TUNNEL_DOMAIN}" ]] && runner_args+=(--tunnel-domain "${TUNNEL_DOMAIN}")
    fi
    remote_step launch 60 GPUREF_LAUNCHED "${runner_args[@]}" "${MODELS[@]}"
    STATE="RUNNING"
}

# --serve record state lives on this machine (<out>/shield/<task>.json), so it is
# read before any VM is started: a resume whose every Shield run has already
# finished only needs reports, never a new VM (see serve_report_only_resume).
if [[ "${SERVE}" -eq 1 ]]; then
    read -r SHIELD_REPO SHIELD_WORKFLOW < <(gpuref shield-workflow)
    ENDPOINT_FILE="${OUT_DIR}/endpoint.url" SHIELD_DIR="${OUT_DIR}/shield" ROLLOVERS=0
    SERVE_PROV="${OUT_DIR}/results/${MODELS[0]//\//__}/provenance.json"
    IFS=',' read -r -a SERVE_TASKS <<< "${AGENTIC_TASKS}"
    mkdir -p "${SHIELD_DIR}"
    field() { "${PYRUN[@]}" -c 'import json,sys; print((json.load(open(sys.argv[1])) or {}).get(sys.argv[2]) or "")' "$1" "$2" 2>/dev/null || true; }
    # >>> serve-record helpers (tests/test_gpu_reference_colab.py runs this block
    # under set -euo pipefail with field/gh/gpuref stubbed)
    record_run_id() {   # REC: the task's recorded Shield run id; empty, status 0, before any record
        local id=""
        if [[ -s "$1" ]]; then id="$(field "$1" run_id)"; fi
        printf '%s' "${id}"
    }
    report_on_disk() {   # DIR: a downloaded report*.json is under DIR
        [[ -d "$1" ]] && [[ -n "$(find "$1" -type f -name 'report*.json' -print -quit 2>/dev/null)" ]]
    }
    retrieve_report() {   # TASK REC RUN_ID CONCLUSION: download the run's report, record the outcome
        # A completed run is recorded "completed" only once its report is on disk;
        # otherwise "report_pending", which a re-run retries instead of skipping.
        local task="$1" rec="$2" run_id="$3" conclusion="$4" state=completed shown endpoint_id dispatched test
        shown="$(field "${rec}" endpoint)" endpoint_id="$(field "${rec}" endpoint_id)"
        dispatched="$(field "${rec}" dispatched_at)" test="$(field "${rec}" test)"
        rm -rf "${SHIELD_DIR:?}/${task}"; mkdir -p "${SHIELD_DIR}/${task}"
        gh run download "${run_id}" -R "${SHIELD_REPO}" -D "${SHIELD_DIR}/${task}" -p '*report*' > /dev/null 2>&1 || true
        report_on_disk "${SHIELD_DIR}/${task}" || state=report_pending
        gpuref driver-status "${rec}" "task=${task}" "run_id=${run_id}" "state=${state}" \
            "conclusion=${conclusion}" "url=https://github.com/${SHIELD_REPO}/actions/runs/${run_id}" \
            "endpoint=${shown}" "endpoint_id=${endpoint_id}" "ttis=${TTIS_SHA}" \
            "dispatched_at=${dispatched}" "test=${test}"
        [[ "${state}" == "completed" ]]
    }
    retry_report() {   # TASK REC RUN_ID CONCLUSION: retrieve_report, REPORT_TRIES times a minute apart
        local try
        for try in $(seq 1 "${REPORT_TRIES:-10}"); do
            retrieve_report "$@" && return 0
            log "$1: report for Shield run $3 not downloaded (try ${try}/${REPORT_TRIES:-10})"
            [[ "${try}" -lt "${REPORT_TRIES:-10}" ]] && sleep "${REPORT_RETRY_SECONDS:-60}"
        done
        return 1
    }
    all_runs_finished() {   # every task has a record whose Shield run finished (completed / report_pending)
        local task rec state
        for task in "${SERVE_TASKS[@]}"; do
            rec="${SHIELD_DIR}/${task}.json"
            [[ -s "${rec}" ]] || return 1
            state="$(field "${rec}" state)"
            [[ "${state}" == "completed" || "${state}" == "report_pending" ]] || return 1
        done
    }
    finish_serve() {   # the verdict from the recorded runs against SERVE_PROV's serving provenance
        # shield-provenance keeps the VM's serving_status as written and
        # recomputes status from it plus the reports on disk, so a finish after a
        # report_pending pass grades the recovered reports, not that pass's result.
        local verdict
        [[ -s "${SERVE_PROV}" ]] || { status FAILED "out=${OUT_DIR}" "reason=no serving provenance"; log "no ${SERVE_PROV}: the serving VM's provenance was never fetched; cannot finish without it"; return 1; }
        verdict="$(gpuref shield-provenance --prov "${SERVE_PROV}" --shield-dir "${SHIELD_DIR}" \
            --records "${SHIELD_DIR}"/*.json --rollovers "${ROLLOVERS}")" || verdict="failed"
        gpuref summary "${SHIELD_DIR}" | tee "${OUT_DIR}/summary.txt" || true
        [[ "${verdict}" == "ok" ]] || { status FAILED "out=${OUT_DIR}" "rollovers=${ROLLOVERS}"; log "a task has no Shield score; see ${SHIELD_DIR}"; return 1; }
        status DONE "tasks=${AGENTIC_TASKS}" "rollovers=${ROLLOVERS}" "out=${OUT_DIR}"
        log "done: ${OUT_DIR}"
    }
    serve_report_only_resume() {   # 0 = handled (exit with FINISH_RC), 1 = a task still needs a serving VM
        # Every Shield run already finished on an earlier pass: the scores belong
        # to that pass's VM, whose GPU and serving settings are in SERVE_PROV.
        # Fetch any missing report and finish against that provenance, with no
        # new VM (a new one could be a different GPU and would overwrite it).
        local task rec
        all_runs_finished || return 1
        log "every Shield run already finished; retrieving reports only (no VM is started)"
        for task in "${SERVE_TASKS[@]}"; do
            rec="${SHIELD_DIR}/${task}.json"
            report_on_disk "${SHIELD_DIR}/${task}" && [[ "$(field "${rec}" state)" == "completed" ]] && continue
            retry_report "${task}" "${rec}" "$(field "${rec}" run_id)" "$(field "${rec}" conclusion)" \
                || log "WARNING: ${task}: report still not retrievable; the record stays report_pending"
        done
        FINISH_RC=0; finish_serve || FINISH_RC=1
        return 0
    }
    keep_serving_provenance() {   # after fetch_results: never let a later VM replace SERVE_PROV
        # $1 = SERVE_PROV saved before the fetch (empty if there was none).
        local saved="$1" fetched
        [[ -n "${saved}" && -s "${saved}" ]] || return 0
        if [[ -s "${SERVE_PROV}" ]] && ! cmp -s "${saved}" "${SERVE_PROV}"; then
            fetched="${SERVE_PROV%.json}.refetch-$(date -u +%Y%m%dT%H%M%SZ).json"
            mv "${SERVE_PROV}" "${fetched}"
            log "WARNING: the fetched provenance differs from the one the scores were measured with; kept the original, saved the new one as ${fetched}"
        fi
        cp -p "${saved}" "${SERVE_PROV}"
    }
    # <<< serve-record helpers
    if [[ "${DRY_RUN}" -eq 0 ]] && serve_report_only_resume; then exit "${FINISH_RC}"; fi
fi

start_session
if [[ "${STATE}" == "ABSENT" || "${STATE}" == "DIED" ]]; then launch_runner; fi

# 5s --serve: serve until Shield's endpoint tests have run every task -----------
# One Shield run per agentic task, dispatched once the VM serves through the
# tunnel and recorded in <out>/shield/<task>.json, so a re-run of this command
# resumes (never re-dispatches a recorded run on the same URL). Polls keep the
# VM alive; a lost VM or a runner that stopped serving rolls over to a new
# session. A new URL (no --tunnel-domain) cancels the task's run and dispatches
# it again; a reserved domain keeps it.
if [[ "${SERVE}" -eq 1 ]]; then
    serve_endpoint() {   # the VM's tunnel URL -> ENDPOINT_FILE (0600); sets ENDPOINT_ID, ENDPOINT_SHOWN
        local out url
        out="$(remote_exec endpoint 60 2>&1)" || return 1
        url="$(printf '%s\n' "${out}" | sed -n 's/^ENDPOINT_URL=//p' | tail -1)"
        [[ "${url}" =~ ^https://[A-Za-z0-9.-]+/?$ ]] || return 1
        ( umask 077; printf '%s\n' "${url}" > "${ENDPOINT_FILE}" )
        read -r ENDPOINT_SHOWN ENDPOINT_ID < <(gpuref redact-url --url-file "${ENDPOINT_FILE}")
    }
    rollover() {   # REASON: replace the serving VM and relaunch the runner
        ROLLOVERS=$((ROLLOVERS + 1)); log "rollover ${ROLLOVERS}: $1"; status ROLLOVER "reason=$1" "rollovers=${ROLLOVERS}"
        [[ "${SESSION_ACTIVE}" -eq 1 ]] && { colab_cmd stop -s "${SESSION}" || true; }
        SESSION_ACTIVE=0 SESSIONS_OUT=""
        start_session; launch_runner
    }
    shield_dispatch() {   # TASK REC: start the task's Shield run against ENDPOINT_FILE
        local task="$1" rec="$2" since runs id=""
        since="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        gpuref shield-inputs --model "${MODELS[0]}" --task "${task}" --url-file "${ENDPOINT_FILE}" \
            --ttis-sha "${TTIS_SHA}" --secret "${SHIELD_SECRET}" > "${LOCAL_TMP}/inputs.json" || die "shield inputs for ${task}"
        gh workflow run "${SHIELD_WORKFLOW}" -R "${SHIELD_REPO}" --ref "${SHIELD_REF}" --json < "${LOCAL_TMP}/inputs.json" > /dev/null \
            || die "could not dispatch ${SHIELD_WORKFLOW} on ${SHIELD_REPO}"
        for _ in $(seq 1 20); do
            sleep 15
            runs="$(gh run list -R "${SHIELD_REPO}" --workflow "${SHIELD_WORKFLOW}" --event workflow_dispatch \
                --limit 40 --json databaseId,displayTitle,createdAt 2>/dev/null)" || continue
            id="$(printf '%s' "${runs}" | gpuref shield-find-run --model "${MODELS[0]}" --url-file "${ENDPOINT_FILE}" --since "${since}")" && break
        done
        [[ -n "${id}" ]] || die "dispatched ${task} but found no matching Shield run"
        gpuref driver-status "${rec}" "task=${task}" "run_id=${id}" "state=dispatched" \
            "url=https://github.com/${SHIELD_REPO}/actions/runs/${id}" "endpoint=${ENDPOINT_SHOWN}" \
            "endpoint_id=${ENDPOINT_ID}" "dispatched_at=${since}" "ttis=${TTIS_SHA}" "test=$(printf 'inference-workflow-agentic --device gpu --agentic-benchmark %s' "${task}")"
        log "${task}: Shield run https://github.com/${SHIELD_REPO}/actions/runs/${id} against ${ENDPOINT_SHOWN}"
    }
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        for task in "${SERVE_TASKS[@]}"; do
            log "would dispatch ${SHIELD_WORKFLOW} on ${SHIELD_REPO} for ${task} once the VM serves:"
            printf 'https://example.invalid\n' > "${LOCAL_TMP}/url"
            gpuref shield-inputs --model "${MODELS[0]}" --task "${task}" --url-file "${LOCAL_TMP}/url" \
                --ttis-sha "${TTIS_SHA}" --secret "${SHIELD_SECRET}" >&2
        done
        exit 0
    fi
    failures=0
    for task in "${SERVE_TASKS[@]}"; do
        rec="${SHIELD_DIR}/${task}.json"
        rec_state=""
        if [[ -s "${rec}" ]]; then rec_state="$(field "${rec}" state)"; fi
        if [[ "${rec_state}" == "completed" ]] && report_on_disk "${SHIELD_DIR}/${task}"; then
            log "${task}: Shield run already completed"; continue
        fi
        if [[ "${rec_state}" == "completed" || "${rec_state}" == "report_pending" ]]; then
            # The run finished on an earlier pass but its report is not on disk:
            # retrieve it, never dispatch the task again.
            if retry_report "${task}" "${rec}" "$(field "${rec}" run_id)" "$(field "${rec}" conclusion)"; then
                log "${task}: report for Shield run $(field "${rec}" run_id) retrieved"
            else
                log "WARNING: ${task}: report still not retrievable; the record stays report_pending for the next run"
            fi
            continue
        fi
        while :; do
            if ! remote_state; then
                failures=$((failures + 1)); log "status poll failed (${failures}/6)"
                if (( failures >= 6 )); then
                    failures=0
                    colab sessions < /dev/null 2>&1 | has_session || { SESSION_ACTIVE=0; rollover "session '${SESSION}' is gone"; }
                fi
                sleep 60; continue
            fi
            failures=0
            case "${STATE}" in
                SERVING) ;;
                RUNNING)   # model download / load: one status line per phase
                    [[ "${PHASE:-}" == "${last_phase:-}" ]] || status STARTING "phase=${PHASE:-}" "task=${task}"
                    last_phase="${PHASE:-}"; sleep 60; continue ;;
                *) rollover "runner ${STATE} before STOP (${PHASE:-})"; continue ;;
            esac
            serve_endpoint || { log "no tunnel URL yet"; sleep 60; continue; }
            run_id="$(record_run_id "${rec}")"
            if [[ -n "${run_id}" && "$(field "${rec}" endpoint_id)" != "${ENDPOINT_ID}" ]]; then
                log "${task}: endpoint changed (rollover without --tunnel-domain); cancelling run ${run_id} and dispatching again"
                gh run cancel "${run_id}" -R "${SHIELD_REPO}" > /dev/null 2>&1 || true
                run_id=""
            fi
            if [[ -z "${run_id}" ]]; then shield_dispatch "${task}" "${rec}"; run_id="$(field "${rec}" run_id)"; fi
            view="$(gh run view "${run_id}" -R "${SHIELD_REPO}" --json status,conclusion -q '.status+" "+.conclusion' 2>/dev/null)" || view="unknown"
            read -r run_status run_conclusion <<< "${view}"
            status SERVING "task=${task}" "shield_run=${run_id}" "shield_status=${run_status}" \
                "endpoint=${ENDPOINT_SHOWN}" "rollovers=${ROLLOVERS}"
            if [[ "${run_status}" == "completed" ]]; then
                if retry_report "${task}" "${rec}" "${run_id}" "${run_conclusion}"; then
                    log "${task}: Shield run ${run_id} completed (${run_conclusion})"
                else
                    log "WARNING: ${task}: Shield run ${run_id} completed but its report was not downloaded; recorded report_pending, a re-run retries it"
                fi
                break
            fi
            if [[ "${MIN_BALANCE}" != "0" ]] && balance="$(colab usage < /dev/null 2>/dev/null | gpuref balance-below "${MIN_BALANCE}")"; then
                log "balance ${balance} CU is below --min-balance ${MIN_BALANCE}; cancelling Shield run ${run_id} and stopping"
                gh run cancel "${run_id}" -R "${SHIELD_REPO}" > /dev/null 2>&1 || true
                status LOW_BALANCE "balance=${balance}" "task=${task}"; exit 1
            fi
            sleep $((POLL_MINUTES * 60))
        done
    done
    # Every task ran: stop serving, bring home the VM's logs, finish provenance.
    remote_step stop 60 GPUREF_STOP_REQUESTED
    for _ in $(seq 1 40); do remote_state && [[ "${STATE}" == "DONE" || "${STATE}" == "FAILED" ]] && break; sleep 15; done
    saved_prov=""
    if [[ -s "${SERVE_PROV}" ]]; then saved_prov="${LOCAL_TMP}/provenance.kept.json"; cp -p "${SERVE_PROV}" "${saved_prov}"; fi
    fetch_results
    keep_serving_provenance "${saved_prov}"
    finish_serve || exit 1
    exit 0
fi

# 5 poll (each poll is a kernel execution, which also keeps the VM alive) -----
if [[ "${DRY_RUN}" -eq 1 ]]; then log "poll every ${POLL_MINUTES} min until DONE/FAILED:"; remote_exec status 120; STATE="DONE"; fi
failures=0 FETCHED=0 NDONE=0
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
    failures=0; log "$(date '+%H:%M:%S') runner ${STATE}, ${NDONE} model(s) finished"
    status "${STATE}" "done=${NDONE}/${#MODELS[@]}" "phase=${PHASE:-}"
    # Bring each finished model home at once, so a lost VM loses at most one.
    if [[ "${STATE}" == "RUNNING" && "${NDONE}" -gt "${FETCHED}" ]]; then fetch_results; FETCHED="${NDONE}"; fi
    [[ "${STATE}" == "DIED" ]] && STATE="FAILED"
    if [[ "${MIN_BALANCE}" != "0" && "${STATE}" == "RUNNING" ]] \
        && balance="$(colab usage < /dev/null 2>/dev/null | gpuref balance-below "${MIN_BALANCE}")"; then
        log "balance ${balance} CU is below --min-balance ${MIN_BALANCE}; collecting partial results"; STATE="FAILED"
        status LOW_BALANCE "balance=${balance}" "done=${NDONE}/${#MODELS[@]}"
    fi
done

# 6 download + summary ----------------------------------------------------------
fetch_results
if [[ "${DRY_RUN}" -eq 1 ]]; then printf '+ gpuref.py summary %q\n' "${OUT_DIR}/results" >&2
else gpuref summary "${OUT_DIR}/results" | tee "${OUT_DIR}/summary.txt" || log "no TTIS eval reports in the results"; fi
[[ "${STATE}" == "DONE" ]] || { status FAILED "runner=${STATE}" "out=${OUT_DIR}"; log "runner finished ${STATE}; partial results in ${OUT_DIR}"; exit 1; }
status DONE "done=${#MODELS[@]}/${#MODELS[@]}" "out=${OUT_DIR}"
log "done: ${OUT_DIR}"
