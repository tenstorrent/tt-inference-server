#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# colab_gpu_reference.sh -- collect GPU reference eval scores (gpu_reference_score)
# on a Google Colab GPU VM with the Colab CLI (https://github.com/googlecolab/google-colab-cli).
#
# For each MODEL the VM serves the HF repo with upstream vLLM and runs TTIS's
# own `run.py --workflow evals --tt-device gpu` against it, so the GPU numbers
# come from the exact TTIS task configs the Tenstorrent runs use. See README.md
# in this directory for the full workflow, outputs and troubleshooting.
#
#   scripts/gpu_reference_colab/colab_gpu_reference.sh MODEL [MODEL...]
#
# The HF token (HF_TOKEN or ~/.cache/huggingface/token) is copied to a 0600
# temp file, sent with `colab upload` and deleted; it never appears on a
# command line, in a log, or in code sent with `colab exec`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REMOTE_DIR="/content/gpuref"             # absolute path on the VM
REMOTE_REL="content/gpuref"              # same path for colab upload/download

GPU="H100"
HIGH_MEM=0
TTIS_REF=""
OUT_DIR=""
KEEP=0
SESSION=""
POLL_MINUTES=5
VLLM_VERSION=""
DRY_RUN=0
NEW_ATTEMPTS=3           # colab new attempts per GPU type on 503 (no capacity)
NEW_RETRY_SECONDS=120    # backoff: 120 s, then 240 s
MODELS=()

usage() {
    cat <<'EOF'
Usage: colab_gpu_reference.sh [options] MODEL [MODEL...]

Collect GPU reference eval scores for TTIS models on a Google Colab GPU VM.

Options:
  --gpu GPU[,GPU...]   Colab GPU types in order of preference (default: H100).
                       Each is tried a few times with backoff while Colab
                       answers 503 (no capacity), then the next, e.g. H100,A100
  --high-mem           request a high-RAM machine shape
  --ttis-ref REF       TTIS commit the VM checks out (default: HEAD of this
                       checkout; it must already be pushed to origin)
  --out DIR            local results dir
                       (default: workflow_logs/gpu_reference_colab/<session>)
  --keep               do not `colab stop` the session on exit
  --session NAME       Colab session name (default: gpuref-<ttis sha[:8]>);
                       re-running with the same name attaches to that run
  --poll-minutes N     minutes between status polls (default: 5)
  --vllm-version V     override the vLLM pin in remote_runner.sh
  --dry-run            print every colab command instead of running it
  -h, --help           show this help
EOF
}

log() { printf '[gpuref] %s\n' "$*" >&2; }
die() { printf '[gpuref] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --high-mem) HIGH_MEM=1; shift ;;
        --ttis-ref) TTIS_REF="$2"; shift 2 ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --session) SESSION="$2"; shift 2 ;;
        --poll-minutes) POLL_MINUTES="$2"; shift 2 ;;
        --vllm-version) VLLM_VERSION="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; die "unknown option: $1" ;;
        *) MODELS+=("$1"); shift ;;
    esac
done

[[ ${#MODELS[@]} -gt 0 ]] || { usage >&2; die "at least one MODEL (HF repo id) is required"; }
for m in "${MODELS[@]}"; do
    [[ "$m" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || die "not an HF repo id: $m"
done
IFS=',' read -r -a GPUS <<< "${GPU}"
[[ ${#GPUS[@]} -gt 0 ]] || die "--gpu needs at least one GPU type"
for g in "${GPUS[@]}"; do
    [[ "${g}" =~ ^(T4|L4|G4|H100|A100)$ ]] || die "unsupported --gpu type: ${g} (T4, L4, G4, H100, A100)"
done
[[ "${POLL_MINUTES}" =~ ^[1-9][0-9]*$ ]] || die "--poll-minutes must be a positive integer"
[[ -z "${VLLM_VERSION}" || "${VLLM_VERSION}" =~ ^[0-9A-Za-z.+-]+$ ]] || die "bad --vllm-version"

# ----------------------------------------------------------------------------
# Local helpers
# ----------------------------------------------------------------------------

LOCAL_TMP="$(mktemp -d "${TMPDIR:-/tmp}/gpuref.XXXXXX")"
chmod 700 "${LOCAL_TMP}"
SESSION_ACTIVE=0   # set once this run created or attached to the session

cleanup() {
    local rc=$?
    rm -rf "${LOCAL_TMP}"
    if [[ "${SESSION_ACTIVE}" -eq 1 ]]; then
        if [[ "${KEEP}" -eq 1 ]]; then
            log "--keep: session '${SESSION}' left running. Re-attach with the same command; stop with: colab stop -s ${SESSION}"
        else
            log "stopping session '${SESSION}'"
            colab_cmd stop -s "${SESSION}" || log "WARNING: 'colab stop -s ${SESSION}' failed; check 'colab sessions'"
        fi
    fi
    exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Every colab call goes through here so --dry-run can print it (to stderr,
# so it also shows for calls whose stdout is captured or discarded) instead.
colab_cmd() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        { printf '+ colab'; printf ' %q' "$@"; printf '\n'; } >&2
        return 0
    fi
    colab "$@" < /dev/null
}

# Python with PyYAML for gpuref.py (TTIS's spec loader needs it).
if python3 -c 'import yaml' 2>/dev/null; then
    PYRUN=(python3)
elif command -v uv >/dev/null 2>&1; then
    PYRUN=(uv run --no-project --quiet --with pyyaml python)
else
    die "need python3 with PyYAML, or uv (https://docs.astral.sh/uv/)"
fi
gpuref() { "${PYRUN[@]}" "${SCRIPT_DIR}/gpuref.py" "$@"; }

# Write a snippet for `colab exec -f` into LOCAL_TMP; prints its path.
snippet() {
    local name="$1" path="${LOCAL_TMP}/$1.py"
    cat > "${path}"
    printf '%s\n' "${path}"
}

# Run a snippet on the VM (stdout = its output); in dry-run just print the call.
remote_exec() {
    local name="$1" timeout="$2"
    colab_cmd exec -s "${SESSION}" -f "${LOCAL_TMP}/${name}.py" --timeout "${timeout}"
}

# remote_exec, then require the snippet's success marker in its output:
# `colab exec` exits 0 even when the code raised on the VM.
remote_step() {
    local name="$1" timeout="$2" marker="$3" out
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        remote_exec "${name}" "${timeout}"
        return 0
    fi
    out="$(remote_exec "${name}" "${timeout}" 2>&1)" || true
    printf '%s\n' "${out}" | grep "^GPUREF_" >&2 || true
    printf '%s\n' "${out}" | grep -q "^${marker}" \
        || die "VM step '${name}' failed:"$'\n'"${out}"
}

# Whether `colab sessions` output (stdin) lists SESSION by name.
has_session() {
    awk -v want="[${SESSION}]" '$1 == want { found = 1 } END { exit !found }'
}

# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------

TTIS_REF="${TTIS_REF:-HEAD}"
TTIS_SHA="$(git -C "${REPO_ROOT}" rev-parse --verify "${TTIS_REF}^{commit}" 2>/dev/null)" \
    || die "--ttis-ref ${TTIS_REF} is not a commit in ${REPO_ROOT}"
git -C "${REPO_ROOT}" fetch -q origin 2>/dev/null || log "WARNING: git fetch origin failed; checking pushed state against cached remote refs"
if [[ -z "$(git -C "${REPO_ROOT}" branch -r --contains "${TTIS_SHA}" 2>/dev/null)" ]]; then
    msg="TTIS ${TTIS_SHA} is not on any origin branch; push it first (the VM clones the public repo at this exact sha)"
    if [[ "${DRY_RUN}" -eq 1 ]]; then log "WARNING: ${msg}"; else die "${msg}"; fi
fi
# The preflight below reads this checkout's catalog; the VM reads TTIS_SHA's.
if ! git -C "${REPO_ROOT}" diff --quiet "${TTIS_SHA}" -- workflows/model_specs reference_config/evals; then
    log "WARNING: local specs/eval configs differ from ${TTIS_SHA}; the preflight may not match what the VM runs"
fi
SESSION="${SESSION:-gpuref-${TTIS_SHA:0:8}}"
[[ "${SESSION}" =~ ^[A-Za-z0-9._-]+$ ]] || die "bad --session: ${SESSION}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/workflow_logs/gpu_reference_colab/${SESSION}}"

log "models:  ${MODELS[*]}"
log "ttis:    ${TTIS_SHA} (${TTIS_REF})"
log "session: ${SESSION} (--gpu ${GPUS[*]}$([[ ${HIGH_MEM} -eq 1 ]] && echo ' --high-mem'))"
log "out:     ${OUT_DIR}"

# Models must resolve to a GPU DeviceModelSpec in the dev catalog (the same
# lookup run.py does on the VM), and gated repos need a token that can read them.
CHECK_JSON="${LOCAL_TMP}/check.json"
gpuref check-models "${MODELS[@]}" > "${CHECK_JSON}" || true
FIT_GPUS="${LOCAL_TMP}/fit_gpus"
"${PYRUN[@]}" - "${CHECK_JSON}" "${FIT_GPUS}" "${GPUS[@]}" <<'PY' || die "model preflight failed (see above)"
import json, sys
rows = json.load(open(sys.argv[1]))
fit_path, gpus = sys.argv[2], sys.argv[3:]
usable = set(gpus)
bad = False
for r in rows:
    if not r.get("ok"):
        print(f"[gpuref]   {r['model']}: {r.get('error')}", file=sys.stderr)
        print("[gpuref]   add a `- device: GPU` entry with default_impl: true to its dev spec (docs/gpu_workflows.md)", file=sys.stderr)
        bad = True
        continue
    gated = {True: "gated", False: "public", None: "gating unknown"}[r["gated"]]
    print(f"[gpuref]   {r['model']}: {r['model_id']} max_context={r['max_context']} "
          f"max_concurrency={r['max_concurrency']} revision={r['revision']} ({gated})", file=sys.stderr)
    if r["needs_token"] and not r["have_token"]:
        print(f"[gpuref]   {r['model']} needs an HF token: set HF_TOKEN or run `hf auth login`", file=sys.stderr)
        bad = True
    if r.get("token_can_read") is False:
        print(f"[gpuref]   the HF token cannot read {r['model']}: accept its license at https://huggingface.co/{r['model']}", file=sys.stderr)
        bad = True
    # bf16 weights + KV for one max_context sequence + overhead must fit each
    # GPU we might land on (vLLM refuses to start otherwise).
    mem = r.get("memory")
    if mem is None:
        print(f"[gpuref]     memory estimate unavailable; not checking fit", file=sys.stderr)
        continue
    print(f"[gpuref]     bf16 weights {mem['weights_gib']} GiB + KV {mem['kv_gib_one_seq']} GiB/seq at max_context "
          f"(x{r['max_concurrency']} = {mem['kv_gib_all_seqs']} GiB) -> needs ~{mem['need_gib']} GiB", file=sys.stderr)
    for gpu in gpus:
        if mem["fits"].get(gpu) is False:
            print(f"[gpuref]     does not fit {gpu}; dropping {gpu} from the preference list", file=sys.stderr)
            usable.discard(gpu)
if not bad and not usable:
    print("[gpuref]   no GPU in --gpu fits every model", file=sys.stderr)
    bad = True
open(fit_path, "w").write(" ".join(g for g in gpus if g in usable))
sys.exit(1 if bad else 0)
PY
read -r -a GPUS < "${FIT_GPUS}" || true   # no trailing newline
[[ ${#GPUS[@]} -gt 0 ]] || die "no usable GPU type"

HAVE_TOKEN=0
if [[ -n "${HF_TOKEN:-}" || -s "${HOME}/.cache/huggingface/token" ]]; then
    HAVE_TOKEN=1
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
    command -v colab >/dev/null 2>&1 \
        || die "the Colab CLI is not installed: uv tool install google-colab-cli"
    # Without a cached token any colab call starts the interactive OAuth flow;
    # with stdin at /dev/null it fails instead of hanging.
    [[ -s "${HOME}/.config/colab-cli/token.json" ]] \
        || die "the Colab CLI is not signed in: run 'colab sessions' once in a terminal and complete the sign-in"
    SESSIONS_OUT="$(colab sessions < /dev/null 2>&1)" \
        || die "'colab sessions' failed (${SESSIONS_OUT}); run 'colab sessions' once in a terminal to sign in again"
else
    colab_cmd sessions
    SESSIONS_OUT=""
fi

# ----------------------------------------------------------------------------
# VM-side snippets (no secrets in any of them)
# ----------------------------------------------------------------------------

snippet status > /dev/null <<PY
import os, subprocess
W = "${REMOTE_DIR}"
def read(name):
    try:
        with open(os.path.join(W, name)) as f:
            return f.read().strip()
    except OSError:
        return ""
def alive(pid):
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False
pid = read("runner.pid")
if os.path.exists(os.path.join(W, "DONE")):
    state = "DONE"
elif os.path.exists(os.path.join(W, "FAILED")):
    state = "FAILED"
elif pid and alive(pid):
    state = "RUNNING"
elif pid:
    state = "DIED"
else:
    state = "ABSENT"
print("GPUREF_STATE=" + state)
print("GPUREF_PHASE=" + read("phase"))
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    print("GPUREF_GPU=" + gpu)
except Exception:
    pass
for line in read("runner.log").splitlines()[-12:]:
    print("GPUREF_LOG| " + line)
PY

snippet prepare > /dev/null <<PY
import os
os.makedirs("${REMOTE_DIR}", exist_ok=True)
hf = os.path.expanduser("~/.cache/huggingface")
os.makedirs(hf, mode=0o700, exist_ok=True)
os.chmod(hf, 0o700)
print("GPUREF_PREPARED")
PY

# The upload lands in the (non-hidden) work dir because Jupyter's contents API
# may refuse hidden paths such as ~/.cache; this moves it into place.
snippet install_token > /dev/null <<PY
import os
src = "${REMOTE_DIR}/hf_token.upload"
dst = os.path.expanduser("~/.cache/huggingface/token")
os.chmod(src, 0o600)
os.replace(src, dst)
os.chmod(dst, 0o600)
print("GPUREF_TOKEN_INSTALLED mode=%o bytes=%d" % (os.stat(dst).st_mode & 0o777, os.stat(dst).st_size))
PY

RUNNER_ARGS="['--ttis-sha', '${TTIS_SHA}', '--work-dir', '${REMOTE_DIR}'"
if [[ -n "${VLLM_VERSION}" ]]; then RUNNER_ARGS+=", '--vllm-version', '${VLLM_VERSION}'"; fi
for m in "${MODELS[@]}"; do RUNNER_ARGS+=", '${m}'"; done
RUNNER_ARGS+="]"

snippet launch > /dev/null <<PY
import os, subprocess
W = "${REMOTE_DIR}"
for name in ("DONE", "FAILED", "phase"):
    try:
        os.remove(os.path.join(W, name))
    except OSError:
        pass
log = open(os.path.join(W, "runner.log"), "ab")
# start_new_session=True is setsid(): the runner has no controlling terminal
# and its own process group, so it outlives this kernel call (nohup-like).
proc = subprocess.Popen(["bash", os.path.join(W, "remote_runner.sh")] + ${RUNNER_ARGS},
                        cwd=W, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, close_fds=True)
with open(os.path.join(W, "runner.pid"), "w") as f:
    f.write(str(proc.pid))
print("GPUREF_LAUNCHED pid=%d" % proc.pid)
PY

snippet pack > /dev/null <<PY
import os, subprocess
W = "${REMOTE_DIR}"
tarball = os.path.join(W, "gpuref-results.tar.gz")
subprocess.run(["tar", "-czf", tarball, "-C", W, "results"], check=True)
print("GPUREF_PACKED bytes=%d" % os.path.getsize(tarball))
PY

# ----------------------------------------------------------------------------
# Remote state
# ----------------------------------------------------------------------------

# Sets STATUS_OUT (raw snippet output) and REMOTE_STATE; fails when unreadable.
STATUS_OUT=""
REMOTE_STATE=""
remote_state() {
    STATUS_OUT="$(remote_exec status 120 2>&1)" || return 1
    REMOTE_STATE="$(printf '%s\n' "${STATUS_OUT}" | sed -n 's/^GPUREF_STATE=//p' | tail -1)"
    [[ -n "${REMOTE_STATE}" ]]
}

show_status() {
    printf '%s\n' "${STATUS_OUT}" | sed -n 's/^GPUREF_PHASE=/  phase: /p; s/^GPUREF_GPU=/  gpu:   /p; s/^GPUREF_LOG| /  | /p' >&2
}

upload_token() {
    [[ "${HAVE_TOKEN}" -eq 1 ]] || { log "no local HF token; skipping token upload"; return 0; }
    local tok="${LOCAL_TMP}/hf_token"
    ( umask 077
      if [[ -n "${HF_TOKEN:-}" ]]; then
          printf '%s' "${HF_TOKEN}" > "${tok}"     # printf is a builtin: no argv exposure
      else
          cp "${HOME}/.cache/huggingface/token" "${tok}"
      fi )
    if ! colab_cmd upload -s "${SESSION}" "${tok}" "${REMOTE_REL}/hf_token.upload" > /dev/null; then
        rm -f "${tok}"
        die "HF token upload failed"
    fi
    rm -f "${tok}"
    remote_step install_token 60 GPUREF_TOKEN_INSTALLED
}

launch() {
    log "preparing the VM"
    remote_step prepare 60 GPUREF_PREPARED
    upload_token
    colab_cmd upload -s "${SESSION}" "${SCRIPT_DIR}/remote_runner.sh" "${REMOTE_REL}/remote_runner.sh"
    colab_cmd upload -s "${SESSION}" "${SCRIPT_DIR}/gpuref.py" "${REMOTE_REL}/gpuref.py"
    remote_step launch 60 GPUREF_LAUNCHED
}

collect() {
    log "packing results on the VM"
    remote_step pack 900 GPUREF_PACKED
    mkdir -p "${OUT_DIR}"
    colab_cmd download -s "${SESSION}" "${REMOTE_REL}/gpuref-results.tar.gz" "${OUT_DIR}/gpuref-results.tar.gz"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '+ tar -xzf %q -C %q\n' "${OUT_DIR}/gpuref-results.tar.gz" "${OUT_DIR}" >&2
        printf '+ gpuref.py summary %q\n' "${OUT_DIR}/results" >&2
        return 0
    fi
    tar -xzf "${OUT_DIR}/gpuref-results.tar.gz" -C "${OUT_DIR}"
    log "results in ${OUT_DIR}/results"
    if [[ -f "${OUT_DIR}/results/status.json" ]]; then
        log "per-model status:"; sed 's/^/  /' "${OUT_DIR}/results/status.json" >&2
    fi
    gpuref summary "${OUT_DIR}/results" | tee "${OUT_DIR}/summary.txt" || log "no TTIS eval reports in the results"
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

STATE="ABSENT"
if [[ "${DRY_RUN}" -eq 0 ]] && printf '%s\n' "${SESSIONS_OUT}" | has_session; then
    log "session '${SESSION}' exists; checking for a runner"
    SESSION_ACTIVE=1
    remote_state || die "could not read the runner state on '${SESSION}': ${STATUS_OUT}"
    STATE="${REMOTE_STATE}"
    log "remote state: ${STATE}"
else
    # Walk the preference list. A 503 from the assign endpoint means no
    # capacity right now: retry that type with backoff, then fall through to
    # the next. Any other failure (quota, entitlement) moves on at once.
    ALLOCATED_GPU=""
    for gpu in "${GPUS[@]}"; do
        new_args=(new -s "${SESSION}" --gpu "${gpu}")
        if [[ "${HIGH_MEM}" -eq 1 ]]; then new_args+=(--high-mem); fi
        attempt=1
        delay="${NEW_RETRY_SECONDS}"
        while true; do
            log "creating session '${SESSION}' on ${gpu} (attempt ${attempt}/${NEW_ATTEMPTS})"
            if colab_cmd "${new_args[@]}" > "${LOCAL_TMP}/new.out" 2>&1; then
                cat "${LOCAL_TMP}/new.out" >&2
                ALLOCATED_GPU="${gpu}"
                break
            fi
            tail -3 "${LOCAL_TMP}/new.out" >&2
            # Make sure a half-created assignment is not left behind.
            if colab sessions < /dev/null 2>&1 | has_session; then
                SESSION_ACTIVE=1
                die "'colab new' failed but '${SESSION}' exists; it will be stopped"
            fi
            if grep -q "Service Unavailable" "${LOCAL_TMP}/new.out" && (( attempt < NEW_ATTEMPTS )); then
                log "no ${gpu} capacity (503); retrying in ${delay}s"
                sleep "${delay}"
                attempt=$((attempt + 1))
                delay=$((delay * 2))
                continue
            fi
            log "${gpu} unavailable; trying the next GPU type"
            break
        done
        [[ -n "${ALLOCATED_GPU}" ]] && break
    done
    [[ -n "${ALLOCATED_GPU}" ]] \
        || die "'colab new' failed on every GPU in --gpu (${GPUS[*]}); see ${HOME}/.config/colab-cli/colab.log"
    SESSION_ACTIVE=1
    log "allocated ${ALLOCATED_GPU} (the exact GPU model is recorded in each provenance.json)"
fi

case "${STATE}" in
    ABSENT|DIED)
        if [[ "${STATE}" == "DIED" ]]; then log "the previous runner died without a marker; relaunching"; fi
        launch ;;
    RUNNING) log "attaching to the running runner" ;;
    DONE|FAILED) log "runner already finished (${STATE})" ;;
    *) die "unexpected remote state: ${STATE}" ;;
esac

# Poll. Each poll is a short kernel execution, which also keeps the session's
# kernel active (Colab liveness is driven by kernel activity).
if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "poll every ${POLL_MINUTES} min until DONE/FAILED:"
    remote_exec status 120
    STATE="DONE"
fi
FAILURES=0
while [[ "${STATE}" != "DONE" && "${STATE}" != "FAILED" ]]; do
    sleep $((POLL_MINUTES * 60))
    if remote_state; then
        FAILURES=0
        STATE="${REMOTE_STATE}"
        log "$(date '+%H:%M:%S') state ${STATE}"
        show_status
        if [[ "${STATE}" == "DIED" ]]; then log "runner died without a marker"; STATE="FAILED"; fi
    else
        FAILURES=$((FAILURES + 1))
        log "status poll failed (${FAILURES}/6): $(printf '%s' "${STATUS_OUT}" | tail -2)"
        if (( FAILURES >= 6 )); then
            colab sessions < /dev/null 2>&1 | has_session \
                || { SESSION_ACTIVE=0; die "session '${SESSION}' is gone (reclaimed or stopped); results on the VM are lost"; }
            FAILURES=0
        fi
    fi
done

collect
[[ "${STATE}" == "DONE" ]] || { log "runner finished FAILED; partial results collected"; exit 1; }
log "done"
