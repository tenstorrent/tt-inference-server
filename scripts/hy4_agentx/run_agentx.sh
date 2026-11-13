#!/bin/bash
# AgentX (InferenceX agentic trace replay, aiperf) against a running server of the model in model.env (Hy4-Preview), from inside the image
# that carries the client venv. Generalisation of scripts/dsv41_agentx/run_agentx.sh (which stays untouched and keeps its DeepSeek-V4.1-Flash constants).
#
#   run_agentx.sh <B> [--duration SECONDS=3600] [--server-url URL=http://localhost:8000] [--mode full|ci=full]
#                     [--concurrency N=B] [--cache-salt STR] [--dry-run]
#
# B must be one of HARNESS_BATCH_SIZES of model.env (Hy4: 4 primary, 8 secondary). Sessions (concurrency) default to B.
# --dry-run: resolve everything, print the plan (variables, catalog entry, run.py command) and exit 0 without a venv, network or output directory.
# Model constants: model.env next to this script (env HARNESS_MODEL_ENV=<file> selects another model).
# Output goes to $OUT_DIR (default /agentx_out; bind-mount a host dir), HF cache in $HF_HOME (default /agentx_hf; bind-mount to keep the dataset).
# Catalog: $MODEL_SPECS_JSON_PATH (default HARNESS_CATALOG of model.env) must contain the server's entry of this model.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MENV=${HARNESS_MODEL_ENV:-$HERE/model.env}
[ -f "$MENV" ] || { echo "ERROR: model env $MENV not found"; exit 2; }
set -a; . "$MENV"; set +a
usage() { sed -n 2,13p "$0"; }
BS=${HARNESS_BATCH_SIZES//,/ }
B=${1:-}; case " $BS " in *" $B "*) ;; *) usage; echo "first argument must be one of: $BS"; exit 2;; esac; shift
DUR=3600; URL=http://localhost:8000; MODE=full; CONC=""; SALT=""; DRY=0
while [ $# -gt 0 ]; do case $1 in
  --duration) DUR=$2; shift 2;; --server-url) URL=$2; shift 2;; --mode) MODE=$2; shift 2;;
  --concurrency) CONC=$2; shift 2;; --cache-salt) SALT=$2; shift 2;; --dry-run) DRY=1; shift;;
  *) echo "unknown option $1"; exit 2;; esac; done
case $MODE in full|ci) ;; *) echo "--mode must be full or ci"; exit 2;; esac
TT=${TT_INFERENCE_SERVER_DIR:-/home/container_app_user/tt-inference-server}
[ -d "$TT" ] || TT=$(cd "$HERE/../.." && pwd)            # source checkout: the script sits in <tree>/scripts/hy4_agentx
IMPL_ID=$(python3 "$HERE/harness_model.py" key "$B") || exit 2       # catalog key, e.g. hy4_preview_b4
IMPL_NAME=$(python3 "$HERE/harness_model.py" impl "$B")             # e.g. hy4-preview-b4
CATALOG=${MODEL_SPECS_JSON_PATH:-$HARNESS_CATALOG}
OUT=${OUT_DIR:-/agentx_out}
HFH=${HF_HOME:-/agentx_hf}
export AGENTX_CONCURRENCY=${CONC:-${AGENTX_CONCURRENCY:-$B}}
[ -n "$SALT" ] && export AGENTX_CACHE_SALT=$SALT
VPY=$TT/.workflow_venvs/.venv_agentic_traces/bin/python
TOKNAME=$(python3 "$HERE/harness_model.py" tokdir)
TOKSRC=$HERE/hf_tokenizer/$TOKNAME
TOKDST=$HFH/hub/$TOKNAME
SPEC=$OUT/runtime_entry_b${B}.json
RUNCMD=("$VPY" run.py --model "$HARNESS_MODEL_ID" --tt-device "$HARNESS_DEVICE" --workflow agentic_traces --server-url "$URL" --no-auth --skip-system-sw-validation
        --runtime-model-spec-json "$SPEC" --agentic-traces-mode "$MODE" --agentic-traces-duration "$DUR")
if [ $DRY = 1 ]; then
  echo "[dry-run] model=$HARNESS_MODEL_ID name=$HARNESS_MODEL_NAME B=$B impl_id=$IMPL_ID impl_name=$IMPL_NAME catalog_model_id=$(python3 "$HERE/harness_model.py" model-id "$B")"
  echo "[dry-run] sessions(AGENTX_CONCURRENCY)=$AGENTX_CONCURRENCY duration=$DUR mode=$MODE server=$URL cache_salt=${SALT:-none} dataset=$HARNESS_TRACE_DATASET"
  echo "[dry-run] catalog=$CATALOG out=$OUT hf_home=$HFH tokenizer_src=$TOKSRC tokenizer_dst=$TOKDST"
  echo "[dry-run] cd $TT && ${RUNCMD[*]}"
  echo "[dry-run] metrics: $VPY $TT/scripts/dsv41_agentx/agentx_metrics.py <dir containing profile_export.jsonl>   (model agnostic)"
  exit 0
fi
mkdir -p "$OUT" 2>/dev/null; [ -w "$OUT" ] || { echo "ERROR: $OUT is not writable: bind-mount a host directory there (-v HOSTDIR:/agentx_out)"; exit 2; }
mkdir -p "$HFH" 2>/dev/null; [ -w "$HFH" ] || { HFH=/tmp/agentx_hf; mkdir -p $HFH; TOKDST=$HFH/hub/$TOKNAME; echo "WARNING: HF_HOME not writable, using $HFH (dataset is re-downloaded on every container)"; }
export HF_HOME=$HFH HOME=${HOME:-/tmp}; [ -w "$HOME" ] || export HOME=/tmp
export CACHE_ROOT=$OUT                      # run.py / run_workflows.py write workflow_logs/ under it
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'   # container uid != owner of the baked tree
# 1. tokenizer: seed the HF cache from the baked copy (checkpoint tokenizer.json / chat template; no network)
[ -d "$TOKSRC" ] || { echo "ERROR: baked tokenizer dir $TOKSRC missing (the .gitignore '*.json' trap: git add -f)"; exit 2; }
[ -e "$TOKDST/snapshots/local/tokenizer.json" ] || { mkdir -p "$HF_HOME/hub" && cp -a "$TOKSRC" "$HF_HOME/hub/"; }
# 1b. preflight: the client venv's transformers must load this tokenizer offline (hy_v4 is a new model type: UNVERIFIED for the venv's transformers)
HF_HUB_OFFLINE=1 "$VPY" - "$TOKDST/snapshots/local" <<'PY' || { echo "ERROR: the AGENTIC_TRACES venv cannot load the baked tokenizer (transformers too old for this model?)"; exit 5; }
import sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(sys.argv[1])
ids = t("Hello, how are you?")["input_ids"]
print("tokenizer OK:", type(t).__name__, "vocab", len(t), "sample ids", len(ids))
PY
# 2. dataset: download once (network) and build the arrow cache, then everything runs offline
DS=$HARNESS_TRACE_DATASET; DSDIR=$HF_HOME/datasets/$(echo "$DS" | sed 's#/#___#')/default
if [ ! -d "$DSDIR" ]; then
  echo "[run_agentx] downloading dataset $DS into $HF_HOME (needs huggingface.co; about 2 GB, a few minutes) ..."
  HF_DATASETS_DISABLE_PROGRESS_BARS=1 "$VPY" -c "from datasets import load_dataset; d = load_dataset('$DS', split='train'); print('dataset rows:', len(d))" 2>&1 | tail -3
  [ -d "$DSDIR" ] || { echo "dataset download failed (network?)"; exit 3; }
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# 3. runtime model spec: the server's own entry from the catalog
export MODEL_SPECS_ENV=dev
python3 "$HERE/mk_runtime_spec.py" "$B" agentic_traces "$OUT" "$CATALOG" --entry-only | head -1 || exit 2
cp "$OUT/entry_${IMPL_ID}.json" "$SPEC"
START=$(date +%s)
cd "$TT"
"${RUNCMD[@]}" 2>&1 | tee "$OUT/run_agentx.log"
RC=${PIPESTATUS[0]}
echo "[run_agentx] run.py rc=$RC  output under $OUT/workflow_logs"
# 4. metrics summary
for f in $(find "$OUT" -name profile_export.jsonl -newermt "@$START" -printf '%h\n' | sort -u); do
  "$VPY" "$TT/scripts/dsv41_agentx/agentx_metrics.py" "$f" 2>&1 | tee -a "$OUT/agentx_summary.txt"
done
[ -s "$OUT/agentx_summary.txt" ] || { echo "[run_agentx] no profile_export.jsonl produced: see $OUT/run_agentx.log"; [ $RC = 0 ] && RC=4; }
exit $RC
