#!/bin/bash
# AgentX (InferenceX agentic trace replay, aiperf) against a running DeepSeek-V4.1-Flash server, from inside the release image.
#
#   run_agentx.sh <8|16> [--duration SECONDS=3600] [--server-url URL=http://localhost:8000] [--mode full|ci=full]
#                        [--concurrency N=batch size] [--cache-salt STR]
#
# Concurrency (= trace sessions) defaults to the batch size. Output goes to $OUT_DIR (default /agentx_out; bind-mount a host dir there).
# $HF_HOME (default /agentx_hf; bind-mount a host dir to keep the trace datasets between runs) holds the HF cache.
# Network: only the first run, to download the public trace dataset from huggingface.co (apache-2.0, no token needed) into $HF_HOME.
# Everything else (tokenizer, InferenceX + aiperf 0.12.0 venv, aggregation) is baked into the image.
set -u
B=${1:-}; [ "$B" = 8 ] || [ "$B" = 16 ] || { sed -n 2,9p "$0"; echo "first argument must be 8 or 16"; exit 2; }; shift
DUR=3600; URL=http://localhost:8000; MODE=full; CONC=""; SALT=""
while [ $# -gt 0 ]; do case $1 in
  --duration) DUR=$2; shift 2;; --server-url) URL=$2; shift 2;; --mode) MODE=$2; shift 2;;
  --concurrency) CONC=$2; shift 2;; --cache-salt) SALT=$2; shift 2;;
  *) echo "unknown option $1"; exit 2;; esac; done
TT=${TT_INFERENCE_SERVER_DIR:-/home/container_app_user/tt-inference-server}
IMPL=deepseek-v41-flash-b${B}-1m; IMPL_KEY=deepseek_v41_flash_b${B}_1m
OUT=${OUT_DIR:-/agentx_out}; mkdir -p "$OUT" 2>/dev/null; [ -w "$OUT" ] || { echo "ERROR: $OUT is not writable: bind-mount a host directory there (-v HOSTDIR:/agentx_out)"; exit 2; }
HFH=${HF_HOME:-/agentx_hf}; mkdir -p "$HFH" 2>/dev/null; [ -w "$HFH" ] || { HFH=/tmp/agentx_hf; mkdir -p $HFH; echo "WARNING: HF_HOME not writable, using $HFH (dataset is re-downloaded on every container)"; }
export HF_HOME=$HFH HOME=${HOME:-/tmp}; [ -w "$HOME" ] || export HOME=/tmp
export CACHE_ROOT=$OUT                      # run.py / run_workflows.py write workflow_logs/ under it
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'   # container uid != owner of the baked tree
export AGENTX_CONCURRENCY=${CONC:-${AGENTX_CONCURRENCY:-$B}}
[ -n "$SALT" ] && export AGENTX_CACHE_SALT=$SALT
VPY=$TT/.workflow_venvs/.venv_agentic_traces/bin/python
# 1. tokenizer: seed the HF cache from the baked copy (the "local" snapshot the server itself uses)
TOKSRC=$TT/scripts/dsv41_agentx/hf_tokenizer/models--deepseek-ai--DeepSeek-V4.1-Flash
TOKDST=$HF_HOME/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
[ -e "$TOKDST/snapshots/local/tokenizer.json" ] || { mkdir -p "$HF_HOME/hub" && cp -a "$TOKSRC" "$HF_HOME/hub/"; }
# 2. dataset: download once (network) and build the arrow cache, then everything runs offline
DS=semianalysisai/cc-traces-weka-062126
if [ ! -d "$HF_HOME/datasets/semianalysisai___cc-traces-weka-062126/default" ]; then
  echo "[run_agentx] downloading dataset $DS into $HF_HOME (needs huggingface.co; about 2 GB, a few minutes) ..."
  HF_DATASETS_DISABLE_PROGRESS_BARS=1 "$VPY" -c "from datasets import load_dataset; d = load_dataset('$DS', split='train'); print('dataset rows:', len(d))" 2>&1 | tail -3
  [ -d "$HF_HOME/datasets/semianalysisai___cc-traces-weka-062126/default" ] || { echo "dataset download failed (network?)"; exit 3; }
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# 3. runtime model spec: the server's own entry from the baked catalog (the image's tt-inference-server tree does not carry it in its yaml catalog)
export MODEL_SPECS_ENV=dev
SPEC=$OUT/runtime_entry_b${B}.json
"$VPY" - "$IMPL_KEY" "$SPEC" <<'PY'
import json, os, sys
d = json.load(open(os.environ.get("MODEL_SPECS_JSON_PATH", "/home/container_app_user/model_specs/model_spec.json")))
v = d["model_specs"]["deepseek-ai/DeepSeek-V4.1-Flash"]["BLACKHOLE_GALAXY"]["vLLM"][sys.argv[1]]
json.dump(v, open(sys.argv[2], "w"))
print("catalog entry:", sys.argv[1], v["model_id"], "max_concurrency", v["device_model_spec"]["max_concurrency"], "max_context", v["device_model_spec"]["max_context"])
PY
START=$(date +%s)
cd "$TT"
"$VPY" run.py --model deepseek-ai/DeepSeek-V4.1-Flash --tt-device blackhole_galaxy \
  --workflow agentic_traces --server-url "$URL" --no-auth --skip-system-sw-validation --runtime-model-spec-json "$SPEC" \
  --agentic-traces-mode "$MODE" --agentic-traces-duration "$DUR" 2>&1 | tee "$OUT/run_agentx.log"
RC=${PIPESTATUS[0]}
echo "[run_agentx] run.py rc=$RC  output under $OUT/workflow_logs"
# 4. metrics summary
for f in $(find "$OUT" -name profile_export.jsonl -newermt "@$START" -printf '%h\n' | sort -u); do
  "$VPY" "$TT/scripts/dsv41_agentx/agentx_metrics.py" "$f" 2>&1 | tee -a "$OUT/agentx_summary.txt"
done
[ -s "$OUT/agentx_summary.txt" ] || { echo "[run_agentx] no profile_export.jsonl produced: see $OUT/run_agentx.log"; [ $RC = 0 ] && RC=4; }
exit $RC
