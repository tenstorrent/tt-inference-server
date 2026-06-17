#!/usr/bin/env bash
# Run an lm-eval task against an ALREADY-SERVED Qwen3-8B vLLM server.
#
# Usage:
#   ./run_eval_qwen3_8b.sh [--task r1_gpqa_diamond|mmlu_pro] [--samples N] [--port P] [options]
#
# Options (all accept --flag value or --flag=value):
#   --task          r1_gpqa_diamond (default) | mmlu_pro
#   --samples       --limit value. INTEGER N = first N docs (default 5);
#                   FLOAT 0<x<=1 = fraction (e.g. 0.2); 0 or "all" = full set.
#   --port          vLLM server port (default 8004)
#   --host          server host (default 127.0.0.1)
#   --concurrency   num_concurrent requests (default 32)
#   --max-gen-toks  max generated tokens (default 12288)
#   --fewshot       num_fewshot (default: gpqa=0, mmlu_pro=5)
#   --timeout       per-request streaming timeout, seconds (default 3600)
#   --api-key       server API key (default: $OPENAI_API_KEY / $API_KEY / "your-secret-key")
#   -h, --help      show this help
#
# Needs HF_TOKEN for r1_gpqa_diamond (gated GPQA dataset); mmlu_pro is open.
set -euo pipefail

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"; exit "${1:-0}"; }

# Defaults (FEWSHOT left empty -> auto per task below)
TASK="r1_gpqa_diamond"; SAMPLES="5"; PORT="8004"; HOST="127.0.0.1"
CONCURRENCY="32"; MAX_GEN_TOKS="12288"; TIMEOUT="3600"; FEWSHOT=""
APIKEY="${OPENAI_API_KEY:-${API_KEY:-your-secret-key}}"

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --task|--samples|--port|--host|--concurrency|--max-gen-toks|--fewshot|--timeout|--api-key)
      key="$1"; val="${2:-}"; shift 2 || { echo "ERROR: $key needs a value"; exit 1; } ;;
    --task=*|--samples=*|--port=*|--host=*|--concurrency=*|--max-gen-toks=*|--fewshot=*|--timeout=*|--api-key=*)
      key="${1%%=*}"; val="${1#*=}"; shift ;;
    *) echo "ERROR: unknown arg '$1'"; usage 1 ;;
  esac
  case "$key" in
    --task) TASK="$val" ;;
    --samples) SAMPLES="$val" ;;
    --port) PORT="$val" ;;
    --host) HOST="$val" ;;
    --concurrency) CONCURRENCY="$val" ;;
    --max-gen-toks) MAX_GEN_TOKS="$val" ;;
    --fewshot) FEWSHOT="$val" ;;
    --timeout) TIMEOUT="$val" ;;
    --api-key) APIKEY="$val" ;;
  esac
done

case "$TASK" in
  r1_gpqa_diamond) [ -n "$FEWSHOT" ] || FEWSHOT=0 ;;
  mmlu_pro)        [ -n "$FEWSHOT" ] || FEWSHOT=5 ;;
  *) echo "ERROR: --task must be 'r1_gpqa_diamond' or 'mmlu_pro' (got '$TASK')"; exit 1 ;;
esac

# --limit: pass floats/ints through; "0"/"all" -> no limit.
LIMIT_ARG=()
if [ "$SAMPLES" != "0" ] && [ "$SAMPLES" != "all" ]; then
  LIMIT_ARG=(--limit "$SAMPLES")
fi

ROOT=/home/kmabee/tt-inference-server
LM_EVAL="$ROOT/.workflow_venvs/.venv_evals_common/bin/lm_eval"
[ -x "$LM_EVAL" ] || { echo "ERROR: lm_eval not found at $LM_EVAL"; exit 1; }

BASE_URL="http://${HOST}:${PORT}/v1/completions"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/eval_${TASK}_qwen3-8b_${TS}"
mkdir -p "$OUT"
LOG="$OUT/eval.log"

# Preflight: confirm the server is up.
if ! curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1 \
   && ! curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: no healthy server at http://${HOST}:${PORT} — start the Qwen3-8B server first."; exit 1
fi
[ -n "${HF_TOKEN:-}" ] || echo "WARN: HF_TOKEN not set — r1_gpqa_diamond needs HF access to the (gated) GPQA dataset (cached copy may suffice)."
# lm-eval local-completions sends Authorization: Bearer $OPENAI_API_KEY.
if [ -n "$APIKEY" ]; then
  export OPENAI_API_KEY="$APIKEY"
else
  echo "WARN: no API key (--api-key / OPENAI_API_KEY / API_KEY) — server will 401 if auth is enabled."
fi

echo "[$(date)] task=$TASK samples=$SAMPLES fewshot=$FEWSHOT concurrency=$CONCURRENCY max_gen_toks=$MAX_GEN_TOKS"
echo "[$(date)] server=$BASE_URL  out=$OUT"

"$LM_EVAL" \
  --tasks "$TASK" \
  --model local-completions \
  --model_args "model=Qwen/Qwen3-8B,base_url=${BASE_URL},tokenizer_backend=huggingface,max_length=65536,timeout=${TIMEOUT},num_concurrent=${CONCURRENCY},max_retries=1" \
  --gen_kwargs "stream=true,max_gen_toks=${MAX_GEN_TOKS},until=[],do_sample=true,temperature=0.6,top_k=20,top_p=0.95" \
  --output_path "$OUT" \
  --seed 42 \
  --num_fewshot "$FEWSHOT" \
  --batch_size 1 \
  --log_samples --show_config --apply_chat_template --trust_remote_code --confirm_run_unsafe_code \
  "${LIMIT_ARG[@]}" \
  |& tee "$LOG"

echo "=== DONE: $TASK | results + samples in $OUT ==="
