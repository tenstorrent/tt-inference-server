#!/bin/bash
# gsm8k_remote.sh <server ip> <outdir> [concurrency=4] [--dry-run]
# GSM8K 8-shot CoT (1,319 problems, lm_eval task hy4_gsm8k_cot = dsv41_gsm8k_cot + a boxed-aware filter), chat API. GATE metric = boxed-aware (last \\boxed{}, else answer-is/####, else last number
# outside math delimiters); flexible-extract and strict-match are reported alongside but mis-read LaTeX answers (closing $$ taken as the answer: 0.66 vs 0.96). Run with thinking OFF per request (reasoning_effort=no_think: a 2048-token cap cannot hold
# high-effort reasoning; DSV4 ran it non-thinking too). Same env as gpqa_think_remote.sh; INCLUDE_PATH = reference_config/evals/lm_eval_tasks of the tt-inference-server tree.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set -a; . "${HARNESS_MODEL_ENV:-$HERE/model.env}"; set +a
SRV=${1:-}; R=${2:-}; C=${3:-4}; DRY=0; for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done
[ "$C" = "--dry-run" ] && C=4
[ -n "$SRV" ] && [ -n "$R" ] || { sed -n 2,7p "$0"; exit 2; }
MODEL_ID=${MODEL_ID:-$HARNESS_MODEL_ID}; TOKENIZER_DIR=${TOKENIZER_DIR:-$HARNESS_CKPT_DIR}; SEED=${SEED:-$HARNESS_SEED}
INCLUDE_PATH=${INCLUDE_PATH:-$(cd "$HERE/../.." && pwd)/reference_config/evals/lm_eval_tasks}
O=$R/gsm8k
ARGS=(--tasks hy4_gsm8k_cot --model local-chat-completions
  --model_args "model=$MODEL_ID,base_url=http://$SRV:8000/v1/chat/completions,tokenizer_backend=huggingface,tokenizer=$TOKENIZER_DIR,max_length=${MAXLEN:-40000},timeout=14400,num_concurrent=$C"
  --gen_kwargs "max_gen_toks=${MAXGEN:-2048},until=[\"Q:\"],do_sample=true,temperature=$HARNESS_TEMPERATURE,top_p=$HARNESS_TOP_P,stream=true,seed=$SEED"
  --output_path "$O/lm_out" --seed "$SEED" --num_fewshot 8 --batch_size 1 --log_samples --show_config
  --include_path "$INCLUDE_PATH" --apply_chat_template --trust_remote_code --confirm_run_unsafe_code)
if [ $DRY = 1 ]; then
  echo "[dry-run] THINK_MODE=off (per-request chat_template_kwargs $(python3 "$HERE/harness_model.py" think off))"
  echo "[dry-run] \${EVALS_VENV:-<unset>}/bin/python $HERE/lm_eval_think.py ${ARGS[*]}"
  exit 0
fi
: "${EVALS_VENV:?}" "${HF_HOME_DIR:?}"
[ -x "$EVALS_VENV/bin/python" ] || { echo "evals venv python missing on $(hostname)" >&2; exit 127; }
mkdir -p "$O/lm_out"; cd /tmp; export CACHE_ROOT=$O/cache_root_eval HF_TOKEN=local-unused HF_HOME=$HF_HOME_DIR HF_HUB_OFFLINE=1 THINK_MODE=off THINK_LOG=$O/think_requests.jsonl
exec "$EVALS_VENV/bin/python" "$HERE/lm_eval_think.py" "${ARGS[@]}"
