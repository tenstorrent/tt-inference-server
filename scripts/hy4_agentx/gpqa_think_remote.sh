#!/bin/bash
# gpqa_think_remote.sh <server ip> <outdir> [concurrency=B of the entry, default 4] [--dry-run]
# GPQA diamond (198 q, task gpqa_diamond_cot_zeroshot, sampling from model.env: Hy4 card temperature 0.9 / top_p 1.0, seed 42) with THINKING ON
# (chat_template_kwargs {"reasoning_effort": "high"} per request), run from THIS host (the one with the evals venv) against http://<ip>:8000.
# Required env: EVALS_VENV (venv dir with lm_eval), HF_HOME_DIR (offline HF cache holding the GPQA dataset). Optional: TOKENIZER_DIR (default HARNESS_CKPT_DIR),
#   MODEL_ID (served model id, default HARNESS_MODEL_ID), PROBE=<n> (first n questions only), MAXGEN (default 64500), MAXLEN (default 67584), SEED, THINK_MODE=on|off.
# Per-request log: <outdir>/gpqa/think_requests.jsonl (finish=length => truncated; report the rate). ~2 h per run at concurrency = B (DSV4 numbers).
# The output directory must NOT be on the cache NVMe (a board reset detaches it).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set -a; . "${HARNESS_MODEL_ENV:-$HERE/model.env}"; set +a
SRV=${1:-}; R=${2:-}; C=${3:-4}; DRY=0; for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done
[ "$C" = "--dry-run" ] && C=4
[ -n "$SRV" ] && [ -n "$R" ] || { sed -n 2,9p "$0"; exit 2; }
MODEL_ID=${MODEL_ID:-$HARNESS_MODEL_ID}; TOKENIZER_DIR=${TOKENIZER_DIR:-$HARNESS_CKPT_DIR}; SEED=${SEED:-$HARNESS_SEED}
O=$R/gpqa
ARGS=(--tasks gpqa_diamond_cot_zeroshot --model local-chat-completions
  --model_args "model=$MODEL_ID,base_url=http://$SRV:8000/v1/chat/completions,tokenizer_backend=huggingface,tokenizer=$TOKENIZER_DIR,max_length=${MAXLEN:-67584},timeout=14400,num_concurrent=$C"
  --gen_kwargs "max_gen_toks=${MAXGEN:-64500},until=[],do_sample=true,temperature=$HARNESS_TEMPERATURE,top_p=$HARNESS_TOP_P,stream=true,seed=$SEED"
  --output_path "$O/gpqa_out" --seed "$SEED" --num_fewshot 0 --batch_size 1 --log_samples --show_config
  --apply_chat_template --trust_remote_code --confirm_run_unsafe_code)
[ -n "${PROBE:-}" ] && ARGS+=(--limit "$PROBE")
if [ $DRY = 1 ]; then
  echo "[dry-run] THINK_MODE=${THINK_MODE:-on} (per-request chat_template_kwargs $(THINK_MODE=${THINK_MODE:-on} python3 "$HERE/harness_model.py" think $([ "${THINK_MODE:-on}" = off ] && echo off || echo on)))"
  echo "[dry-run] THINK_LOG=$O/think_requests.jsonl HF_HOME=${HF_HOME_DIR:-<HF_HOME_DIR unset>} HF_HUB_OFFLINE=1"
  echo "[dry-run] \${EVALS_VENV:-<unset>}/bin/python $HERE/lm_eval_think.py ${ARGS[*]}"
  exit 0
fi
: "${EVALS_VENV:?}" "${HF_HOME_DIR:?}"
[ -x "$EVALS_VENV/bin/python" ] || { echo "evals venv python missing on $(hostname)" >&2; exit 127; }
mkdir -p "$O/gpqa_out"; cd /tmp; export HF_TOKEN=local-unused HF_HOME=$HF_HOME_DIR HF_HUB_OFFLINE=1 THINK_LOG=$O/think_requests.jsonl
exec "$EVALS_VENV/bin/python" "$HERE/lm_eval_think.py" "${ARGS[@]}"
