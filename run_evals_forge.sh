#!/bin/bash
# Run a forge LLM's evals against an ALREADY-RUNNING server, via run.py.
# Generic over model + port (defaults reproduce the Llama-3.1-8B-Instruct case).
#
# Start a server first, from the tt-xla venv, e.g.:
#   cd <tt-xla checkout> && source venv/activate
#   cd <tt-inference-server checkout>/tt-media-server
#   DEVICE_IDS=0 PORT=8012 ./launch_llama_8b.sh
#
# Drives `run.py --workflow evals`, which reads the model's EvalConfig in
# evals/eval_config.py and handles BOTH eval code paths:
#   - meta_* (e.g. meta_ifeval)  -> WorkflowVenvType.EVALS_META (builds work_dir)
#   - longbench_* / mmlu_pro / gpqa -> WorkflowVenvType.EVALS_COMMON (lm-eval-harness)
# A plain `lm_eval` one-liner cannot reproduce the meta_* tasks, which is why
# this wraps run.py rather than calling lm_eval directly.
#
# Usage:
#   ./run_evals_forge.sh                                   # default model, ALL tasks, ci-nightly
#   ./run_evals_forge.sh --model Qwen3-8B --port 8019      # different model + port
#   ./run_evals_forge.sh --mode smoke-test                 # ALL tasks, ~1% (fast smoke)
#   ./run_evals_forge.sh --task longbench_code_e           # one task, first 20 docs
#   ./run_evals_forge.sh --model Qwen3-4B --port 8010 --task mmlu_pro --samples 50
#   ./run_evals_forge.sh --task meta_gpqa --samples all    # one task, full doc set
#
# Options (--flag value or --flag=value):
#   --model NAME   model name as in evals/eval_config.py / model spec
#                  (default Llama-3.1-8B-Instruct).
#   --port P       server port (default 8012).
#   --task NAME    run a single eval task (uses --eval-samples doc-id filter).
#                  Default: unset -> run ALL tasks for the model.
#   --samples N    single-task mode only: run docs 0..N-1 (default 20).
#                  ("all" runs the full task set for that task.)
#   --mode MODE    all-tasks mode only: ci-nightly | smoke-test | ci-long | ci-commit
#                  (default ci-nightly, matching CI). Ignored when --task is set
#                  (--eval-samples and --limit-samples-mode are mutually exclusive).
#   --device DEV   tt device (default p150).
#   --server-url U target a non-localhost server, e.g. http://10.0.0.5 (default 127.0.0.1)
#   -h, --help     show this help
#
# meta_*/gpqa need HF access (HF_TOKEN); longbench_*/mmlu_pro are open. Default
# API key is "your-secret-key".
set -e

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"; exit "${1:-0}"; }

MODEL="Llama-3.1-8B-Instruct"; PORT="8012"; DEVICE="p150"
TASK=""; SAMPLES="20"; MODE="ci-nightly"; SERVER_URL=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --model|--port|--device|--task|--samples|--mode|--server-url)
      key="$1"; val="${2:-}"; shift 2 || { echo "ERROR: $key needs a value"; exit 1; } ;;
    --model=*|--port=*|--device=*|--task=*|--samples=*|--mode=*|--server-url=*)
      key="${1%%=*}"; val="${1#*=}"; shift ;;
    *) echo "ERROR: unknown arg '$1'"; usage 1 ;;
  esac
  case "$key" in
    --model) MODEL="$val" ;;
    --port) PORT="$val" ;;
    --device) DEVICE="$val" ;;
    --task) TASK="$val" ;;
    --samples) SAMPLES="$val" ;;
    --mode) MODE="$val" ;;
    --server-url) SERVER_URL="$val" ;;
  esac
done

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Preflight: confirm the server is up.
HOST_FOR_CHECK="${SERVER_URL:-http://127.0.0.1}"
if ! curl -sf "${HOST_FOR_CHECK}:${PORT}/health" >/dev/null 2>&1 \
   && ! curl -sf "${HOST_FOR_CHECK}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: no healthy server at ${HOST_FOR_CHECK}:${PORT} — start one first (a launch_*.sh on this port, from the tt-xla venv)."; exit 1
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY:-your-secret-key}}"
[ -n "${HF_TOKEN:-}" ] || echo "WARN: HF_TOKEN not set — meta_*/gpqa need HF access (cached datasets may suffice)."

# --dev-mode: forge LLM specs live in the dev catalog (CI runs run.py --dev-mode);
# without it run.py looks in 'prod' and errors "does not support ... forge-vllm-plugin".
ARGS=(--model "$MODEL" --tt-device "$DEVICE" --engine forge
      --impl forge-vllm-plugin --workflow evals --service-port "$PORT"
      --dev-mode --skip-system-sw-validation)
[ -n "$SERVER_URL" ] && ARGS+=(--server-url "$SERVER_URL")

if [ -n "$TASK" ]; then
  if [ "$SAMPLES" = "all" ]; then
    # "all" = run the full, unrestricted doc set. A null value for the task's
    # --eval-samples entry still selects the task (task_configs.py's task
    # selection only checks dict keys) but applies no --samples limit at all,
    # so lm-eval runs its natural full set for every (sub)task regardless of
    # size. A flat index-range cap (e.g. range(4000)) is the wrong way to do
    # this: lm-eval hard-rejects it for any (sub)task with fewer than that many
    # examples ("Elements of --samples should be in the interval [0,k-1]...")
    # and silently under-covers anything bigger (e.g. full mmlu_pro ~12k).
    echo "[evals] --samples all -> no --samples limit (full natural task size)"
    EVAL_SAMPLES=$(python3 -c "import json,sys; print(json.dumps({sys.argv[1]: None}))" "$TASK")
  else
    EVAL_SAMPLES=$(python3 -c "import json,sys; print(json.dumps({'$TASK': list(range(int(sys.argv[1])))}))" "$SAMPLES")
  fi
  echo "[evals] model=$MODEL single task=$TASK samples=$SAMPLES port=$PORT"
  ARGS+=(--eval-samples "$EVAL_SAMPLES")
else
  echo "[evals] model=$MODEL ALL tasks  mode=$MODE port=$PORT"
  ARGS+=(--limit-samples-mode "$MODE")
fi

# Note: not echoing ${ARGS[*]} — in single-task mode it contains the (large)
# --eval-samples index list, which spams the log.
exec python3 run.py "${ARGS[@]}"
