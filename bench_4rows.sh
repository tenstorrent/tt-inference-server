#!/usr/bin/env bash
# Re-run the 4 "Stage-2 sweep" benchmark rows against a running forge LLM server.
# Matches the original methodology (vllm bench serve, openai-chat, random dataset)
# so numbers are comparable to the recorded table.
#
#   row  conc  ISL  OSL    n     baseline agg / TTFT (gmu0.30)
#    1     1   128  128    8     13.28 / 356
#    2    32   128  128   256    77.47 / 3025
#    3     1   128 1024    4     13.30 / 357
#    4    32   128 1024   128    180.68 / 539
#
# Usage:
#   ./bench_4rows.sh                      # defaults: Qwen3-8B @ :8004
#   PORT=8004 MODEL=Qwen/Qwen3-8B ./bench_4rows.sh
#   WARMUP=1 ./bench_4rows.sh             # warm each shape's trace first (cleaner TTFT)
#
# NOTE: run it TWICE and use the 2nd run if you don't use WARMUP=1 — the first
# request of each shape can hit a cold trace-capture that inflates mean TTFT.
# Caveat: the random dataset hits early-EOS, so osl=1024 rows generate ~200 tok,
# not 1024 (matches how the original table was measured).

set -u
cd "$(dirname "$0")"

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8004}
MODEL=${MODEL:-Qwen/Qwen3-8B}
API_KEY=${API_KEY:-your-secret-key}
WARMUP=${WARMUP:-0}
VLLM=${VLLM:-$(command -v vllm)}
export OPENAI_API_KEY="$API_KEY"

[ -n "$VLLM" ] && command -v "$VLLM" >/dev/null 2>&1 || { echo "ERROR: vllm not found on \$PATH (activate the benchmark venv, or set VLLM=/path/to/vllm)"; exit 1; }

# readiness check (real completion)
echo ">>> checking server $HOST:$PORT ..."
curl -s -m 90 "http://$HOST:$PORT/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" \
  2>/dev/null | grep -q '"choices"' || { echo "ERROR: server not ready (model not loaded?) on $HOST:$PORT"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT="bench_4rows_${TS}"; mkdir -p "$OUT"
echo ">>> ready. results -> $OUT/"

# rows: "conc num_prompts isl osl"
ROWS=( "1 8 128 128" "32 256 128 128" "1 4 128 1024" "32 128 128 1024" )

run_one() {  # conc n isl osl resultfile
  local C=$1 N=$2 ISL=$3 OSL=$4 RF=$5
  "$VLLM" bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --model "$MODEL" --host "$HOST" --port "$PORT" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --max-concurrency "$C" --num-prompts "$N" \
    --percentile-metrics ttft,tpot,itl,e2el --save-result --result-filename "$RF" \
    --extra-body "{\"truncate_prompt_tokens\": \"$ISL\", \"max_tokens\": $OSL}"
}

if [ "$WARMUP" = "1" ]; then
  echo ">>> warmup pass (1 prompt per shape, discarded) ..."
  for r in "${ROWS[@]}"; do set -- $r; run_one "$1" "$1" "$3" "$4" "$OUT/warm_c$1_o$4.json" >/dev/null 2>&1; done
fi

declare -a SUMMARY
i=0
for r in "${ROWS[@]}"; do
  set -- $r; C=$1; N=$2; ISL=$3; OSL=$4; i=$((i+1))
  L="$OUT/row${i}_c${C}_isl${ISL}_osl${OSL}.log"
  echo ">>> [$(date +%H:%M:%S)] row$i: conc=$C isl=$ISL osl=$OSL n=$N -> $L"
  run_one "$C" "$N" "$ISL" "$OSL" "$OUT/row${i}.json" > "$L" 2>&1
  agg=$(grep -aoE "Output token throughput \(tok/s\):\s+[0-9.]+" "$L" | grep -oE "[0-9.]+$")
  ttft=$(grep -aoE "Mean TTFT \(ms\):\s+[0-9.]+" "$L" | grep -oE "[0-9.]+$")
  tpot=$(grep -aoE "Mean TPOT \(ms\):\s+[0-9.]+" "$L" | grep -oE "[0-9.]+$")
  SUMMARY+=("$(printf '%-4s %-5s %-5s %-6s %-5s %-9s %-9s %-8s' "$i" "$C" "$ISL" "$OSL" "$N" "${agg:-?}" "${ttft:-?}" "${tpot:-?}")")
done

echo
echo "================ RESULTS  (model=$MODEL  $HOST:$PORT) ================"
printf '%-4s %-5s %-5s %-6s %-5s %-9s %-9s %-8s\n' row conc ISL OSL n agg_tok/s TTFT_ms TPOT_ms
printf '%s\n' "${SUMMARY[@]}"
echo "------------------------------------------------------------------"
echo "baseline (gmu0.30): row1 13.28/356  row2 77.47/3025  row3 13.30/357  row4 180.68/539"
echo "logs+JSON: $OUT/"
