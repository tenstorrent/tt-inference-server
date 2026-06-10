#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# run_sharegpt_multiturn.sh
#
# Multi-turn benchmark over an OpenAI-compatible /v1/chat/completions endpoint
# using real-English ShareGPT conversations driven by guidellm.
#
# Why this exists:
#   - vllm bench serve random / guidellm synthetic = lorem ipsum, no chat
#     history, weak signal for prefix caching and speculative decoding.
#   - Goal of this script: SIMPLE, MULTI-TURN, REAL ENGLISH baseline for CI.
#
# How multi-turn works in guidellm 0.6.0:
#   guidellm detects per-turn columns by suffix (`prompt_0`, `prompt_1`, ...).
#   Each turn issues one request to /v1/chat/completions; the model's reply is
#   appended as an `assistant` message in the next turn's `messages` array.
#   So we feed ONLY the human prompts from each ShareGPT conversation -- the
#   assistant turns come from the actual server under test.

set -euo pipefail

TARGET="${TARGET:-http://localhost:8000}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
API_KEY="${API_KEY:-your-secret-key}"
CONCURRENCY=64
DURATION_SECONDS=3600
DATASET_ID="anon8231489123/ShareGPT_Vicuna_unfiltered"
DATASET_FILE="ShareGPT_V3_unfiltered_cleaned_split.json"
MIN_TURNS=4
MAX_TURNS=20
MAX_OUTPUT_TOKENS=512
DATA_CACHE_DIR="${HOME}/.cache/tt_benchmarks/sharegpt_multiturn"
OUTPUT_DIR=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Multi-turn ShareGPT benchmark via guidellm.
Default: ${DURATION_SECONDS}s (1h) at ${CONCURRENCY} concurrent users.

Options:
  --target URL            Inference server base URL    (default: ${TARGET})
  --model NAME            Model name in requests       (default: ${MODEL})
  --api-key KEY           Bearer token                 (default: ${API_KEY})
  --concurrency N         Concurrent users / in-flight (default: ${CONCURRENCY})
  --duration SECONDS      Wall-clock benchmark length  (default: ${DURATION_SECONDS})
  --min-turns N           Drop conversations shorter than this many human turns
                                                       (default: ${MIN_TURNS})
  --max-turns N           Truncate conversations to this many human turns
                                                       (default: ${MAX_TURNS})
  --max-output-tokens N   max_completion_tokens per turn
                                                       (default: ${MAX_OUTPUT_TOKENS})
  --dataset-id ID         HuggingFace dataset id       (default: ${DATASET_ID})
  --dataset-file NAME     File within the dataset repo (default: ${DATASET_FILE})
  --data-cache-dir DIR    Where to cache the flattened JSONL
                                                       (default: ${DATA_CACHE_DIR})
  --output-dir DIR        Directory for guidellm result files
                          (default: bench_results/sharegpt_multiturn)
                          A file 'benchmark_<timestamp>.json' is created here.
  -h, --help              Show this help and exit

Notes:
  - Each row in the prepped JSONL has columns prompt_0..prompt_{N-1} (one per
    human turn) plus output_tokens_count (caps each turn's reply). guidellm
    expands each row into N sequential requests, threading the model's actual
    response as conversation history.
  - First run flattens ShareGPT into a JSONL cache; subsequent runs reuse it.
  - HF download requires HF_TOKEN if you swap in a gated dataset.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)            TARGET="$2";            shift 2 ;;
        --model)             MODEL="$2";             shift 2 ;;
        --api-key)           API_KEY="$2";           shift 2 ;;
        --concurrency)       CONCURRENCY="$2";       shift 2 ;;
        --duration)          DURATION_SECONDS="$2";  shift 2 ;;
        --min-turns)         MIN_TURNS="$2";         shift 2 ;;
        --max-turns)         MAX_TURNS="$2";         shift 2 ;;
        --max-output-tokens) MAX_OUTPUT_TOKENS="$2"; shift 2 ;;
        --dataset-id)        DATASET_ID="$2";        shift 2 ;;
        --dataset-file)      DATASET_FILE="$2";      shift 2 ;;
        --data-cache-dir)    DATA_CACHE_DIR="$2";    shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";        shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v guidellm >/dev/null 2>&1; then
    echo "error: 'guidellm' not found in PATH. Activate the venv that has it" >&2
    echo "       (e.g. tt-media-server/cpp_server/.venv) or pip install guidellm." >&2
    exit 127
fi

mkdir -p "${DATA_CACHE_DIR}"
DATASET_SLUG="$(echo "${DATASET_ID}" | tr '/' '_')"
CACHE_FILE="${DATA_CACHE_DIR}/${DATASET_SLUG}_min${MIN_TURNS}_max${MAX_TURNS}_out${MAX_OUTPUT_TOKENS}.jsonl"

if [[ ! -s "${CACHE_FILE}" ]]; then
    echo "Preparing ShareGPT multi-turn cache at ${CACHE_FILE}"
    DATASET_ID="${DATASET_ID}" \
    DATASET_FILE="${DATASET_FILE}" \
    CACHE_FILE="${CACHE_FILE}" \
    MIN_TURNS="${MIN_TURNS}" \
    MAX_TURNS="${MAX_TURNS}" \
    MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS}" \
    python3 - <<'PYEOF'
import json
import os
import pathlib

from datasets import load_dataset

dataset_id = os.environ["DATASET_ID"]
dataset_file = os.environ["DATASET_FILE"]
cache = pathlib.Path(os.environ["CACHE_FILE"])
min_turns = int(os.environ["MIN_TURNS"])
max_turns = int(os.environ["MAX_TURNS"])
max_output_tokens = int(os.environ["MAX_OUTPUT_TOKENS"])

load_kwargs = {"split": "train"}
if dataset_file:
    load_kwargs["data_files"] = dataset_file
ds = load_dataset(dataset_id, **load_kwargs)

written = 0
cache.parent.mkdir(parents=True, exist_ok=True)
with cache.open("w") as f:
    for row in ds:
        convs = row.get("conversations") or row.get("conversation") or []
        prompts = []
        for msg in convs:
            role = msg.get("from") or msg.get("role")
            if role in ("human", "user"):
                value = (msg.get("value") or msg.get("content") or "").strip()
                if value:
                    prompts.append(value)
        if len(prompts) < min_turns:
            continue
        prompts = prompts[:max_turns]
        record = {f"prompt_{i}": p for i, p in enumerate(prompts)}
        record["output_tokens_count"] = max_output_tokens
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1

print(f"wrote {written} multi-turn conversations to {cache}")
PYEOF
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="bench_results/sharegpt_multiturn"
fi
mkdir -p "${OUTPUT_DIR}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
RESULT_FILE="${OUTPUT_DIR}/benchmark_${RUN_TS}.json"

echo "=== ShareGPT multi-turn benchmark (guidellm) ==="
echo "target           : ${TARGET}"
echo "model            : ${MODEL}"
echo "concurrency      : ${CONCURRENCY}"
echo "duration         : ${DURATION_SECONDS}s"
echo "dataset          : ${DATASET_ID}/${DATASET_FILE}"
echo "turns range      : [${MIN_TURNS}, ${MAX_TURNS}]"
echo "max_output_tokens: ${MAX_OUTPUT_TOKENS} per turn"
echo "data cache       : ${CACHE_FILE}"
echo "result file      : ${RESULT_FILE}"
echo ""

guidellm benchmark run \
    --target "${TARGET}" \
    --model "${MODEL}" \
    --request-format /v1/chat/completions \
    --data "${CACHE_FILE}" \
    --profile concurrent \
    --rate "${CONCURRENCY}" \
    --max-seconds "${DURATION_SECONDS}" \
    --backend-kwargs "{\"api_key\":\"${API_KEY}\"}" \
    --output-path "${RESULT_FILE}"

echo ""
echo "Done. Result: ${RESULT_FILE}"
