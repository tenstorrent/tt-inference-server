# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#!/usr/bin/env bash
#
# single_user_multi_turn.sh
#
# Smoke test that drives a SINGLE simulated user through a long multi-turn
# chat against an OpenAI-compatible /v1/chat/completions endpoint, growing
# the accumulated conversation context toward ~128k tokens.
#
# Traffic shape is intentionally "agentic":
#   - large system prefix (tool descriptions / agent scaffolding)
#   - per-turn user/tool input is LARGER than the assistant output
#   - short assistant replies (decisions / tool calls)
#
# Per-turn input growth (from the guidellm multiturn guide):
#   Turn_N_in = prefix + N*prompt + (N-1)*output
#
# With defaults below (prefix=2048, prompt=4096, output=256, turns=29):
#   Turn 29 in = 2048 + 29*4096 + 28*256 = 128,000 tokens
#   + 256-token reply => total ~128,256 tokens (fits in 131,072 = 128k ctx).
#
# Because guidellm counts each turn as one request, a single 29-turn
# conversation = 29 requests. This is NOT a steady-state benchmark; use it
# to:
#   1. Confirm the server stays healthy as context grows toward the 128k cap.
#   2. Validate prefix caching: turn 2+ should hit cached KV for the whole
#      accumulated history, so TTFT must stay roughly flat even as
#      prompt_tokens grows linearly across turns.
#
# Requires guidellm >= 0.7 (CLI is `guidellm run`, not `guidellm benchmark run`).

set -euo pipefail

TARGET="http://localhost:8000"
MODEL="deepseek-ai/DeepSeek-R1-0528"
API_KEY="your-secret-key"
PREFIX_TOKENS=2048
PROMPT_TOKENS=4096
OUTPUT_TOKENS=256
TURNS=29
OUTPUT_DIR=""

REQUEST_FORMAT="/v1/chat/completions"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Smoke test: single simulated user, multi-turn chat on /v1/chat/completions,
growing context toward ~128k tokens. All configuration is via CLI flags.

Options:
  --target URL            Inference server base URL (default: ${TARGET})
  --model NAME            Model name sent in requests   (default: ${MODEL})
  --api-key KEY           API key placed in Authorization header
                          (default: ${API_KEY})
  --prefix-tokens N       System-prompt length, sent every turn (default: ${PREFIX_TOKENS})
  --prompt-tokens N       New user/tool input per turn         (default: ${PROMPT_TOKENS})
  --output-tokens N       Assistant reply length per turn      (default: ${OUTPUT_TOKENS})
  --turns N               Number of turns in the conversation  (default: ${TURNS})
  --output-dir DIR        Directory for guidellm result files
                          (default: bench_results/single_user_multi_turn/<timestamp>)
  -h, --help              Show this help and exit

Per-turn input grows as: prefix + N*prompt + (N-1)*output
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)         TARGET="$2";         shift 2 ;;
        --model)          MODEL="$2";          shift 2 ;;
        --api-key)        API_KEY="$2";        shift 2 ;;
        --prefix-tokens)  PREFIX_TOKENS="$2";  shift 2 ;;
        --prompt-tokens)  PROMPT_TOKENS="$2";  shift 2 ;;
        --output-tokens)  OUTPUT_TOKENS="$2";  shift 2 ;;
        --turns)          TURNS="$2";          shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v guidellm >/dev/null 2>&1; then
    echo "error: 'guidellm' not found in PATH. Activate cpp_server/.venv" >&2
    echo "       (source .venv/bin/activate) or pip install 'guidellm>=0.7'." >&2
    exit 127
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="bench_results/single_user_multi_turn/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "${OUTPUT_DIR}"

FINAL_TURN_INPUT=$(( PREFIX_TOKENS + TURNS * PROMPT_TOKENS + (TURNS - 1) * OUTPUT_TOKENS ))
FINAL_TURN_TOTAL=$(( FINAL_TURN_INPUT + OUTPUT_TOKENS ))
SCENARIO_FILE="${OUTPUT_DIR}/scenario.json"
RESULT_FILE="${OUTPUT_DIR}/benchmark.json"

# guidellm 0.7: write a scenario file (nested prefix_buckets are awkward on CLI).
python3 - "${SCENARIO_FILE}" "${TARGET}" "${MODEL}" "${API_KEY}" \
    "${REQUEST_FORMAT}" "${PREFIX_TOKENS}" "${PROMPT_TOKENS}" \
    "${OUTPUT_TOKENS}" "${TURNS}" "${RESULT_FILE}" <<'PY'
import json, sys
(
    scenario_file, target, model, api_key, request_format,
    prefix_tokens, prompt_tokens, output_tokens, turns, result_file,
) = sys.argv[1:]
spec = {
    "backend": {
        "kind": "openai_http",
        "target": target,
        "model": model,
        "api_key": api_key,
        "request_format": request_format,
    },
    "profile": {"kind": "concurrent", "streams": [1]},
    "constraints": [{"kind": "max_requests", "count": int(turns)}],
    "data": [{
        "kind": "synthetic_text",
        "prompt_tokens": int(prompt_tokens),
        "output_tokens": int(output_tokens),
        "turns": int(turns),
        "prefix_buckets": [{
            "bucket_weight": 100,
            "prefix_count": 1,
            "prefix_tokens": int(prefix_tokens),
        }],
    }],
    "outputs": [{"kind": "json", "path": result_file}],
}
with open(scenario_file, "w", encoding="utf-8") as f:
    json.dump({"spec": spec}, f, indent=2)
    f.write("\n")
PY

echo "=== Single-user multi-turn smoke test (guidellm) ==="
echo "target           : ${TARGET}"
echo "model            : ${MODEL}"
echo "request format   : ${REQUEST_FORMAT}"
echo "prefix tokens    : ${PREFIX_TOKENS}  (system prompt, sent on every turn)"
echo "prompt tokens    : ${PROMPT_TOKENS}  (new user/tool input per turn)"
echo "output tokens    : ${OUTPUT_TOKENS}  (assistant reply per turn)"
echo "turns            : ${TURNS}  (= total requests issued)"
echo "final turn input : ${FINAL_TURN_INPUT} tokens"
echo "final turn total : ${FINAL_TURN_TOTAL} tokens (input + reply)"
echo "scenario         : ${SCENARIO_FILE}"
echo "output dir       : ${OUTPUT_DIR}"
echo ""

guidellm run --scenario "${SCENARIO_FILE}"

echo ""
echo "Done. Results in ${OUTPUT_DIR}"
echo ""
echo "Prefix-cache sanity check:"
echo "  - Open ${RESULT_FILE} and compare per-turn TTFT vs prompt_tokens."
echo "  - With prefix caching, TTFT should be roughly constant across turns"
echo "    even though prompt_tokens grows by ~${PROMPT_TOKENS}+${OUTPUT_TOKENS} each turn."
