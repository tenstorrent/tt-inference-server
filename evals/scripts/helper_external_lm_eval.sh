#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Run a single lm-eval task against a remote server exposing OpenAI-compatible
# completions or chat-completions endpoints.

set -euo pipefail

usage() {
    cat <<'EOF' >&2
usage: helper_external_lm_eval.sh --task TASK [options]

Required:
  --task TASK               lm-eval task name

Options:
  --chat-api                Use /v1/chat/completions with local-chat-completions (default)
  --completions-api         Use /v1/completions with local-completions
  --include-path PATH       Extra task include path
  --output-dir DIR          lm-eval output directory
  --limit N|PCT            Pass through lm-eval --limit
  --num-fewshot N          Override lm-eval --num_fewshot
  --batch-size N           Batch size (default: 1)
  --max-concurrent N       Add num_concurrent=N to model_args (default: 15)
  --max-gen-toks N         Generation token cap (default: 65535)
  --temperature N          Sampling temperature (default: 0.6)
  --top-p N                Nucleus sampling top_p (default: 0.95)
  --model-seed N           API model seed; useful for repeated sampling
  --gen-kwargs STR         Pass through lm-eval --gen_kwargs
  --stream                 Request streaming responses (default; also STREAM=1)
  --no-stream              Disable streaming responses (also STREAM=0)
  --no-apply-chat-template Disable --apply_chat_template
  --show-config            Print full lm-eval task configuration after evaluation
  --print-command          Print the full lm-eval command before running
  --log-level LEVEL        lm-eval log level (default: WARNING)
  --help                   Show this message

Environment:
  BASE_URL / DEPLOY_URL / SERVICE_PORT (default: https://console.tenstorrent.com)
  OPENAI_API_KEY
  VLLM_API_KEY
  VLLM_MODEL
  TOKENIZER_MODEL
  MAX_CONCURRENT
  MAX_GEN_TOKS
  TEMPERATURE
  TOP_P
  MODEL_SEED
  STREAM
  LMEVAL_LOG_LEVEL
  LM_EVAL_VENV
  LM_EVAL_BIN
EOF
    exit 64
}

TASK=""
INCLUDE_PATH=""
OUTPUT_DIR="${OUTPUT_DIR:-}"
LIMIT_ARG=""
NUM_FEWSHOT=""
BATCH_SIZE="1"
MAX_CONCURRENT="${MAX_CONCURRENT:-15}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-65535}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MODEL_SEED="${MODEL_SEED:-}"
GEN_KWARGS=""
STREAM="${STREAM:-1}"
USE_CHAT_API="1"
APPLY_CHAT_TEMPLATE="1"
SHOW_CONFIG="0"
PRINT_COMMAND="0"
LOG_LEVEL="${LMEVAL_LOG_LEVEL:-WARNING}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK="${2:-}"
            shift 2
            ;;
        --chat-api)
            USE_CHAT_API="1"
            shift
            ;;
        --completions-api)
            USE_CHAT_API="0"
            shift
            ;;
        --include-path)
            INCLUDE_PATH="${2:-}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:-}"
            shift 2
            ;;
        --limit)
            LIMIT_ARG="${2:-}"
            shift 2
            ;;
        --num-fewshot)
            NUM_FEWSHOT="${2:-}"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="${2:-}"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="${2:-}"
            shift 2
            ;;
        --max-gen-toks)
            MAX_GEN_TOKS="${2:-}"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="${2:-}"
            shift 2
            ;;
        --top-p)
            TOP_P="${2:-}"
            shift 2
            ;;
        --model-seed)
            MODEL_SEED="${2:-}"
            shift 2
            ;;
        --gen-kwargs)
            GEN_KWARGS="${2:-}"
            shift 2
            ;;
        --stream)
            STREAM="1"
            shift
            ;;
        --no-stream)
            STREAM="0"
            shift
            ;;
        --no-apply-chat-template)
            APPLY_CHAT_TEMPLATE="0"
            shift
            ;;
        --show-config)
            SHOW_CONFIG="1"
            shift
            ;;
        --print-command)
            PRINT_COMMAND="1"
            shift
            ;;
        --log-level)
            LOG_LEVEL="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            ;;
    esac
done

if [[ -z "${TASK}" ]]; then
    usage
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_LM_EVAL_VENV="${REPO_ROOT}/.workflow_venvs/.venv_evals_common"
LM_EVAL_VENV_OVERRIDDEN="0"
LM_EVAL_BIN_OVERRIDDEN="0"
if [[ -n "${LM_EVAL_VENV:-}" ]]; then
    LM_EVAL_VENV_OVERRIDDEN="1"
fi
LM_EVAL_VENV="${LM_EVAL_VENV:-${DEFAULT_LM_EVAL_VENV}}"
if [[ -n "${LM_EVAL_BIN:-}" ]]; then
    LM_EVAL_BIN_OVERRIDDEN="1"
fi
LM_EVAL_BIN="${LM_EVAL_BIN:-${LM_EVAL_VENV}/bin/lm_eval}"
REDACTOR="${REPO_ROOT}/evals/scripts/helper_redact_tt_keys.py"
STREAMING_PATCH_DIR="${REPO_ROOT}/evals/scripts/lm_eval_streaming_patch"
DEFAULT_BASE_URL="https://console.tenstorrent.com"
DEFAULT_TOKEN_FILE="${HOME}/.tt-api-token"
DEFAULT_KEY_URL="https://console.tenstorrent.com/dashboard/inference/keys"

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

append_gen_kwarg_if_missing() {
    local key="$1"
    local value="$2"
    if [[ -z "${GEN_KWARGS}" ]]; then
        GEN_KWARGS="${key}=${value}"
    elif [[ ",${GEN_KWARGS}," != *",${key}="* ]]; then
        GEN_KWARGS="${GEN_KWARGS},${key}=${value}"
    fi
}

read_default_token() {
    local token
    token="$(tr -d '\r\n' < "${DEFAULT_TOKEN_FILE}")"
    token="${token#"${token%%[![:space:]]*}"}"
    token="${token%"${token##*[![:space:]]}"}"
    case "${token}" in
        "Bearer "*) token="${token#Bearer }" ;;
        "bearer "*) token="${token#bearer }" ;;
    esac
    printf '%s' "${token}"
}

print_tt_key_setup() {
    local reason="$1"
    cat >&2 <<EOF
${reason}

Request a TT Console inference key at:
  ${DEFAULT_KEY_URL}

Then save it locally:
  printf '%s\n' 'sk-tt-...' > ${DEFAULT_TOKEN_FILE}
  chmod 600 ${DEFAULT_TOKEN_FILE}

Alternatively, export it for this shell:
  export OPENAI_API_KEY='sk-tt-...'
EOF
}

find_host_python() {
    if [[ -n "${PYTHON:-}" ]]; then
        printf '%s' "${PYTHON}"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif command -v python >/dev/null 2>&1; then
        command -v python
    else
        return 1
    fi
}

bootstrap_lm_eval_venv() {
    if [[ -x "${LM_EVAL_BIN}" ]]; then
        return
    fi

    if [[ "${LM_EVAL_BIN_OVERRIDDEN}" == "1" || "${LM_EVAL_VENV_OVERRIDDEN}" == "1" ]]; then
        echo "lm_eval not found at ${LM_EVAL_BIN}." >&2
        echo "Unset LM_EVAL_BIN/LM_EVAL_VENV to use the auto-managed default env, or point them at a valid lm_eval install." >&2
        exit 1
    fi

    local host_python
    if ! host_python="$(find_host_python)"; then
        echo "Could not find python3 or python to bootstrap ${LM_EVAL_VENV}." >&2
        echo "Install Python, then rerun this script." >&2
        exit 1
    fi

    cat >&2 <<EOF
lm_eval was not found at:
  ${LM_EVAL_BIN}

Bootstrapping the default eval environment now:
  ${LM_EVAL_VENV}

This is a one-time setup and can take 5 to 15+ minutes on the first run.
EOF

    local uv_venv="${REPO_ROOT}/.workflow_venvs/.venv_bootstrap_uv"
    local uv_exec="${uv_venv}/bin/uv"

    mkdir -p "${REPO_ROOT}/.workflow_venvs"

    if [[ ! -x "${uv_exec}" ]]; then
        echo "Installing uv into ${uv_venv} ..." >&2
        if [[ -d "${uv_venv}" && ! -x "${uv_venv}/bin/python" ]]; then
            rm -rf "${uv_venv}"
        fi
        "${host_python}" -m venv "${uv_venv}"
        if [[ ! -x "${uv_venv}/bin/pip" ]]; then
            "${uv_venv}/bin/python" -m ensurepip --upgrade
        fi
        "${uv_venv}/bin/pip" install uv
    fi

    "${uv_exec}" venv --managed-python --python=3.10 "${LM_EVAL_VENV}" --allow-existing
    "${uv_exec}" pip install \
        --managed-python \
        --python "${LM_EVAL_VENV}/bin/python" \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "git+https://github.com/tstescoTT/lm-evaluation-harness.git@evals-common#egg=lm-eval[api,ifeval,math,sentencepiece,r1_evals,ruler,longbench,hf]" \
        protobuf \
        pillow==11.1 \
        pyjwt==2.7.0 \
        datasets==3.1.0

    if [[ ! -x "${LM_EVAL_BIN}" ]]; then
        echo "Bootstrap completed, but lm_eval is still missing at ${LM_EVAL_BIN}." >&2
        echo "Remove .workflow_venvs and rerun, or inspect the install logs above." >&2
        exit 1
    fi
}

if [[ -n "${BASE_URL:-}" ]]; then
    ROOT_URL="${BASE_URL}"
elif [[ -n "${DEPLOY_URL:-}" ]]; then
    ROOT_URL="${DEPLOY_URL%/}:${SERVICE_PORT:-443}"
else
    ROOT_URL="${DEFAULT_BASE_URL}"
fi

# Strip any trailing /v1 or /v1/... so we can append cleanly.
ROOT_URL="${ROOT_URL%/}"
ROOT_URL="${ROOT_URL%/v1}"
ROOT_URL="${ROOT_URL%/v1/chat/completions}"
ROOT_URL="${ROOT_URL%/v1/completions}"

if [[ "${USE_CHAT_API}" == "1" ]]; then
    API_URL="${ROOT_URL}/v1/chat/completions"
    EVAL_MODEL="local-chat-completions"
else
    API_URL="${ROOT_URL}/v1/completions"
    EVAL_MODEL="local-completions"
fi

if [[ "${ROOT_URL}" == "${DEFAULT_BASE_URL}" && -s "${DEFAULT_TOKEN_FILE}" && -n "${OPENAI_API_KEY:-}" && "${OPENAI_API_KEY}" != sk-tt-* ]]; then
    echo "OPENAI_API_KEY is set but does not look like a TT Console key; using ${DEFAULT_TOKEN_FILE} for ${DEFAULT_BASE_URL}." >&2
    export OPENAI_API_KEY="$(read_default_token)"
elif [[ "${ROOT_URL}" == "${DEFAULT_BASE_URL}" && -n "${OPENAI_API_KEY:-}" && "${OPENAI_API_KEY}" != sk-tt-* ]]; then
    print_tt_key_setup "OPENAI_API_KEY is set but does not look like a TT Console key, and ${DEFAULT_TOKEN_FILE} was not found."
    exit 1
elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
    export OPENAI_API_KEY
elif [[ "${ROOT_URL}" == "${DEFAULT_BASE_URL}" && -s "${DEFAULT_TOKEN_FILE}" && -n "${VLLM_API_KEY:-}" && "${VLLM_API_KEY}" != sk-tt-* ]]; then
    echo "VLLM_API_KEY is set but does not look like a TT Console key; using ${DEFAULT_TOKEN_FILE} for ${DEFAULT_BASE_URL}." >&2
    export OPENAI_API_KEY="$(read_default_token)"
elif [[ "${ROOT_URL}" == "${DEFAULT_BASE_URL}" && -n "${VLLM_API_KEY:-}" && "${VLLM_API_KEY}" != sk-tt-* ]]; then
    print_tt_key_setup "VLLM_API_KEY is set but does not look like a TT Console key, and ${DEFAULT_TOKEN_FILE} was not found."
    exit 1
elif [[ -n "${VLLM_API_KEY:-}" ]]; then
    export OPENAI_API_KEY="${VLLM_API_KEY}"
elif [[ -s "${DEFAULT_TOKEN_FILE}" ]]; then
    export OPENAI_API_KEY="$(read_default_token)"
elif [[ "${ROOT_URL}" == "${DEFAULT_BASE_URL}" ]]; then
    print_tt_key_setup "No TT Console API key found. OPENAI_API_KEY/VLLM_API_KEY are unset and ${DEFAULT_TOKEN_FILE} does not exist."
    exit 1
else
    export OPENAI_API_KEY="your-secret-key"
fi

bootstrap_lm_eval_venv

MODEL_NAME="${VLLM_MODEL:-deepseek-ai/DeepSeek-R1-0528}"
TOKENIZER_NAME="${TOKENIZER_MODEL:-${MODEL_NAME}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/${TASK}_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

MODEL_ARGS="model=${MODEL_NAME}"
if [[ "${USE_CHAT_API}" != "1" ]]; then
    MODEL_ARGS="${MODEL_ARGS},tokenizer=${TOKENIZER_NAME},tokenizer_backend=huggingface"
fi
if [[ -n "${MAX_CONCURRENT}" ]]; then
    MODEL_ARGS="${MODEL_ARGS},num_concurrent=${MAX_CONCURRENT}"
fi
if [[ -n "${MODEL_SEED}" ]]; then
    MODEL_ARGS="${MODEL_ARGS},seed=${MODEL_SEED}"
fi
MODEL_ARGS="${MODEL_ARGS},base_url=${API_URL}"

if [[ -n "${MAX_GEN_TOKS}" ]]; then
    append_gen_kwarg_if_missing max_gen_toks "${MAX_GEN_TOKS}"
fi
if [[ -n "${TEMPERATURE}" ]]; then
    append_gen_kwarg_if_missing temperature "${TEMPERATURE}"
fi
if [[ -n "${TOP_P}" ]]; then
    append_gen_kwarg_if_missing top_p "${TOP_P}"
fi

# Always load local lm_eval compatibility patches. The streaming request parser
# only activates when stream=true, but other fixes such as LiveCodeBench sample
# logging are useful for non-streaming runs too.
export PYTHONPATH="${STREAMING_PATCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if is_truthy "${STREAM}"; then
    append_gen_kwarg_if_missing stream true
    if [[ "${MAX_CONCURRENT}" =~ ^[0-9]+$ && "${MAX_CONCURRENT}" -gt 15 ]]; then
        echo "STREAM=1 with MAX_CONCURRENT=${MAX_CONCURRENT} opens many long-lived requests; TT Console may return 504s or interrupted streams. Prefer MAX_CONCURRENT<=15." >&2
    fi
fi

cmd=(
    "${LM_EVAL_BIN}"
    --tasks "${TASK}"
    --model "${EVAL_MODEL}"
    --model_args "${MODEL_ARGS}"
    --output_path "${OUTPUT_DIR}"
    --seed 42
    --batch_size "${BATCH_SIZE}"
    --log_samples
    --trust_remote_code
    --confirm_run_unsafe_code
)

if [[ -n "${GEN_KWARGS}" ]]; then
    cmd+=(--gen_kwargs "${GEN_KWARGS}")
fi

if [[ "${SHOW_CONFIG}" == "1" ]]; then
    cmd+=(--show_config)
fi

if [[ -n "${NUM_FEWSHOT}" ]]; then
    cmd+=(--num_fewshot "${NUM_FEWSHOT}")
fi

if [[ -n "${LIMIT_ARG}" ]]; then
    cmd+=(--limit "${LIMIT_ARG}")
fi

if [[ "${APPLY_CHAT_TEMPLATE}" == "1" ]]; then
    cmd+=(--apply_chat_template)
fi

if [[ -n "${INCLUDE_PATH}" ]]; then
    cmd+=(--include_path "${INCLUDE_PATH}")
fi

export LMEVAL_LOG_LEVEL="${LOG_LEVEL}"

echo "Running ${TASK} on ${MODEL_NAME} @ ${API_URL}"
echo "Max concurrent requests: ${MAX_CONCURRENT:-1}"
if [[ -n "${MODEL_SEED}" ]]; then
    echo "Model seed: ${MODEL_SEED}"
fi
echo "Generation kwargs: ${GEN_KWARGS:-task defaults}"
echo "Output dir: ${OUTPUT_DIR}"

if [[ "${PRINT_COMMAND}" == "1" ]]; then
    printf '+'
    printf ' %q' "${cmd[@]}"
    printf '\n'
fi

set +e
if [[ -x "${REDACTOR}" ]]; then
    "${cmd[@]}" 2>&1 | "${REDACTOR}"
    status="${PIPESTATUS[0]}"
else
    "${cmd[@]}"
    status=$?
fi
set -e

if [[ -d "${OUTPUT_DIR}" ]]; then
    if [[ -x "${REDACTOR}" ]]; then
        while IFS= read -r artifact; do
            "${REDACTOR}" "${artifact}"
        done < <(find "${OUTPUT_DIR}" -maxdepth 3 -type f \( -name '*.json' -o -name '*.jsonl' \))
    fi
fi

exit "${status}"
