#!/usr/bin/env bash
#
# agentic_bench.sh - long-running agentic-shape multi-turn Poisson soak
#                    against an OpenAI-compatible endpoint, driven by
#                    GuideLLM.
#
# Rolling sub-runs for SIGINT safety; a single self-contained report.html
# is produced at the end.
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# -----------------------------------------------------------------------------
# Defaults (tuned for DeepSeek-R1-0528 on HW sized for 64 concurrent users)
# -----------------------------------------------------------------------------
TARGET="${TARGET:-http://localhost:8000}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
PROCESSOR=""                 # defaults to MODEL at resolve time
DURATION="${DURATION:-28800}"    # 8 h
CHUNK="${CHUNK:-900}"            # 15 min

# Concurrency target (via Little's Law)
#   target_concurrency = mean rate (req/s) x average turn latency (s)
# Set --rate directly if you know the right number, or set
# --target-concurrency + --avg-turn-sec and we compute --rate for you.
RATE=""                                              # unset => derive from target/avg
TARGET_CONCURRENCY="${TARGET_CONCURRENCY:-64}"       # 64 sessions default
AVG_TURN_SEC="${AVG_TURN_SEC:-40}"                   # seconds per turn on real cpp_server; mock is much faster

TURNS="${TURNS:-6}"
PROMPT_TOKENS="${PROMPT_TOKENS:-2000}"
PROMPT_TOKENS_STDEV="${PROMPT_TOKENS_STDEV:-1500}"
PROMPT_TOKENS_MIN="${PROMPT_TOKENS_MIN:-100}"
PROMPT_TOKENS_MAX="${PROMPT_TOKENS_MAX:-30000}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-800}"
OUTPUT_TOKENS_STDEV="${OUTPUT_TOKENS_STDEV:-600}"
OUTPUT_TOKENS_MIN="${OUTPUT_TOKENS_MIN:-64}"
OUTPUT_TOKENS_MAX="${OUTPUT_TOKENS_MAX:-4096}"
PREFIX_TOKENS="${PREFIX_TOKENS:-4000}"
PREFIX_COUNT="${PREFIX_COUNT:-8}"

# Auth: first CLI --api-key wins, otherwise OPENAI_API_KEY env.
API_KEY="${OPENAI_API_KEY:-}"

OUT=""
MAX_ERRORS="${MAX_ERRORS:-500}"
WARMUP="${WARMUP:-0.05}"
EXTRAS=""
DRY_RUN=0
PASSTHROUGH=()

# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
usage() {
  cat <<'EOF'
Usage: agentic_bench.sh [options] [-- <extra guidellm args>]

Target & duration:
  --target URL              OpenAI-compatible base URL       [http://localhost:8000]
  --model NAME              Model id for the backend         [deepseek-ai/DeepSeek-R1-0528]
  --processor NAME          HF tokenizer id                  [<same as --model>]
  --duration SEC            Total run seconds (0 = forever)  [28800]    # 8h
  --chunk SEC               Sub-run chunk seconds            [900]      # 15 min

Concurrency (Little's Law: target_concurrency = rate x avg_turn_latency)
  --rate RPS                Poisson mean requests/sec. If not set, computed
                            from --target-concurrency / --avg-turn-sec.
  --target-concurrency N    Target in-flight turns (~active sessions)  [64]
  --avg-turn-sec S          Expected per-turn latency in seconds       [40]
                            Rough guides: DeepSeek-R1 real HW ~ 30-60s,
                            cpp_server mock pipeline ~ 1-5s.
                            Watch Grafana and re-tune if off.

Agentic-shape synthetic data (maps to guidellm --data key=value,...):
  --turns N                 Turns per conversation           [6]
  --prompt-tokens N         Mean user-prompt tokens/turn     [2000]
  --prompt-tokens-stdev N   Stdev                            [1500]
  --prompt-tokens-min N     Clamp min                        [100]
  --prompt-tokens-max N     Clamp max                        [30000]
  --output-tokens N         Mean assistant-output tokens     [800]
  --output-tokens-stdev N   Stdev                            [600]
  --output-tokens-min N     Clamp min                        [64]
  --output-tokens-max N     Clamp max                        [4096]
  --prefix-tokens N         Size of each shared system prompt[4000]
  --prefix-count N          Pool size of distinct prefixes   [8]

  Note on prefix cost: prefix_count x prefix_tokens prefixes are
  pre-generated eagerly in a single DataLoader worker using the model
  tokenizer. With DeepSeek-R1-style tokenizers this is serial Python
  work. Keep prefix_count x prefix_tokens <= ~40k for fast startup
  (seconds). Very large products can deadlock the GuideLLM
  DataLoader -> scheduler pipe before the first request is ever sent.

Auth (OpenAI-compatible Bearer token):
  --api-key KEY             Sent as Authorization: Bearer KEY.
                            Also honors $OPENAI_API_KEY. If unset no
                            Authorization header is sent.

Output & reliability:
  --out DIR                 Run directory                    [./runs/<ts>]
  --max-errors N            Stop chunk after N errors        [500]
  --warmup FRAC             Per-chunk warmup fraction        [0.05]
  --extras 'JSON'           Extra guidellm --extras          [""]
  --dry-run                 Print the guidellm command and exit
  --help                    Show this and exit

Everything after `--` is forwarded verbatim to `guidellm benchmark`.

Interrupt:
  SIGINT (Ctrl-C) lets the current chunk finish naturally, then merges
  all completed chunks into one report.html. Prior chunks are always
  intact; worst-case data loss is the in-flight requests of the chunk
  that was running at SIGINT time.
EOF
}

# -----------------------------------------------------------------------------
# Arg parsing (supports --foo=bar and --foo bar; everything after -- is passthrough)
# -----------------------------------------------------------------------------
parse_args() {
  while (( $# > 0 )); do
    case "$1" in
      --target|--target=*)                      _read_opt TARGET "$@"; shift $_SHIFT ;;
      --model|--model=*)                        _read_opt MODEL "$@"; shift $_SHIFT ;;
      --processor|--processor=*)                _read_opt PROCESSOR "$@"; shift $_SHIFT ;;
      --rate|--rate=*)                          _read_opt RATE "$@"; shift $_SHIFT ;;
      --target-concurrency|--target-concurrency=*) _read_opt TARGET_CONCURRENCY "$@"; shift $_SHIFT ;;
      --avg-turn-sec|--avg-turn-sec=*)          _read_opt AVG_TURN_SEC "$@"; shift $_SHIFT ;;
      --duration|--duration=*)                  _read_opt DURATION "$@"; shift $_SHIFT ;;
      --chunk|--chunk=*)                        _read_opt CHUNK "$@"; shift $_SHIFT ;;
      --turns|--turns=*)                        _read_opt TURNS "$@"; shift $_SHIFT ;;
      --prompt-tokens|--prompt-tokens=*)        _read_opt PROMPT_TOKENS "$@"; shift $_SHIFT ;;
      --prompt-tokens-stdev|--prompt-tokens-stdev=*) _read_opt PROMPT_TOKENS_STDEV "$@"; shift $_SHIFT ;;
      --prompt-tokens-min|--prompt-tokens-min=*) _read_opt PROMPT_TOKENS_MIN "$@"; shift $_SHIFT ;;
      --prompt-tokens-max|--prompt-tokens-max=*) _read_opt PROMPT_TOKENS_MAX "$@"; shift $_SHIFT ;;
      --output-tokens|--output-tokens=*)        _read_opt OUTPUT_TOKENS "$@"; shift $_SHIFT ;;
      --output-tokens-stdev|--output-tokens-stdev=*) _read_opt OUTPUT_TOKENS_STDEV "$@"; shift $_SHIFT ;;
      --output-tokens-min|--output-tokens-min=*) _read_opt OUTPUT_TOKENS_MIN "$@"; shift $_SHIFT ;;
      --output-tokens-max|--output-tokens-max=*) _read_opt OUTPUT_TOKENS_MAX "$@"; shift $_SHIFT ;;
      --prefix-tokens|--prefix-tokens=*)        _read_opt PREFIX_TOKENS "$@"; shift $_SHIFT ;;
      --prefix-count|--prefix-count=*)          _read_opt PREFIX_COUNT "$@"; shift $_SHIFT ;;
      --api-key|--api-key=*)                    _read_opt API_KEY "$@"; shift $_SHIFT ;;
      --out|--out=*)                            _read_opt OUT "$@"; shift $_SHIFT ;;
      --max-errors|--max-errors=*)              _read_opt MAX_ERRORS "$@"; shift $_SHIFT ;;
      --warmup|--warmup=*)                      _read_opt WARMUP "$@"; shift $_SHIFT ;;
      --extras|--extras=*)                      _read_opt EXTRAS "$@"; shift $_SHIFT ;;
      --dry-run)                                DRY_RUN=1; shift ;;
      -h|--help)                                usage; exit 0 ;;
      --)                                       shift; PASSTHROUGH=("$@"); break ;;
      *)                                        echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
  done
}

# Reads a single option that may be passed as `--foo value` or `--foo=value`.
# Sets the named variable and sets _SHIFT to the number of args consumed.
_read_opt() {
  local var_name="$1"; shift
  local arg="$1"
  if [[ "$arg" == *=* ]]; then
    printf -v "$var_name" '%s' "${arg#*=}"
    _SHIFT=1
  else
    if (( $# < 2 )); then
      echo "option $arg requires a value" >&2; exit 2
    fi
    printf -v "$var_name" '%s' "$2"
    _SHIFT=2
  fi
}

# -----------------------------------------------------------------------------
# Resolve & validate
# -----------------------------------------------------------------------------
resolve() {
  [[ -z "$PROCESSOR" ]] && PROCESSOR="$MODEL"
  [[ -z "$OUT" ]]       && OUT="./runs/$(date +%Y%m%d_%H%M%S)"

  # If --rate wasn't given, derive it from Little's Law.
  if [[ -z "$RATE" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
      echo "error: python3 required to compute --rate from --target-concurrency/--avg-turn-sec" >&2; exit 1
    fi
    RATE=$(python3 -c "print(f'{${TARGET_CONCURRENCY}/${AVG_TURN_SEC}:.3f}')")
    RATE_DERIVED=1
  else
    RATE_DERIVED=0
  fi

  if ! command -v guidellm >/dev/null 2>&1; then
    echo "error: 'guidellm' not on PATH. Install with:" >&2
    echo "   pip install $SCRIPT_DIR/../../guidellm" >&2
    exit 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 not on PATH" >&2; exit 1
  fi

  DATA="turns=${TURNS}"
  DATA+=",prompt_tokens=${PROMPT_TOKENS}"
  DATA+=",prompt_tokens_stdev=${PROMPT_TOKENS_STDEV}"
  DATA+=",prompt_tokens_min=${PROMPT_TOKENS_MIN}"
  DATA+=",prompt_tokens_max=${PROMPT_TOKENS_MAX}"
  DATA+=",output_tokens=${OUTPUT_TOKENS}"
  DATA+=",output_tokens_stdev=${OUTPUT_TOKENS_STDEV}"
  DATA+=",output_tokens_min=${OUTPUT_TOKENS_MIN}"
  DATA+=",output_tokens_max=${OUTPUT_TOKENS_MAX}"
  DATA+=",prefix_count=${PREFIX_COUNT}"
  DATA+=",prefix_tokens=${PREFIX_TOKENS}"

  # GuideLLM takes backend auth via --backend-kwargs JSON. We build it
  # here so users just set --api-key / $OPENAI_API_KEY and don't touch
  # JSON themselves.
  BACKEND_KWARGS=""
  if [[ -n "$API_KEY" ]]; then
    # compact JSON, escape double-quotes in the key just in case
    local escaped_key=${API_KEY//\"/\\\"}
    BACKEND_KWARGS="{\"api_key\":\"${escaped_key}\"}"
  fi
}

# -----------------------------------------------------------------------------
# Write a reproducibility config snapshot
# -----------------------------------------------------------------------------
save_config() {
  local out="$1/config.txt"
  {
    echo "# agentic_bench.sh config snapshot"
    echo "# $(date -u +%FT%TZ)"
    for v in TARGET MODEL PROCESSOR RATE TARGET_CONCURRENCY AVG_TURN_SEC DURATION CHUNK \
             TURNS PROMPT_TOKENS PROMPT_TOKENS_STDEV PROMPT_TOKENS_MIN PROMPT_TOKENS_MAX \
             OUTPUT_TOKENS OUTPUT_TOKENS_STDEV OUTPUT_TOKENS_MIN OUTPUT_TOKENS_MAX \
             PREFIX_TOKENS PREFIX_COUNT OUT MAX_ERRORS WARMUP EXTRAS; do
      printf '%s=%s\n' "$v" "${!v}"
    done
    # never dump the raw API key into files
    if [[ -n "${API_KEY}" ]]; then
      printf 'API_KEY=<set, length %d>\n' "${#API_KEY}"
    else
      printf 'API_KEY=\n'
    fi
    echo "DATA=${DATA}"
    echo "PASSTHROUGH=${PASSTHROUGH[*]:-}"
  } >"$out"
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
main() {
  parse_args "$@"
  resolve

  if (( DRY_RUN == 1 )); then
    printf 'guidellm benchmark \\\n'
    printf '  --target %q \\\n'       "$TARGET"
    printf '  --model %q \\\n'        "$MODEL"
    printf '  --processor %q \\\n'    "$PROCESSOR"
    printf '  --profile poisson \\\n'
    printf '  --rate %q \\\n'         "$RATE"
    printf '  --max-seconds %q \\\n'  "$CHUNK"
    printf '  --max-errors %q \\\n'   "$MAX_ERRORS"
    printf '  --warmup %q \\\n'       "$WARMUP"
    printf '  --request-type chat_completions \\\n'
    printf '  --data %q \\\n'         "$DATA"
    [[ -n "$EXTRAS" ]] && printf '  --extras %q \\\n' "$EXTRAS"
    if [[ -n "$BACKEND_KWARGS" ]]; then
      printf '  --backend-kwargs %q \\\n' "$BACKEND_KWARGS"
    fi
    printf '  --output-path <chunk_dir>/benchmarks.json'
    if (( ${#PASSTHROUGH[@]} > 0 )); then
      printf ' \\\n  '
      printf '%q ' "${PASSTHROUGH[@]}"
    fi
    printf '\n'
    exit 0
  fi

  mkdir -p "$OUT"
  save_config "$OUT"
  echo "==> run dir: $OUT"
  echo "==> target:  $TARGET"
  echo "==> model:   $MODEL"
  if (( RATE_DERIVED == 1 )); then
    echo "==> rate:    ${RATE} req/s (Poisson; derived: ${TARGET_CONCURRENCY} target / ${AVG_TURN_SEC}s avg turn)"
  else
    echo "==> rate:    ${RATE} req/s (Poisson; explicit --rate)"
  fi
  echo "==> plan:    $(( DURATION > 0 ? (DURATION + CHUNK - 1) / CHUNK : 0 )) chunks of ${CHUNK}s each (total ${DURATION}s)"
  if [[ -n "$API_KEY" ]]; then
    echo "==> auth:    Bearer <api key set, length ${#API_KEY}>"
  else
    echo "==> auth:    none (no --api-key / \$OPENAI_API_KEY set)"
  fi
  echo "==> prefix:  ${PREFIX_COUNT} x ${PREFIX_TOKENS} tokens ($(( PREFIX_COUNT * PREFIX_TOKENS )) total) - pre-generated once per chunk"
  echo

  STOP=0
  trap '
    if (( STOP == 0 )); then
      echo; echo "==> SIGINT received - current chunk will finish, then merge and exit."
      STOP=1
    fi
  ' INT TERM

  merger="$SCRIPT_DIR/scripts/merge_report.py"
  trap '
    if [[ -f "$merger" ]]; then
      echo "==> merging chunks into report.html"
      python3 "$merger" "$OUT" || echo "merge_report.py failed (chunks still on disk in $OUT)"
    fi
  ' EXIT

  i=0
  elapsed=0
  while (( STOP == 0 )) && { (( DURATION == 0 )) || (( elapsed < DURATION )); }; do
    chunk_sec=$CHUNK
    if (( DURATION > 0 )) && (( elapsed + chunk_sec > DURATION )); then
      chunk_sec=$(( DURATION - elapsed ))
    fi

    chunk_dir="$OUT/chunk_$(printf '%04d' "$i")"
    mkdir -p "$chunk_dir"
    echo "==> chunk $i - ${chunk_sec}s (elapsed ${elapsed}/${DURATION}s)"

    extras_args=()
    [[ -n "$EXTRAS" ]] && extras_args=(--extras "$EXTRAS")
    backend_args=()
    [[ -n "$BACKEND_KWARGS" ]] && backend_args=(--backend-kwargs "$BACKEND_KWARGS")
    passthrough_args=()
    (( ${#PASSTHROUGH[@]} > 0 )) && passthrough_args=("${PASSTHROUGH[@]}")

    if ! guidellm benchmark \
          --target "$TARGET" \
          --model "$MODEL" \
          --processor "$PROCESSOR" \
          --profile poisson \
          --rate "$RATE" \
          --max-seconds "$chunk_sec" \
          --max-errors "$MAX_ERRORS" \
          --warmup "$WARMUP" \
          --request-type chat_completions \
          --data "$DATA" \
          "${extras_args[@]}" \
          "${backend_args[@]}" \
          --output-path "$chunk_dir/benchmarks.json" \
          "${passthrough_args[@]}"; then
      echo "!! chunk $i failed; continuing (artifacts under $chunk_dir)"
    fi

    elapsed=$(( elapsed + chunk_sec ))
    i=$(( i + 1 ))
  done
  echo "==> loop finished (chunks=$i, elapsed=${elapsed}s)"
}

main "$@"
