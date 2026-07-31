#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# run_test.sh - dispatcher for serving_bench/<suite>/ benchmarks.
#
# Usage:
#   run_test.sh --test <name> --target <url> [--output-dir DIR]
#               [--job-suffix SUFFIX] [--inference-server-dir DIR]
#
# Per-test directory layout:
#   <test>/run.sh           the benchmark
#   <test>/requirements.txt pip deps installed into a fresh per-test venv
#   <test>/defaults.env     `: "${KEY:=value}"` lines; caller env wins
#
# Designed to be runnable locally against `kubectl port-forward`:
#   kubectl -n ttop-ci port-forward svc/blaze-a29-server 8000:8000 &
#   DURATION=60 TARGET_CONCURRENCY=2 \
#     test_module/serving_bench/run_test.sh \
#       --test agentic_bench --target http://localhost:8000 \
#       --output-dir /tmp/results

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

TEST=""
TARGET=""
OUT="./test_results"
JOB_SUFFIX="local"
INFERENCE_SERVER_DIR=""

while (( $# > 0 )); do
  case "$1" in
    --test)                  TEST="$2"; shift 2 ;;
    --target)                TARGET="$2"; shift 2 ;;
    --output-dir)            OUT="$2"; shift 2 ;;
    --job-suffix)            JOB_SUFFIX="$2"; shift 2 ;;
    --inference-server-dir)  INFERENCE_SERVER_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '4,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg '$1'" >&2; exit 2 ;;
  esac
done

[ -n "$TEST" ]   || { echo "ERROR: --test required" >&2; exit 2; }
[ -n "$TARGET" ] || { echo "ERROR: --target required" >&2; exit 2; }

# These suites live at test_module/serving_bench inside
# tt-inference-server, so the repo root three levels up is the default;
# --inference-server-dir overrides it to run against a different checkout.
if [ -z "$INFERENCE_SERVER_DIR" ]; then
  INFERENCE_SERVER_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
fi

# Test name accepts both hyphenated ("agentic-bench") and underscored
# ("agentic_bench") forms — workflows historically use hyphens, directories
# use underscores so the names are valid as Python module identifiers etc.
TEST_DIR_NAME="${TEST//-/_}"
TEST_DIR="$SCRIPT_DIR/$TEST_DIR_NAME"
[ -d "$TEST_DIR" ] || { echo "ERROR: unknown test '$TEST' (no dir $TEST_DIR)" >&2; exit 1; }

mkdir -p "$OUT"
OUT_ABS="$(cd "$OUT" && pwd)"

export OUT="$OUT_ABS"
export TARGET
export JOB_SUFFIX
export TEST_DIR
export INFERENCE_SERVER_DIR
export OPENAI_API_KEY="${OPENAI_API_KEY:-warsaw2026}"

# 1. Load per-test defaults. defaults.env uses `: "${KEY:=value}"` so any KEY
#    already exported by the caller wins; `set -a` auto-exports the vars so
#    they propagate into the `exec`'d run.sh child process.
if [ -f "$TEST_DIR/defaults.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$TEST_DIR/defaults.env"
  set +a
fi

# 2. Fresh per-test venv so deps don't cross-contaminate between tests on the
#    same runner.
VENV="${BENCH_VENV_ROOT:-/tmp}/bench-env-$TEST"
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv (one-time)"
  python3 -m pip install --user --quiet uv
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "==> Creating venv $VENV"
uv venv "$VENV" >/dev/null
# shellcheck disable=SC1091
source "$VENV/bin/activate"
if [ -s "$TEST_DIR/requirements.txt" ]; then
  echo "==> Installing $TEST requirements"
  uv pip install --quiet -r "$TEST_DIR/requirements.txt"
fi

# 3. Wait for the server to be live, then snapshot /info into SERVER_INFO so
#    each test can stamp it onto its results JSON.
"$SCRIPT_DIR/_helpers/wait_for_server.sh" "$TARGET"
SERVER_INFO="$("$SCRIPT_DIR/_helpers/fetch_server_info.sh" "$TARGET")"
export SERVER_INFO

# 4. Run.
echo "==> Running $TEST against $TARGET (results -> $OUT)"
exec "$TEST_DIR/run.sh"
