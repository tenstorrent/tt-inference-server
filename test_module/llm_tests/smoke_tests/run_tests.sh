#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# run_tests.sh — bring up the disaggregated stack, run the pytest suite, tear down.
#
#   ./run_tests.sh             # test_prefill_decode.py against decode + prefill
#   ./run_tests.sh -x -k 03    # extra args pass through to pytest
#
# Clears a leftover deploy.sh stack first so run_stack.sh's etcd lands on the
# reachable bridge (a stale etcd on dynamo-net is reused by name and times out).

set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
BENCH_DIR="${REPO_ROOT}/tt-media-server/cpp_server/benchmarks"
RUN_STACK="${BENCH_DIR}/run_stack.sh"
TEST_FILE="${SCRIPT_DIR}/test_prefill_decode.py"
# Prefer a local .venv (pytest + datasets) if present; else PATH pytest.
if [[ -z "${PYTEST:-}" && -x "${BENCH_DIR}/.venv/bin/pytest" ]]; then
    PYTEST="${BENCH_DIR}/.venv/bin/pytest"
fi
PYTEST="${PYTEST:-pytest}"
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.43}"
export RESULT_LOG="${RESULT_LOG:-/tmp/tt_test_results.log}"
export MAX_ISL="${MAX_ISL:-64000}"

log() { printf '[run_tests] %s\n' "$*"; }

guard_etcd() {
    docker rm -f dynamo-frontend tt-cpp-worker >/dev/null 2>&1 || true
    local ip; ip="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' etcd 2>/dev/null || true)"
    case "$ip" in
        172.17.*) ;;                                   # already on the reachable bridge
        *) [[ -n "$ip" ]] && log "removing stale etcd at $ip (not on bridge)"
           docker rm -f etcd >/dev/null 2>&1 || true ;;
    esac
}

: > "$RESULT_LOG"
log "result log -> $RESULT_LOG"

guard_etcd
log "stack up (disaggregated)"
"$RUN_STACK" up
set +e
"$PYTEST" -v -s "${TEST_FILE}" "$@"
rc=$?
set -e
"$RUN_STACK" down

log "done (rc=$rc); full usage/cache log in $RESULT_LOG"
exit $rc
