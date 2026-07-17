#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# run_tests.sh — bring up the Dynamo stack, run the pytest suite, tear down.
#
# Routing modes (ROUTING_MODE):
#   prefill-first  (default)  Dynamo routing + USE_PREFILL_FIRST_DISAGGREGATION
#                             → Dynamo -> prefill -> slot ZMQ -> decode
#   native                    Dynamo routing pools without prefill-first sockets
#   direct                    classic decode-owned offload via MAX_TOKENS_...
#
#   ./run_tests.sh
#   ROUTING_MODE=direct ./run_tests.sh -x -k 03
#
# Clears a leftover deploy.sh stack first so run_stack.sh's etcd lands on the
# reachable bridge (a stale etcd on dynamo-net is reused by name and times out).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STACK="${SCRIPT_DIR}/run_stack.sh"
PYTEST="${PYTEST:-pytest}"
ROUTING_MODE="${ROUTING_MODE:-prefill-first}"
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.43}"
export RESULT_LOG="${RESULT_LOG:-/tmp/tt_test_results.log}"
export MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"

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

configure_routing() {
    case "${ROUTING_MODE}" in
        prefill-first)
            export DYNAMO_ROUTING=1
            export USE_PREFILL_FIRST_DISAGGREGATION=1
            TEST_FILE="${SCRIPT_DIR}/test_prefill_decode_prefill_first.py"
            ;;
        native)
            export DYNAMO_ROUTING=1
            export USE_PREFILL_FIRST_DISAGGREGATION=0
            TEST_FILE="${SCRIPT_DIR}/test_prefill_decode_prefill_first.py"
            ;;
        direct)
            export DYNAMO_ROUTING=0
            export USE_PREFILL_FIRST_DISAGGREGATION=0
            TEST_FILE="${SCRIPT_DIR}/test_prefill_decode.py"
            ;;
        *)
            log "unknown ROUTING_MODE=${ROUTING_MODE} (use prefill-first|native|direct)"
            exit 2
            ;;
    esac
    log "ROUTING_MODE=${ROUTING_MODE} MODEL=${MODEL} test=$(basename "$TEST_FILE")"
}

: > "$RESULT_LOG"
log "result log -> $RESULT_LOG"

configure_routing
guard_etcd
log "stack up"
"$RUN_STACK" up
set +e
"$PYTEST" -v -s "${TEST_FILE}" "$@"
rc=$?
set -e
"$RUN_STACK" down

log "done (rc=$rc); full usage/cache log in $RESULT_LOG"
exit $rc
