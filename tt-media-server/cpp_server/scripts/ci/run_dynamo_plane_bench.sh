#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# CI entrypoint for scripts/bench_dynamo_request_planes.sh.
# Expects etcd already listening at ETCD_ENDPOINTS and cpp_server built under build/.

set -euo pipefail

CPP_SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DYN_VENV="${DYN_VENV:-${HOME}/.venvs/dynamo}"
export BENCH_VENV="${BENCH_VENV:-${DYN_VENV}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CPP_SERVER_DIR}/dynamo_plane_bench}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-empty}"
export DEVICE_IDS="${DEVICE_IDS:-(0)}"
export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-etcd}"
export ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"
export DYN_EVENT_PLANE="${DYN_EVENT_PLANE:-zmq}"
export LLM_DEVICE_BACKEND="${LLM_DEVICE_BACKEND:-mock_pipeline}"
export DYNAMO_ENDPOINT_ENABLED=1
export TT_LOG_LEVEL="${TT_LOG_LEVEL:-warn}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dynamo-bypass}"

# Same load profile as dynamo-frontend-integration; override for smoke runs.
export NUM_PROMPTS="${NUM_PROMPTS:-1000}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
export RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-128}"
export RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"

exec "${CPP_SERVER_DIR}/scripts/bench_dynamo_request_planes.sh"
