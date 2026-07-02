# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# lib_dynamo_stack.sh — shared helpers for the Dynamo decode/prefill stack
# scripts. Sourced (not executed) by:
#   tt-media-server/cpp_server/benchmarks/run_stack.sh  (host-process mock stack)
#   dynamo_frontend/deploy.sh                            (containerized stack)
#
# Set LOG_PREFIX before sourcing to tag log()/die() output.

LOG_PREFIX="${LOG_PREFIX:-dynamo}"
ETCD_IMAGE="${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.13}"

log() { printf '[%s] %s\n' "${LOG_PREFIX}" "$*"; }
die() { printf '[%s] %s\n' "${LOG_PREFIX}" "$*" >&2; exit 1; }

# start_etcd <name> [network]
# Runs the etcd container on the default bridge, or on <network> if given.
start_etcd() {
    local name="$1" network="${2:-}"
    local net_args=()
    [[ -n "${network}" ]] && net_args=(--network "${network}")
    docker run -d --name "${name}" "${net_args[@]}" -p 2379:2379 \
        "${ETCD_IMAGE}" /usr/local/bin/etcd --name dyn-etcd \
            --advertise-client-urls http://0.0.0.0:2379 \
            --listen-client-urls http://0.0.0.0:2379 >/dev/null
}

# wait_etcd_healthy <name> [retries]
# Polls etcdctl endpoint health; dies (dumping recent logs) if it never passes.
wait_etcd_healthy() {
    local name="$1" retries="${2:-30}"
    for _ in $(seq 1 "${retries}"); do
        docker exec "${name}" etcdctl endpoint health >/dev/null 2>&1 && return 0
        sleep 1
    done
    docker logs --tail 50 "${name}" >&2 2>/dev/null || true
    die "etcd never became healthy"
}

# dynamo_worker_env <etcd-endpoints> [mode]
# Emits the dynamo registration env for the worker the frontend discovers.
# mode "env" (default) -> bare KEY=val lines (host `env`);
# mode "docker"        -> `-e KEY=val` tokens (docker run).
dynamo_worker_env() {
    local endpoints="$1" mode="${2:-env}" p=""
    [[ "${mode}" == "docker" ]] && p="-e "
    printf '%sDYNAMO_ENDPOINT_ENABLED=1\n' "${p}"
    printf '%sDYNAMO_DISCOVERY_BACKEND=etcd\n' "${p}"
    printf '%sDYNAMO_ETCD_ENDPOINTS=%s\n' "${p}" "${endpoints}"
    printf '%sDYNAMO_NAMESPACE=default\n' "${p}"
    printf '%sDYNAMO_COMPONENT=backend\n' "${p}"
    printf '%sDYNAMO_ENDPOINT_NAME=generate\n' "${p}"
}
