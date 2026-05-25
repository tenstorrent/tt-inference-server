#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Capture an on-CPU profile from a running tt_media_server_cpp process.
# Produces two files per process:
#   - <name>.folded  → drag-drop into https://www.speedscope.app/ for the
#                      best interactive UI (real search, sandwich view,
#                      time-order view). This is the preferred way to view.
#   - <name>.svg     → quick-look flamegraph that opens in any browser with
#                      no internet needed; easy to attach to a PR comment.
#
# Usage:
#   ./flamegraph-capture.sh [TARGET] [SECONDS]
#     TARGET   "main" | "worker" | "all" | <pid>   (default: all)
#     SECONDS  capture duration (default: 30)
#
# Examples:
#   ./flamegraph-capture.sh                  # main + each worker, 30s each
#   ./flamegraph-capture.sh main 60          # main only, 60s
#   ./flamegraph-capture.sh 12345 20         # specific PID for 20s
#
# Output dir: ./bench_results/flamegraph_<timestamp>/  (.folded + .svg per name)
#
# Requirements (one-time):
#   sudo apt install linux-tools-$(uname -r) linux-tools-generic
#   # In Docker, perf_event_paranoid is usually locked; the script runs perf
#   # via sudo. Outside Docker, you can drop sudo by setting once:
#   #   sudo sysctl -w kernel.perf_event_paranoid=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-all}"
DURATION="${2:-30}"

OUTPUT_ROOT="${SCRIPT_DIR}/bench_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_ROOT}/flamegraph_${TIMESTAMP}"
FLAMEGRAPH_DIR="${SCRIPT_DIR}/build/_deps/flamegraph"

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
if ! command -v perf >/dev/null 2>&1; then
    echo "ERROR: perf not found. Install it with:"
    echo "  sudo apt install linux-tools-\$(uname -r) linux-tools-generic"
    exit 1
fi

if [ ! -d "${FLAMEGRAPH_DIR}" ]; then
    echo "Cloning Brendan Gregg's FlameGraph scripts to ${FLAMEGRAPH_DIR}..."
    mkdir -p "$(dirname "${FLAMEGRAPH_DIR}")"
    git clone --depth 1 https://github.com/brendangregg/FlameGraph "${FLAMEGRAPH_DIR}" >/dev/null 2>&1
fi

# ---------------------------------------------------------------------------
# Resolve target PIDs
# ---------------------------------------------------------------------------
resolve_main_pid() {
    # Drogon parent: the tt_media_server_cpp process NOT spawned with --worker.
    pgrep -af tt_media_server_cpp \
        | grep -v -- '--worker' \
        | grep -v flamegraph-capture \
        | awk '{print $1}' \
        | head -1
}

resolve_worker_pids() {
    pgrep -af 'tt_media_server_cpp.*--worker' \
        | awk '{print $1}'
}

declare -a TARGETS=()  # array of "name:pid"

case "${TARGET}" in
    main)
        pid="$(resolve_main_pid || true)"
        [ -z "${pid:-}" ] && { echo "ERROR: no main tt_media_server_cpp process found."; exit 1; }
        TARGETS+=("main:${pid}")
        ;;
    worker|workers)
        mapfile -t worker_pids < <(resolve_worker_pids)
        [ "${#worker_pids[@]}" -eq 0 ] && { echo "ERROR: no worker tt_media_server_cpp process found."; exit 1; }
        i=0
        for pid in "${worker_pids[@]}"; do
            TARGETS+=("worker${i}:${pid}")
            i=$((i+1))
        done
        ;;
    all)
        main_pid="$(resolve_main_pid || true)"
        [ -n "${main_pid:-}" ] && TARGETS+=("main:${main_pid}")
        mapfile -t worker_pids < <(resolve_worker_pids)
        i=0
        for pid in "${worker_pids[@]}"; do
            TARGETS+=("worker${i}:${pid}")
            i=$((i+1))
        done
        [ "${#TARGETS[@]}" -eq 0 ] && { echo "ERROR: no tt_media_server_cpp processes found."; exit 1; }
        ;;
    ''|*[!0-9]*)
        echo "ERROR: TARGET must be 'main', 'worker', 'all', or a numeric PID; got: '${TARGET}'"
        exit 1
        ;;
    *)  # numeric pid
        [ -d "/proc/${TARGET}" ] || { echo "ERROR: PID ${TARGET} does not exist."; exit 1; }
        TARGETS+=("pid${TARGET}:${TARGET}")
        ;;
esac

mkdir -p "${OUTPUT_DIR}"

echo "Capture target(s): ${TARGETS[*]}"
echo "Duration:          ${DURATION}s per process (captured in parallel)"
echo "Output:            ${OUTPUT_DIR}"
echo

# ---------------------------------------------------------------------------
# Capture in parallel
# ---------------------------------------------------------------------------
declare -a PERF_PIDS=()
for entry in "${TARGETS[@]}"; do
    name="${entry%%:*}"
    pid="${entry##*:}"
    data="${OUTPUT_DIR}/${name}.data"
    log="${OUTPUT_DIR}/${name}.perf.log"
    echo "  [${name}] perf record -p ${pid} -F 99 --call-graph dwarf -o ${data} (${DURATION}s)"
    sudo perf record -F 99 --call-graph dwarf -o "${data}" -p "${pid}" -- sleep "${DURATION}" \
        >"${log}" 2>&1 &
    PERF_PIDS+=("$!")
done

# Wait for all perf recorders to finish.
for p in "${PERF_PIDS[@]}"; do
    wait "${p}" || { echo "ERROR: perf record (job ${p}) failed; see *.perf.log in ${OUTPUT_DIR}"; exit 1; }
done

echo
echo "perf record done. Rendering flamegraph(s)..."

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
for entry in "${TARGETS[@]}"; do
    name="${entry%%:*}"
    pid="${entry##*:}"
    data="${OUTPUT_DIR}/${name}.data"
    script="${OUTPUT_DIR}/${name}.script"
    folded="${OUTPUT_DIR}/${name}.folded"
    svg="${OUTPUT_DIR}/${name}.svg"

    sudo chown "$(id -u):$(id -g)" "${data}"
    perf script -i "${data}" >"${script}" 2>/dev/null
    # Fold the long [[kernel.kallsyms]] chains (caused by container's kptr_restrict)
    # into a single [kernel] frame so the SVG is readable.
    "${FLAMEGRAPH_DIR}/stackcollapse-perf.pl" "${script}" \
        | sed 's/\(;\[\[kernel\.kallsyms\]\]\)\+/;[kernel]/g' \
        > "${folded}"
    "${FLAMEGRAPH_DIR}/flamegraph.pl" \
        --title "tt_media_server ${name} (pid ${pid})" \
        --subtitle "perf -F 99 --call-graph dwarf, ${DURATION}s @ ${TIMESTAMP}" \
        "${folded}" >"${svg}"
    samples=$(wc -l <"${script}")
    stacks=$(wc -l <"${folded}")
    echo "  [${name}] ${svg}  (samples=${samples}, unique stacks=${stacks})"
done

echo
echo "Preferred: open https://www.speedscope.app/ and drag-drop one of these"
echo ".folded files (real search, sandwich view, time-order view):"
for entry in "${TARGETS[@]}"; do
    name="${entry%%:*}"
    echo "  ${OUTPUT_DIR}/${name}.folded"
done
echo
echo "Quick look: open the .svg files in any browser. Wider = more CPU time."
echo "Click any frame to zoom; the search box (top-right) highlights symbols."
