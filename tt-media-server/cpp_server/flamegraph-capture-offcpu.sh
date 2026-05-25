#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Capture an OFF-CPU flamegraph: shows where threads are blocking/waiting,
# not where they're burning CPU. The on-CPU flamegraph
# (flamegraph-capture.sh) answers "what is consuming CPU?"; this one answers
# "what is the polling loop *waiting* on?".
#
# Mechanism: perf records a stack on every context switch (-e cs). Stacks
# whose tail is pthread_mutex_lock or pthread_cond_wait mean the thread
# blocked there. Stacks ending in random code mean the thread was preempted
# (less interesting).
#
# Note: the y-axis is COUNT of off-CPU events, not duration. For true
# duration-weighted off-CPU you need BPF (bcc/bpftrace), which is not
# available inside this container. Counts are still highly actionable for
# finding lock contention and missed wakeups.
#
# Usage:
#   ./flamegraph-capture-offcpu.sh [TARGET] [SECONDS]
#     TARGET   "main" | "worker" | "all" | <pid>   (default: all)
#     SECONDS  capture duration (default: 30)
#
# Output dir: ./bench_results/flamegraph_offcpu_<timestamp>/
# Per process, you get both:
#   - <name>.folded  → drag-drop into https://www.speedscope.app/ for the
#                      best interactive UI. Preferred.
#   - <name>.svg     → quick-look flamegraph, opens in any browser.
#
# See flamegraph-capture.sh for requirements (perf, FlameGraph repo).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-all}"
DURATION="${2:-30}"

OUTPUT_ROOT="${SCRIPT_DIR}/bench_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_ROOT}/flamegraph_offcpu_${TIMESTAMP}"
FLAMEGRAPH_DIR="${SCRIPT_DIR}/build/_deps/flamegraph"

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

# Filter by /proc/<pid>/exe (not pgrep -af cmdline) — otherwise a parent
# shell or load-generator with the binary name in its argv can be picked as
# "main" and yield a zero-sample capture. See flamegraph-capture.sh for the
# longer explanation.
list_server_pids() {
    local pid exe
    for pid in $(pgrep -f tt_media_server_cpp 2>/dev/null); do
        exe="$(readlink "/proc/${pid}/exe" 2>/dev/null || true)"
        [[ "${exe}" == */tt_media_server_cpp ]] && echo "${pid}"
    done
}

is_worker_pid() {
    local pid="$1"
    tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -q -- '--worker'
}

resolve_main_pid() {
    local pid
    for pid in $(list_server_pids); do
        is_worker_pid "${pid}" || { echo "${pid}"; return; }
    done
}

resolve_worker_pids() {
    local pid
    for pid in $(list_server_pids); do
        is_worker_pid "${pid}" && echo "${pid}"
    done
}

declare -a TARGETS=()

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
    *)
        [ -d "/proc/${TARGET}" ] || { echo "ERROR: PID ${TARGET} does not exist."; exit 1; }
        TARGETS+=("pid${TARGET}:${TARGET}")
        ;;
esac

mkdir -p "${OUTPUT_DIR}"

echo "Off-CPU capture target(s): ${TARGETS[*]}"
echo "Duration:                  ${DURATION}s per process (parallel)"
echo "Output:                    ${OUTPUT_DIR}"
echo

declare -a PERF_PIDS=()
for entry in "${TARGETS[@]}"; do
    name="${entry%%:*}"
    pid="${entry##*:}"
    data="${OUTPUT_DIR}/${name}.data"
    log="${OUTPUT_DIR}/${name}.perf.log"
    echo "  [${name}] perf record -e cs -p ${pid} --call-graph dwarf -o ${data} (${DURATION}s)"
    sudo perf record -e cs --call-graph dwarf -o "${data}" -p "${pid}" -- sleep "${DURATION}" \
        >"${log}" 2>&1 &
    PERF_PIDS+=("$!")
done

for p in "${PERF_PIDS[@]}"; do
    wait "${p}" || { echo "ERROR: perf record (job ${p}) failed; see *.perf.log in ${OUTPUT_DIR}"; exit 1; }
done

echo
echo "perf record done. Rendering off-CPU flamegraph(s)..."

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
        --colors=io \
        --countname=switches \
        --title "tt_media_server ${name} (pid ${pid}) — OFF-CPU" \
        --subtitle "perf -e cs --call-graph dwarf, ${DURATION}s @ ${TIMESTAMP}  (wider = more context-switch events, e.g. lock waits)" \
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
echo "Quick look: open the .svg files in any browser. Wider = more off-CPU"
echo "events for that call path. Look for tails of pthread_mutex_lock or"
echo "pthread_cond_wait — those are real blocking points. The y-axis is event"
echo "COUNT, not duration."
