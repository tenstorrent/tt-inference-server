// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstddef>

#include "runtime/worker/worker_metrics_shm.hpp"

namespace tt::worker::sp_pipeline {

/**
 * Scratch-area index convention for the sp_pipeline runner family
 * (tagged in shared memory as MetricsLayout::SP_PIPELINE_RUNNER).
 *
 * Both writer (worker-side runner via SingleProcessWorkerMetrics) and reader
 * (main-side SpPipelineWorkerMetricsRenderer) include this header so they
 * agree on what each scratch slot means.
 *
 * Indices are append-only. To remove a metric, leave the constant in place
 * and stop writing to it. Reordering or repurposing an index requires a
 * coordinated main+worker restart (which is guaranteed since both come from
 * the same binary).
 */

constexpr size_t SCRATCH_STEP_EPOCH_MS = 0;
constexpr size_t SCRATCH_LAST_OUTPUT_EPOCH_MS = 1;
constexpr size_t SCRATCH_ACTIVE_REQUESTS = 2;

// Cumulative event counters for the BlazeRunner slot state machine. These are
// monotonic since worker (re)start and let ops see whether the defer/supersede
// paths — otherwise invisible — actually fire in production.
constexpr size_t SCRATCH_EV_IDLE_TO_RUNNING = 3;
constexpr size_t SCRATCH_EV_RUNNING_TO_STOP_ACK = 4;
constexpr size_t SCRATCH_EV_DEFERRED_EVICT_REPLAYED = 5;
constexpr size_t SCRATCH_EV_DEFERRED_SUBMIT_LATCHED = 6;
constexpr size_t SCRATCH_EV_DEFERRED_SUBMIT_REPLAYED = 7;
constexpr size_t SCRATCH_EV_DEFERRED_SUBMIT_SUPERSEDED = 8;
constexpr size_t SCRATCH_EV_DEFERRED_EVICT_LATCHED = 9;
// ... up to index 31

static_assert(SCRATCH_EV_DEFERRED_EVICT_LATCHED < WORKER_SCRATCH_U64_COUNT,
              "sp_pipeline scratch indices exceed scratch capacity");

}  // namespace tt::worker::sp_pipeline
