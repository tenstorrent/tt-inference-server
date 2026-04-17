// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstddef>

#include "worker/worker_metrics_shm.hpp"

namespace tt::worker::sp_pipeline {

/**
 * Scratch-area index convention for the sp_pipeline runner family
 * (tagged in shared memory as MetricsLayout::LLM).
 *
 * Both writer (worker-side runner via WorkerMetrics) and reader
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
// reserved for future sp_pipeline metrics:
// constexpr size_t SCRATCH_KV_CACHE_BPS    = 3;  // basis points 0-10000
// constexpr size_t SCRATCH_NUM_DECODING    = 4;
// ... up to index 31

static_assert(SCRATCH_ACTIVE_REQUESTS < WORKER_SCRATCH_U64_COUNT,
              "sp_pipeline scratch indices exceed scratch capacity");

}  // namespace tt::worker::sp_pipeline
