// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstddef>

#include "worker/worker_metrics_shm.hpp"

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
// Cumulative aggregates across all turns / slots in this worker.
constexpr size_t SCRATCH_TOTAL_PROMPT_TOKENS = 3;
constexpr size_t SCRATCH_TOTAL_GENERATION_TOKENS = 4;
constexpr size_t SCRATCH_TOTAL_SPEC_ACCEPTS = 5;
constexpr size_t SCRATCH_TOTAL_SPEC_REJECTS = 6;
// Indices 7..31 reserved for future aggregates.

// ---------------------------------------------------------------------------
//  Per-LLM-slot region (starts at index LLM_SLOT_BASE).
//
//  Layout:  scratch[LLM_SLOT_BASE + slot_id * LLM_SLOT_FIELDS + field]
//
//  Fields are written by the runner on its step thread (single writer per
//  worker) and read by the main process at scrape time. All accesses use
//  relaxed atomics — visibility across the scrape-to-step boundary is
//  acceptable to be a few ms stale.
// ---------------------------------------------------------------------------
constexpr size_t LLM_SLOT_BASE = 32;
constexpr size_t MAX_LLM_SLOTS = 128;  // matches defaults::PM_MAX_USERS
constexpr size_t LLM_SLOT_FIELDS = 7;

constexpr size_t LLM_FIELD_LAST_INPUT_TOKENS = 0;       // ISL of last submitted turn
constexpr size_t LLM_FIELD_CURRENT_OUTPUT_TOKENS = 1;   // tokens emitted so far in current turn
constexpr size_t LLM_FIELD_LAST_OUTPUT_TOKENS = 2;      // OSL of last completed turn
constexpr size_t LLM_FIELD_TURN_START_EPOCH_MS = 3;
constexpr size_t LLM_FIELD_FIRST_TOKEN_EPOCH_MS = 4;    // 0 until first decode token of current turn
constexpr size_t LLM_FIELD_LAST_TPOT_US = 5;            // (last_token_ms - first_token_ms) * 1000 / (osl - 1)
constexpr size_t LLM_FIELD_LAST_ACCEPTANCE_RATE_BPS = 6;  // basis points 0-10000

inline size_t llmSlotIdx(uint32_t slotId, size_t field) {
  return LLM_SLOT_BASE + static_cast<size_t>(slotId) * LLM_SLOT_FIELDS + field;
}

static_assert(SCRATCH_TOTAL_SPEC_REJECTS < LLM_SLOT_BASE,
              "sp_pipeline aggregate indices overlap per-slot region");
static_assert(LLM_SLOT_BASE + MAX_LLM_SLOTS * LLM_SLOT_FIELDS <=
                  WORKER_SCRATCH_U64_COUNT,
              "sp_pipeline per-slot region exceeds scratch capacity");

}  // namespace tt::worker::sp_pipeline
