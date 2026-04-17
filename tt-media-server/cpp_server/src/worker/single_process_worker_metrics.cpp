// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "worker/single_process_worker_metrics.hpp"

#include <unistd.h>

#include <chrono>

#include "config/settings.hpp"
#include "utils/logger.hpp"
#include "worker/sp_pipeline_metrics_layout.hpp"
#include "worker/worker_metrics_shm.hpp"

namespace tt::worker {

SingleProcessWorkerMetrics& SingleProcessWorkerMetrics::instance() {
  static SingleProcessWorkerMetrics inst;
  return inst;
}

void SingleProcessWorkerMetrics::initialize(int workerId, MetricsLayout layout) {
  workerId_ = workerId;
  layout_ = layout;

  const std::string shmName = tt::config::workerMetricsShmName();
  WorkerMetricsShmRegion* region = openSharedRegion(shmName);
  if (region == nullptr) {
    TT_LOG_CRITICAL(
        "[SingleProcessWorkerMetrics] Worker {} failed to attach to shm '{}'; "
        "metrics disabled",
        workerId, shmName);
    slot_ = nullptr;
    return;
  }

  uint32_t numSlots = region->num_workers.load(std::memory_order_acquire);
  if (workerId < 0 || static_cast<uint32_t>(workerId) >= numSlots) {
    TT_LOG_CRITICAL(
        "[SingleProcessWorkerMetrics] Worker id {} out of range "
        "(num_workers={}); metrics disabled",
        workerId, numSlots);
    slot_ = nullptr;
    return;
  }

  slot_ = &region->slots[workerId];
  slot_->generation.fetch_add(1, std::memory_order_acq_rel);
  slot_->metrics_layout.store(static_cast<uint8_t>(layout),
                              std::memory_order_release);
  slot_->pid.store(static_cast<int32_t>(getpid()), std::memory_order_release);

  // Seed sp_pipeline timestamps so age starts at ~0 instead of since-epoch.
  if (layout == MetricsLayout::SP_PIPELINE_RUNNER) {
    auto now = nowMs();
    slot_->scratch[sp_pipeline::SCRATCH_STEP_EPOCH_MS].store(
        now, std::memory_order_relaxed);
    slot_->scratch[sp_pipeline::SCRATCH_LAST_OUTPUT_EPOCH_MS].store(
        now, std::memory_order_relaxed);
    slot_->scratch[sp_pipeline::SCRATCH_ACTIVE_REQUESTS].store(
        0, std::memory_order_relaxed);
  }

  TT_LOG_INFO(
      "[SingleProcessWorkerMetrics] Worker {} attached to shm '{}' slot {}, "
      "layout={}",
      workerId, shmName, workerId, static_cast<uint32_t>(layout));
}

uint64_t SingleProcessWorkerMetrics::nowMs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void SingleProcessWorkerMetrics::updateStepHeartbeat() {
  if (slot_ == nullptr) return;
  slot_->scratch[sp_pipeline::SCRATCH_STEP_EPOCH_MS].store(
      nowMs(), std::memory_order_relaxed);
}

void SingleProcessWorkerMetrics::updateOutputHeartbeat() {
  if (slot_ == nullptr) return;
  slot_->scratch[sp_pipeline::SCRATCH_LAST_OUTPUT_EPOCH_MS].store(
      nowMs(), std::memory_order_relaxed);
}

void SingleProcessWorkerMetrics::incrementActiveRequests() {
  if (slot_ == nullptr) return;
  uint64_t prev = slot_->scratch[sp_pipeline::SCRATCH_ACTIVE_REQUESTS]
                      .fetch_add(1, std::memory_order_relaxed);
  if (prev == 0) {
    slot_->scratch[sp_pipeline::SCRATCH_LAST_OUTPUT_EPOCH_MS].store(
        nowMs(), std::memory_order_relaxed);
  }
}

void SingleProcessWorkerMetrics::decrementActiveRequests() {
  if (slot_ == nullptr) return;
  slot_->scratch[sp_pipeline::SCRATCH_ACTIVE_REQUESTS].fetch_sub(
      1, std::memory_order_relaxed);
}

void SingleProcessWorkerMetrics::scratchStoreU64(size_t idx, uint64_t value) {
  if (slot_ == nullptr || idx >= WORKER_SCRATCH_U64_COUNT) return;
  slot_->scratch[idx].store(value, std::memory_order_relaxed);
}

void SingleProcessWorkerMetrics::scratchAddU64(size_t idx, uint64_t delta) {
  if (slot_ == nullptr || idx >= WORKER_SCRATCH_U64_COUNT) return;
  slot_->scratch[idx].fetch_add(delta, std::memory_order_relaxed);
}

}  // namespace tt::worker
