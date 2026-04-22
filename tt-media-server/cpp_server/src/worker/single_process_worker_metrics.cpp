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

void SingleProcessWorkerMetrics::initialize(int workerId,
                                            MetricsLayout layout) {
  workerId_ = workerId;
  layout_ = layout;

  const std::string shmName = tt::config::workerMetricsShmName();
  shm_ = WorkerMetricsShm::open(shmName);
  if (shm_ == nullptr) {
    TT_LOG_CRITICAL(
        "[SingleProcessWorkerMetrics] Worker {} failed to attach to shm '{}'; "
        "metrics disabled",
        workerId, shmName);
    return;
  }

  size_t numSlots = shm_->numWorkers();
  if (workerId < 0 || static_cast<size_t>(workerId) >= numSlots) {
    TT_LOG_CRITICAL(
        "[SingleProcessWorkerMetrics] Worker id {} out of range "
        "(num_workers={}); metrics disabled",
        workerId, numSlots);
    shm_.reset();
    return;
  }

  shm_->bumpGeneration(workerId);
  shm_->setLayout(workerId, layout);
  shm_->setPid(workerId, static_cast<int32_t>(getpid()));

  // Seed sp_pipeline timestamps so age starts at ~0 instead of since-epoch.
  if (layout == MetricsLayout::SP_PIPELINE_RUNNER) {
    auto now = nowMs();
    shm_->storeScratch(workerId, sp_pipeline::SCRATCH_STEP_EPOCH_MS, now);
    shm_->storeScratch(workerId, sp_pipeline::SCRATCH_LAST_OUTPUT_EPOCH_MS,
                       now);
    shm_->storeScratch(workerId, sp_pipeline::SCRATCH_ACTIVE_REQUESTS, 0);
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
  if (shm_ == nullptr) return;
  shm_->storeScratch(workerId_, sp_pipeline::SCRATCH_STEP_EPOCH_MS, nowMs());
}

void SingleProcessWorkerMetrics::updateOutputHeartbeat() {
  if (shm_ == nullptr) return;
  shm_->storeScratch(workerId_, sp_pipeline::SCRATCH_LAST_OUTPUT_EPOCH_MS,
                     nowMs());
}

void SingleProcessWorkerMetrics::incrementActiveRequests() {
  if (shm_ == nullptr) return;
  uint64_t prev =
      shm_->fetchAddScratch(workerId_, sp_pipeline::SCRATCH_ACTIVE_REQUESTS, 1);
  if (prev == 0) {
    shm_->storeScratch(workerId_, sp_pipeline::SCRATCH_LAST_OUTPUT_EPOCH_MS,
                       nowMs());
  }
}

void SingleProcessWorkerMetrics::decrementActiveRequests() {
  if (shm_ == nullptr) return;
  shm_->fetchSubScratch(workerId_, sp_pipeline::SCRATCH_ACTIVE_REQUESTS, 1);
}

void SingleProcessWorkerMetrics::scratchStoreU64(size_t idx, uint64_t value) {
  if (shm_ == nullptr || idx >= WORKER_SCRATCH_U64_COUNT) return;
  shm_->storeScratch(workerId_, idx, value);
}

void SingleProcessWorkerMetrics::scratchAddU64(size_t idx, uint64_t delta) {
  if (shm_ == nullptr || idx >= WORKER_SCRATCH_U64_COUNT) return;
  shm_->fetchAddScratch(workerId_, idx, delta);
}

}  // namespace tt::worker
