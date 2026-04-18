// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "worker/worker_metrics_shm.hpp"

namespace tt::worker {

/**
 * Per-worker metrics writer for single-process workers, backed by a POSIX
 * shared-memory slot.
 *
 * On initialize() the worker attaches to the segment created by main and
 * claims slots[workerId], stamping its pid and metrics_layout. The hot path
 * is then a single relaxed atomic store into a layout-defined scratch cell;
 * no syscalls, no allocations, no locks.
 *
 * Convenience methods (updateStepHeartbeat etc.) are valid only when the
 * worker was initialized with MetricsLayout::SP_PIPELINE_RUNNER and target the
 * indices declared in sp_pipeline_metrics_layout.hpp. For other layouts, use
 * the scratchStoreU64 / scratchAddU64 primitives with that layout's own index
 * constants.
 */
class SingleProcessWorkerMetrics {
 public:
  static SingleProcessWorkerMetrics& instance();

  /**
   * Attach to the shared region (name from settings::workerMetricsShmName())
   * and claim the slot for the given worker id. Stamps the current pid and
   * the metrics_layout tag so the main-side aggregator can dispatch the
   * slot to the right renderer.
   */
  void initialize(int workerId, MetricsLayout layout);

  // ----- sp_pipeline (MetricsLayout::SP_PIPELINE_RUNNER) convenience writers
  // ---------------
  void updateStepHeartbeat();
  void updateOutputHeartbeat();
  void incrementActiveRequests();
  void decrementActiveRequests();

  // ----- low-level layout-agnostic writers ----------------------------------
  void scratchStoreU64(size_t idx, uint64_t value);
  void scratchAddU64(size_t idx, uint64_t delta);

 private:
  SingleProcessWorkerMetrics() = default;

  static uint64_t nowMs();

  int workerId_{0};
  MetricsLayout layout_{MetricsLayout::UNKNOWN};
  std::unique_ptr<WorkerMetricsShm> shm_;
};

}  // namespace tt::worker
