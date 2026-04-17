// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <prometheus/registry.h>
#include <prometheus/text_serializer.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "worker/worker_metrics_renderer.hpp"
#include "worker/worker_metrics_shm.hpp"

namespace tt::worker {
class WorkerManager;
}

namespace tt::worker {

/**
 * Main-side collector for the per-worker shared-memory metrics.
 *
 * Responsibilities:
 *   - Owns a private prometheus::Registry that holds worker-side gauges.
 *   - On initialize() pre-registers gauges via the renderer for every worker
 *     so that label cardinality is steady and known at scrape time.
 *   - On refresh() (called from the /metrics handler) loads each slot's
 *     dispatch tag and forwards to the matching renderer, also passing the
 *     worker's process-level liveness flag from WorkerManager.
 *
 * Singleton because it must be reachable from MetricsController without
 * threading the wiring through Drogon's controller registration. main.cpp
 * is responsible for calling initialize() before any scrape can arrive.
 */
class WorkerMetricsAggregator {
 public:
  static WorkerMetricsAggregator& instance();

  /**
   * Wire the aggregator. Must be called once from main, after the shm
   * region has been created and the WorkerManager has been constructed
   * (workers may still be starting up; renderers tolerate empty slots).
   */
  void initialize(const WorkerMetricsShmRegion* region, WorkerManager* mgr,
                  size_t numWorkers);

  /** Register a renderer for a layout. May be called only between
   *  initialize() and the first refresh(). */
  void registerRenderer(MetricsLayout layout,
                        std::unique_ptr<IWorkerMetricsRenderer> renderer);

  /** Pre-build the gauges for every worker_id using the registered
   *  renderers. Called by main after all renderers have been registered. */
  void prebuildAll();

  /** Update gauges from the latest shm state. Safe to call concurrently
   *  with workers writing into the slots. */
  void refresh();

  /** Render the worker-side registry in Prometheus text format. */
  std::string renderText();

  /** True once initialize() has been called. */
  bool isInitialized() const { return initialized_; }

 private:
  WorkerMetricsAggregator() = default;

  IWorkerMetricsRenderer* rendererFor(MetricsLayout layout);

  bool initialized_{false};
  size_t numWorkers_{0};
  const WorkerMetricsShmRegion* region_{nullptr};
  WorkerManager* mgr_{nullptr};

  std::shared_ptr<prometheus::Registry> registry_;
  std::unordered_map<MetricsLayout, std::unique_ptr<IWorkerMetricsRenderer>>
      by_layout_;
  std::mutex refresh_mutex_;
};

}  // namespace tt::worker
