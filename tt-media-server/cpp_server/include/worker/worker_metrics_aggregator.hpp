// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <prometheus/registry.h>
#include <prometheus/text_serializer.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

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
 *   - On prebuildAll() resolves each worker's renderer from its configured
 *     MetricsLayout (passed in via initialize) and pre-registers that
 *     renderer's gauges, so label cardinality is steady at scrape time.
 *   - On refresh() (called from the /metrics handler) walks the cached
 *     per-worker renderer vector and forwards each slot + its is_alive flag
 *     (from WorkerManager) to its renderer. No per-scrape layout lookup.
 *
 * Why the layout is fixed at initialize() instead of read from the slot on
 * every scrape: in a given deployment main already knows which runner each
 * worker is about to run, and that assignment cannot change at runtime
 * (workers come from the same binary, their layout is a compile/config-time
 * property). Resolving renderers once avoids an unordered_map lookup per
 * worker per scrape and makes the dispatch static data.
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
   *
   * layout_by_worker[i] is the MetricsLayout that worker i will write
   * into its slot. Size must equal numWorkers.
   */
  void initialize(const WorkerMetricsShmRegion* region, WorkerManager* mgr,
                  std::vector<MetricsLayout> layout_by_worker);

  /** Register a renderer for a layout. May be called only between
   *  initialize() and prebuildAll(). */
  void registerRenderer(MetricsLayout layout,
                        std::unique_ptr<IWorkerMetricsRenderer> renderer);

  /** Resolve per-worker renderer pointers from the layout vector passed to
   *  initialize() and pre-build each renderer's gauges. Must be called
   *  after every registerRenderer() call. */
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
  const WorkerMetricsShmRegion* region_{nullptr};
  WorkerManager* mgr_{nullptr};

  std::vector<MetricsLayout> layout_by_worker_;
  std::vector<IWorkerMetricsRenderer*> renderer_by_worker_;
  bool layout_tags_verified_{false};

  std::shared_ptr<prometheus::Registry> registry_;
  std::unordered_map<MetricsLayout, std::unique_ptr<IWorkerMetricsRenderer>>
      by_layout_;
  std::mutex refresh_mutex_;
};

}  // namespace tt::worker
