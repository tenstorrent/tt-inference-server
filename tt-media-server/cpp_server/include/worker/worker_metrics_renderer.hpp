// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <prometheus/registry.h>

#include "worker/worker_metrics_shm.hpp"

namespace tt::worker {

/**
 * Strategy interface for translating a single worker's shared-memory slot
 * into Prometheus series. One implementation per MetricsLayout.
 *
 * The aggregator on the main side calls prebuildGauges() once per worker_id
 * at startup so that every gauge is registered up-front (steady label
 * cardinality), then calls render() on every scrape.
 */
class IWorkerMetricsRenderer {
 public:
  virtual ~IWorkerMetricsRenderer() = default;

  IWorkerMetricsRenderer() = default;
  IWorkerMetricsRenderer(const IWorkerMetricsRenderer&) = delete;
  IWorkerMetricsRenderer& operator=(const IWorkerMetricsRenderer&) = delete;

  /** Register all gauges this renderer ever emits for the given worker. */
  virtual void prebuildGauges(prometheus::Registry& registry, int workerId) = 0;

  /** Update gauges from the shm slot. is_alive is sourced from
   *  WorkerManager (waitpid-based liveness). */
  virtual void render(const WorkerMetricsShm& shm, int workerId,
                      bool is_alive) = 0;
};

}  // namespace tt::worker
