// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <prometheus/gauge.h>
#include <prometheus/registry.h>

#include <unordered_map>

#include "worker/worker_metrics_renderer.hpp"
#include "worker/worker_metrics_shm.hpp"

namespace tt::worker {

/**
 * Renderer for slots tagged MetricsLayout::SP_PIPELINE_RUNNER (currently
 * produced by SpPipelineRunner / SpPipelineRunnerDemo). Translates the
 * sp_pipeline scratch indices into the externally-visible Prometheus series:
 *   - tt_worker_alive
 *   - tt_worker_heartbeat_age_seconds
 *   - tt_worker_last_output_age_seconds
 *   - tt_worker_active_requests
 *
 * Class is named after the layout it reads (sp_pipeline), not after the
 * runner that writes it, so a future second runner producing the same
 * MetricsLayout::SP_PIPELINE_RUNNER can reuse this renderer unchanged.
 */
class SpPipelineWorkerMetricsRenderer : public IWorkerMetricsRenderer {
 public:
  void prebuildGauges(prometheus::Registry& registry, int workerId) override;
  void render(const WorkerMetricsShm& shm, int workerId,
              bool is_alive) override;

 private:
  struct WorkerGauges {
    prometheus::Gauge* alive{nullptr};
    prometheus::Gauge* step_age{nullptr};
    prometheus::Gauge* output_age{nullptr};
    prometheus::Gauge* active_requests{nullptr};
  };

  prometheus::Family<prometheus::Gauge>* alive_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* step_age_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* output_age_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* active_family_{nullptr};

  std::unordered_map<int, WorkerGauges> gauges_;
};

}  // namespace tt::worker
