// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <prometheus/gauge.h>
#include <prometheus/registry.h>

#include <unordered_map>
#include <vector>

#include "worker/worker_metrics_renderer.hpp"
#include "worker/worker_metrics_shm.hpp"

namespace tt::worker {

/**
 * Renderer for slots tagged MetricsLayout::SP_PIPELINE_RUNNER (currently
 * produced by SpPipelineRunner / SpPipelineRunnerDemo). Translates the
 * sp_pipeline scratch indices into the externally-visible Prometheus series:
 *
 *   Worker-level (label: worker_id):
 *     - tt_worker_alive
 *     - tt_worker_heartbeat_age_seconds
 *     - tt_worker_last_output_age_seconds
 *     - tt_worker_active_requests
 *     - tt_worker_prompt_tokens_total
 *     - tt_worker_generation_tokens_total
 *     - tt_worker_spec_accepts_total
 *     - tt_worker_spec_rejects_total
 *     - tt_worker_total_acceptance_rate
 *     - tt_worker_total_tokens_processed_total
 *           (prompt + spec_accepts + spec_rejects; saturation throughput
 *           series — includes draft tokens that were verified but rejected)
 *
 *   Per-LLM-slot (labels: worker_id, slot_id) — wired so that
 *   "TPOT vs ISL/OSL" correlation can be plotted in Grafana:
 *     - tt_worker_slot_input_tokens          (last submitted turn's ISL)
 *     - tt_worker_slot_output_tokens         (last completed turn's OSL)
 *     - tt_worker_slot_current_output_tokens (in-flight turn's emitted tokens)
 *     - tt_worker_slot_tpot_seconds          (last completed turn's TPOT)
 *     - tt_worker_slot_acceptance_rate       (last completed turn, 0..1)
 *
 * Class is named after the layout it reads (sp_pipeline), not after the
 * runner that writes it, so a future second runner producing the same
 * MetricsLayout::SP_PIPELINE_RUNNER can reuse this renderer unchanged.
 */
class SpPipelineWorkerMetricsRenderer : public IWorkerMetricsRenderer {
 public:
  void prebuildGauges(prometheus::Registry& registry, int workerId) override;
  void render(const WorkerMetricsShm& shm, int workerId, bool isAlive) override;

 private:
  struct SlotGauges {
    prometheus::Gauge* input_tokens{nullptr};
    prometheus::Gauge* output_tokens{nullptr};
    prometheus::Gauge* current_output_tokens{nullptr};
    prometheus::Gauge* tpot_seconds{nullptr};
    prometheus::Gauge* acceptance_rate{nullptr};
  };

  struct WorkerGauges {
    prometheus::Gauge* alive{nullptr};
    prometheus::Gauge* step_age{nullptr};
    prometheus::Gauge* output_age{nullptr};
    prometheus::Gauge* active_requests{nullptr};
    prometheus::Gauge* prompt_tokens_total{nullptr};
    prometheus::Gauge* generation_tokens_total{nullptr};
    prometheus::Gauge* spec_accepts_total{nullptr};
    prometheus::Gauge* spec_rejects_total{nullptr};
    prometheus::Gauge* total_acceptance_rate{nullptr};
    prometheus::Gauge* total_tokens_processed_total{nullptr};
    std::vector<SlotGauges> slots;
  };

  prometheus::Family<prometheus::Gauge>* alive_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* step_age_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* output_age_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* active_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* prompt_tokens_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* generation_tokens_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* spec_accepts_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* spec_rejects_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* total_acceptance_rate_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* total_tokens_processed_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* slot_input_tokens_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* slot_output_tokens_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* slot_current_output_tokens_family_{
      nullptr};
  prometheus::Family<prometheus::Gauge>* slot_tpot_seconds_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* slot_acceptance_rate_family_{nullptr};

  std::unordered_map<int, WorkerGauges> gauges_;
};

}  // namespace tt::worker
