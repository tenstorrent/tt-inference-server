// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "worker/sp_pipeline_worker_metrics_renderer.hpp"

#include <chrono>
#include <string>

#include "worker/sp_pipeline_metrics_layout.hpp"

namespace tt::worker {

namespace {

uint64_t nowMs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

double ageSeconds(uint64_t lastEpochMs, uint64_t nowEpochMs) {
  if (lastEpochMs == 0 || lastEpochMs > nowEpochMs) return 0.0;
  return static_cast<double>(nowEpochMs - lastEpochMs) / 1000.0;
}

}  // namespace

void SpPipelineWorkerMetricsRenderer::prebuildGauges(
    prometheus::Registry& registry, int workerId) {
  if (alive_family_ == nullptr) {
    alive_family_ = &prometheus::BuildGauge()
                         .Name("tt_worker_alive")
                         .Help("1 while the worker process is running")
                         .Register(registry);
    step_age_family_ = &prometheus::BuildGauge()
                            .Name("tt_worker_heartbeat_age_seconds")
                            .Help("Seconds since the worker last called step()")
                            .Register(registry);
    output_age_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_last_output_age_seconds")
             .Help("Seconds since the worker last produced a token")
             .Register(registry);
    active_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_active_requests")
             .Help("Number of requests currently in the worker pipeline")
             .Register(registry);
  }

  const std::string idStr = std::to_string(workerId);
  WorkerGauges g;
  g.alive = &alive_family_->Add({{"worker_id", idStr}});
  g.step_age = &step_age_family_->Add({{"worker_id", idStr}});
  g.output_age = &output_age_family_->Add({{"worker_id", idStr}});
  g.active_requests = &active_family_->Add({{"worker_id", idStr}});
  gauges_[workerId] = g;
}

void SpPipelineWorkerMetricsRenderer::render(const WorkerMetricsShm& shm,
                                             int workerId, bool isAlive) {
  auto it = gauges_.find(workerId);
  if (it == gauges_.end()) return;
  WorkerGauges& g = it->second;

  const size_t slot = static_cast<size_t>(workerId);
  uint64_t now = nowMs();
  uint64_t stepMs = shm.loadScratch(slot, sp_pipeline::SCRATCH_STEP_EPOCH_MS);
  uint64_t outputMs =
      shm.loadScratch(slot, sp_pipeline::SCRATCH_LAST_OUTPUT_EPOCH_MS);
  uint64_t active = shm.loadScratch(slot, sp_pipeline::SCRATCH_ACTIVE_REQUESTS);

  g.alive->Set(isAlive ? 1.0 : 0.0);
  g.step_age->Set(ageSeconds(stepMs, now));
  g.output_age->Set(ageSeconds(outputMs, now));
  g.active_requests->Set(static_cast<double>(active));
}

}  // namespace tt::worker
