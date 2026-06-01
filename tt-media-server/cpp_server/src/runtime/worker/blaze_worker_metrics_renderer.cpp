// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runtime/worker/blaze_worker_metrics_renderer.hpp"

#include <chrono>
#include <iterator>
#include <string>

#include "runtime/worker/blaze_metrics_layout.hpp"

namespace tt::worker {

namespace {

// Maps each exposed event series (the `event` label) to its scratch index.
// Order must match WorkerGauges::events. Append-only, like the scratch layout.
struct BlazeEventDef {
  const char* label;
  size_t scratchIdx;
};

constexpr BlazeEventDef BLAZE_EVENTS[] = {
    {"idle_to_running", sp_pipeline::SCRATCH_EV_IDLE_TO_RUNNING},
    {"running_to_stop_ack", sp_pipeline::SCRATCH_EV_RUNNING_TO_STOP_ACK},
    {"deferred_evict_replayed", sp_pipeline::SCRATCH_EV_DEFERRED_EVICT_REPLAYED},
    {"deferred_submit_latched", sp_pipeline::SCRATCH_EV_DEFERRED_SUBMIT_LATCHED},
    {"deferred_submit_replayed",
     sp_pipeline::SCRATCH_EV_DEFERRED_SUBMIT_REPLAYED},
    {"deferred_submit_superseded",
     sp_pipeline::SCRATCH_EV_DEFERRED_SUBMIT_SUPERSEDED},
    {"deferred_evict_latched", sp_pipeline::SCRATCH_EV_DEFERRED_EVICT_LATCHED},
};

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
    events_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_blaze_events")
             .Help("Cumulative count of BlazeRunner slot state transitions and "
                   "defer-path events since the worker last (re)started")
             .Register(registry);
  }

  const std::string idStr = std::to_string(workerId);
  WorkerGauges g;
  g.alive = &alive_family_->Add({{"worker_id", idStr}});
  g.step_age = &step_age_family_->Add({{"worker_id", idStr}});
  g.output_age = &output_age_family_->Add({{"worker_id", idStr}});
  g.active_requests = &active_family_->Add({{"worker_id", idStr}});
  g.events.reserve(std::size(BLAZE_EVENTS));
  for (const auto& def : BLAZE_EVENTS) {
    g.events.push_back(
        &events_family_->Add({{"worker_id", idStr}, {"event", def.label}}));
  }
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

  for (size_t i = 0; i < g.events.size(); ++i) {
    g.events[i]->Set(
        static_cast<double>(shm.loadScratch(slot, BLAZE_EVENTS[i].scratchIdx)));
  }
}

}  // namespace tt::worker
