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
    prompt_tokens_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_prompt_tokens_total")
             .Help("Cumulative prompt tokens submitted to this worker")
             .Register(registry);
    generation_tokens_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_generation_tokens_total")
             .Help("Cumulative generation tokens emitted by this worker")
             .Register(registry);
    spec_accepts_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_spec_accepts_total")
             .Help("Cumulative speculative-decode accepts on this worker")
             .Register(registry);
    spec_rejects_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_spec_rejects_total")
             .Help("Cumulative speculative-decode rejects on this worker")
             .Register(registry);
    total_acceptance_rate_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_total_acceptance_rate")
             .Help("Cumulative speculative-decode acceptance rate (0..1)")
             .Register(registry);
    total_tokens_processed_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_total_tokens_processed_total")
             .Help(
                 "Cumulative tokens this worker drove through the model: "
                 "prompt + spec_accepts + spec_rejects. Use rate() to derive "
                 "saturation throughput including draft tokens that were "
                 "verified but rejected. Cluster total: "
                 "sum(rate(tt_worker_total_tokens_processed_total[1m])).")
             .Register(registry);
    slot_input_tokens_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_slot_input_tokens")
             .Help("ISL of the last turn submitted to this LLM slot")
             .Register(registry);
    slot_output_tokens_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_slot_output_tokens")
             .Help("OSL of the last completed turn on this LLM slot")
             .Register(registry);
    slot_current_output_tokens_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_slot_current_output_tokens")
             .Help("Tokens emitted so far in the in-flight turn on this slot")
             .Register(registry);
    slot_tpot_seconds_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_slot_tpot_seconds")
             .Help("Time-per-output-token of the last completed turn (seconds)")
             .Register(registry);
    slot_acceptance_rate_family_ =
        &prometheus::BuildGauge()
             .Name("tt_worker_slot_acceptance_rate")
             .Help(
                 "Speculative-decode acceptance rate of the last completed "
                 "turn (0..1)")
             .Register(registry);
  }

  const std::string idStr = std::to_string(workerId);
  WorkerGauges g;
  g.alive = &alive_family_->Add({{"worker_id", idStr}});
  g.step_age = &step_age_family_->Add({{"worker_id", idStr}});
  g.output_age = &output_age_family_->Add({{"worker_id", idStr}});
  g.active_requests = &active_family_->Add({{"worker_id", idStr}});
  g.prompt_tokens_total = &prompt_tokens_family_->Add({{"worker_id", idStr}});
  g.generation_tokens_total =
      &generation_tokens_family_->Add({{"worker_id", idStr}});
  g.spec_accepts_total = &spec_accepts_family_->Add({{"worker_id", idStr}});
  g.spec_rejects_total = &spec_rejects_family_->Add({{"worker_id", idStr}});
  g.total_acceptance_rate =
      &total_acceptance_rate_family_->Add({{"worker_id", idStr}});
  g.total_tokens_processed_total =
      &total_tokens_processed_family_->Add({{"worker_id", idStr}});

  g.slots.resize(sp_pipeline::MAX_LLM_SLOTS);
  for (size_t s = 0; s < sp_pipeline::MAX_LLM_SLOTS; ++s) {
    const std::string slotStr = std::to_string(s);
    SlotGauges sg;
    sg.input_tokens = &slot_input_tokens_family_->Add(
        {{"worker_id", idStr}, {"slot_id", slotStr}});
    sg.output_tokens = &slot_output_tokens_family_->Add(
        {{"worker_id", idStr}, {"slot_id", slotStr}});
    sg.current_output_tokens = &slot_current_output_tokens_family_->Add(
        {{"worker_id", idStr}, {"slot_id", slotStr}});
    sg.tpot_seconds = &slot_tpot_seconds_family_->Add(
        {{"worker_id", idStr}, {"slot_id", slotStr}});
    sg.acceptance_rate = &slot_acceptance_rate_family_->Add(
        {{"worker_id", idStr}, {"slot_id", slotStr}});
    g.slots[s] = sg;
  }

  gauges_[workerId] = std::move(g);
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
  uint64_t promptTotal =
      shm.loadScratch(slot, sp_pipeline::SCRATCH_TOTAL_PROMPT_TOKENS);
  uint64_t genTotal =
      shm.loadScratch(slot, sp_pipeline::SCRATCH_TOTAL_GENERATION_TOKENS);
  uint64_t accTotal =
      shm.loadScratch(slot, sp_pipeline::SCRATCH_TOTAL_SPEC_ACCEPTS);
  uint64_t rejTotal =
      shm.loadScratch(slot, sp_pipeline::SCRATCH_TOTAL_SPEC_REJECTS);

  g.alive->Set(isAlive ? 1.0 : 0.0);
  g.step_age->Set(ageSeconds(stepMs, now));
  g.output_age->Set(ageSeconds(outputMs, now));
  g.active_requests->Set(static_cast<double>(active));
  g.prompt_tokens_total->Set(static_cast<double>(promptTotal));
  g.generation_tokens_total->Set(static_cast<double>(genTotal));
  g.spec_accepts_total->Set(static_cast<double>(accTotal));
  g.spec_rejects_total->Set(static_cast<double>(rejTotal));
  const uint64_t specTotal = accTotal + rejTotal;
  g.total_acceptance_rate->Set(
      specTotal > 0 ? static_cast<double>(accTotal) / specTotal : 0.0);
  // Saturation throughput series: includes prompt tokens the model prefilled,
  // accepted (emitted) decode tokens, and draft tokens that were verified but
  // rejected. rate() of this gauge over a worker, summed across the cluster,
  // is the "tokens the model actually had to compute per second".
  g.total_tokens_processed_total->Set(
      static_cast<double>(promptTotal + accTotal + rejTotal));

  for (size_t s = 0; s < g.slots.size(); ++s) {
    const uint64_t isl = shm.loadScratch(
        slot,
        sp_pipeline::llmSlotIdx(static_cast<uint32_t>(s),
                                sp_pipeline::LLM_FIELD_LAST_INPUT_TOKENS));
    const uint64_t osl = shm.loadScratch(
        slot,
        sp_pipeline::llmSlotIdx(static_cast<uint32_t>(s),
                                sp_pipeline::LLM_FIELD_LAST_OUTPUT_TOKENS));
    const uint64_t curOsl = shm.loadScratch(
        slot,
        sp_pipeline::llmSlotIdx(static_cast<uint32_t>(s),
                                sp_pipeline::LLM_FIELD_CURRENT_OUTPUT_TOKENS));
    const uint64_t tpotUs = shm.loadScratch(
        slot, sp_pipeline::llmSlotIdx(static_cast<uint32_t>(s),
                                      sp_pipeline::LLM_FIELD_LAST_TPOT_US));
    const uint64_t bps = shm.loadScratch(
        slot, sp_pipeline::llmSlotIdx(
                  static_cast<uint32_t>(s),
                  sp_pipeline::LLM_FIELD_LAST_ACCEPTANCE_RATE_BPS));

    SlotGauges& sg = g.slots[s];
    sg.input_tokens->Set(static_cast<double>(isl));
    sg.output_tokens->Set(static_cast<double>(osl));
    sg.current_output_tokens->Set(static_cast<double>(curOsl));
    sg.tpot_seconds->Set(static_cast<double>(tpotUs) / 1.0e6);
    sg.acceptance_rate->Set(static_cast<double>(bps) / 10000.0);
  }
}

}  // namespace tt::worker
