// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runtime/worker/tts_worker_metrics_renderer.hpp"

#include <chrono>
#include <string>

#include "config/settings.hpp"
#include "runtime/worker/tts_metrics_layout.hpp"

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

void TtsWorkerMetricsRenderer::prebuildGauges(prometheus::Registry& registry,
                                              int workerId) {
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
             .Help("Seconds since the worker last emitted a codec token")
             .Register(registry);
    codec_tokens_family_ =
        &prometheus::BuildGauge()
             .Name("tt_tts_codec_tokens_total")
             .Help(
                 "Cumulative acoustic/codec tokens emitted by the TTS decoder "
                 "on this worker since it last (re)started. Codec-token "
                 "throughput is rate(tt_tts_codec_tokens_total[...]) — the "
                 "autoregressive decode capacity that has to stay ahead of "
                 "playback. Labelled by worker_id, its DEVICE_IDS group "
                 "(device), model_name, and how the voice was specified "
                 "(voice_source).")
             .Register(registry);
  }

  const std::string idStr = std::to_string(workerId);
  // DEVICE_IDS group for this worker, e.g. "0,1,2,3" — its TT_VISIBLE_DEVICES
  // value, which is the per-device identity of the worker process.
  const std::string device =
      tt::config::visibleDevicesForWorker(static_cast<size_t>(workerId));
  const std::string modelName = tt::config::runnerType();

  WorkerGauges g;
  g.alive = &alive_family_->Add({{"worker_id", idStr}});
  g.step_age = &step_age_family_->Add({{"worker_id", idStr}});
  g.output_age = &output_age_family_->Add({{"worker_id", idStr}});
  for (size_t i = 0; i < tts::VOICE_SOURCE_COUNT; ++i) {
    g.codec_tokens[i] = &codec_tokens_family_->Add(
        {{"worker_id", idStr},
         {"device", device},
         {"model_name", modelName},
         {"voice_source",
          tts::voiceSourceLabel(static_cast<tts::VoiceSource>(i))}});
  }
  gauges_[workerId] = g;
}

void TtsWorkerMetricsRenderer::render(const WorkerMetricsShm& shm, int workerId,
                                      bool isAlive) {
  auto it = gauges_.find(workerId);
  if (it == gauges_.end()) return;
  WorkerGauges& g = it->second;

  const size_t slot = static_cast<size_t>(workerId);
  const uint64_t now = nowMs();
  const uint64_t stepMs = shm.loadScratch(slot, tts::SCRATCH_STEP_EPOCH_MS);
  const uint64_t outputMs =
      shm.loadScratch(slot, tts::SCRATCH_LAST_OUTPUT_EPOCH_MS);

  g.alive->Set(isAlive ? 1.0 : 0.0);
  g.step_age->Set(ageSeconds(stepMs, now));
  g.output_age->Set(ageSeconds(outputMs, now));

  for (size_t i = 0; i < tts::VOICE_SOURCE_COUNT; ++i) {
    const uint64_t tokens = shm.loadScratch(
        slot, tts::codecTokensIdx(static_cast<tts::VoiceSource>(i)));
    g.codec_tokens[i]->Set(static_cast<double>(tokens));
  }
}

}  // namespace tt::worker
