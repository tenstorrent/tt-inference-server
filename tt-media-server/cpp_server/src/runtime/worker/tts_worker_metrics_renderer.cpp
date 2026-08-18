// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runtime/worker/tts_worker_metrics_renderer.hpp"

#include <string>

#include "config/settings.hpp"
#include "runtime/worker/tts_metrics_layout.hpp"
#include "runtime/worker/worker_metrics_clock.hpp"

namespace tt::worker {

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
    audio_frames_family_ =
        &prometheus::BuildGauge()
             .Name("tt_tts_audio_frames_total")
             .Help(
                 "Cumulative PCM frames (samples per channel) the vocoder "
                 "reconstructed from acoustic tokens on this worker since it "
                 "last (re)started. Samples/s is "
                 "rate(tt_tts_audio_frames_total[...]); audio seconds per "
                 "wall second (the real-time factor, <1.0 means "
                 "reconstruction cannot keep up with playback) is that rate "
                 "divided by tt_tts_audio_sample_rate_hz. Read against "
                 "tt_tts_codec_tokens_total it separates a waveform-"
                 "reconstruction bottleneck from a token-generation one. "
                 "Counts what the vocoder produced, including chunks a full "
                 "audio queue then rejected. Labelled by worker_id, its "
                 "DEVICE_IDS group (device), model_name, and the bucketed "
                 "vocode batch size (batch).")
             .Register(registry);
    vocoder_chunks_family_ =
        &prometheus::BuildGauge()
             .Name("tt_tts_vocoder_chunks_total")
             .Help(
                 "Cumulative audio chunks the vocoder emitted, bucketed like "
                 "tt_tts_audio_frames_total. Dividing frames by chunks gives "
                 "mean frames per chunk, which tells apart 'fewer chunks' "
                 "from 'shorter chunks' when audio throughput drops.")
             .Register(registry);
    sample_rate_family_ =
        &prometheus::BuildGauge()
             .Name("tt_tts_audio_sample_rate_hz")
             .Help(
                 "Output sample rate this worker emits (TTS_AUDIO_SAMPLE_RATE"
                 "_HZ). Divisor that converts tt_tts_audio_frames_total into "
                 "audio seconds.")
             .Register(registry);
    vocode_age_family_ =
        &prometheus::BuildGauge()
             .Name("tt_tts_last_vocode_age_seconds")
             .Help(
                 "Seconds since the worker last emitted a vocoded audio "
                 "chunk. Its own clock, separate from "
                 "tt_worker_last_output_age_seconds (last codec token): this "
                 "one ageing while that one does not is the vocoder stalling "
                 "behind a healthy decoder.")
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
  g.vocode_age = &vocode_age_family_->Add({{"worker_id", idStr}});
  g.sample_rate = &sample_rate_family_->Add({{"worker_id", idStr}});
  for (size_t i = 0; i < tts::VOICE_SOURCE_COUNT; ++i) {
    g.codec_tokens[i] = &codec_tokens_family_->Add(
        {{"worker_id", idStr},
         {"device", device},
         {"model_name", modelName},
         {"voice_source",
          tts::voiceSourceLabel(static_cast<tts::VoiceSource>(i))}});
  }
  for (size_t i = 0; i < tts::BATCH_BUCKET_COUNT; ++i) {
    const prometheus::Labels labels{
        {"worker_id", idStr},
        {"device", device},
        {"model_name", modelName},
        {"batch", tts::batchBucketLabel(static_cast<tts::BatchBucket>(i))}};
    g.audio_frames[i] = &audio_frames_family_->Add(labels);
    g.vocoder_chunks[i] = &vocoder_chunks_family_->Add(labels);
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
  const uint64_t vocodeMs =
      shm.loadScratch(slot, tts::SCRATCH_LAST_VOCODE_EPOCH_MS);

  g.alive->Set(isAlive ? 1.0 : 0.0);
  g.step_age->Set(ageSeconds(stepMs, now));
  g.output_age->Set(ageSeconds(outputMs, now));
  g.vocode_age->Set(ageSeconds(vocodeMs, now));
  g.sample_rate->Set(static_cast<double>(
      shm.loadScratch(slot, tts::SCRATCH_AUDIO_SAMPLE_RATE_HZ)));

  for (size_t i = 0; i < tts::VOICE_SOURCE_COUNT; ++i) {
    const uint64_t tokens = shm.loadScratch(
        slot, tts::codecTokensIdx(static_cast<tts::VoiceSource>(i)));
    g.codec_tokens[i]->Set(static_cast<double>(tokens));
  }

  for (size_t i = 0; i < tts::BATCH_BUCKET_COUNT; ++i) {
    const auto bucket = static_cast<tts::BatchBucket>(i);
    g.audio_frames[i]->Set(static_cast<double>(
        shm.loadScratch(slot, tts::audioFramesIdx(bucket))));
    g.vocoder_chunks[i]->Set(static_cast<double>(
        shm.loadScratch(slot, tts::vocoderChunksIdx(bucket))));
  }
}

}  // namespace tt::worker
