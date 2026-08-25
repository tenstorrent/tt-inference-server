// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <prometheus/gauge.h>
#include <prometheus/registry.h>

#include <array>
#include <unordered_map>

#include "runtime/worker/tts_metrics_layout.hpp"
#include "runtime/worker/worker_metrics_renderer.hpp"
#include "runtime/worker/worker_metrics_shm.hpp"

namespace tt::worker {

/**
 * Renderer for slots tagged MetricsLayout::TTS_RUNNER (produced by
 * BlazeTtsRunner). Translates the tts scratch indices into:
 *   - tt_worker_alive{worker_id}
 *   - tt_worker_heartbeat_age_seconds{worker_id}
 *   - tt_worker_last_output_age_seconds{worker_id}
 *   - tt_tts_codec_tokens_total{worker_id, device, model_name, voice_source}
 *   - tt_tts_audio_frames_total{worker_id, device, model_name, batch}
 *   - tt_tts_vocoder_chunks_total{worker_id, device, model_name, batch}
 *   - tt_tts_audio_sample_rate_hz{worker_id}
 *   - tt_tts_last_vocode_age_seconds{worker_id}
 *
 * The `_total` series are cumulative counts exported as gauges: the shm
 * transport publishes an absolute value per scrape that the renderer Set()s,
 * and prometheus-cpp counters cannot be set. Each is monotonic for the worker's
 * lifetime, so rate() is well behaved and reads a worker restart as a reset.
 *
 * `device` is the worker's DEVICE_IDS group (its TT_VISIBLE_DEVICES value).
 * There is deliberately no `voice` or `language` label — the TTS API exposes
 * neither dimension, and `voice_source` is the bounded stand-in. `batch` is
 * derived, not reported by the engine; see BatchBucket in
 * tts_metrics_layout.hpp for what it does and does not mean.
 *
 * What these series answer, and the queries for it, are in
 * monitoring/README.md.
 */
class TtsWorkerMetricsRenderer : public IWorkerMetricsRenderer {
 public:
  void prebuildGauges(prometheus::Registry& registry, int workerId) override;
  void render(const WorkerMetricsShm& shm, int workerId, bool isAlive) override;

 private:
  struct WorkerGauges {
    prometheus::Gauge* alive{nullptr};
    prometheus::Gauge* step_age{nullptr};
    prometheus::Gauge* output_age{nullptr};
    prometheus::Gauge* vocode_age{nullptr};
    prometheus::Gauge* sample_rate{nullptr};
    // One gauge per VoiceSource, indexed by its enum value.
    std::array<prometheus::Gauge*, tts::VOICE_SOURCE_COUNT> codec_tokens{};
    // One gauge per BatchBucket, indexed by its enum value.
    std::array<prometheus::Gauge*, tts::BATCH_BUCKET_COUNT> audio_frames{};
    std::array<prometheus::Gauge*, tts::BATCH_BUCKET_COUNT> vocoder_chunks{};
  };

  prometheus::Family<prometheus::Gauge>* alive_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* step_age_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* output_age_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* codec_tokens_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* audio_frames_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* vocoder_chunks_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* sample_rate_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* vocode_age_family_{nullptr};

  std::unordered_map<int, WorkerGauges> gauges_;
};

}  // namespace tt::worker
