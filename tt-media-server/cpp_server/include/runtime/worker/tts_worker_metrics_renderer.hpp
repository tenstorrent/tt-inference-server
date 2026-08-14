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
 *
 * The codec-token series is the cumulative count of acoustic tokens the TTS
 * decoder emitted on that worker; throughput is
 * `rate(tt_tts_codec_tokens_total[$__rate_interval])`. It carries the same
 * `_total` naming as the other shm-backed cumulative series and is likewise
 * exported as a gauge, because the shm transport publishes an absolute value
 * per scrape that the renderer Set()s (prometheus-cpp counters cannot be set).
 * The value is monotonic for the worker's lifetime, so rate() is well behaved
 * and treats a worker restart as a counter reset.
 *
 * `device` is the worker's DEVICE_IDS group (its TT_VISIBLE_DEVICES value) —
 * the per-device granularity. There is no `voice` or `language` label: the
 * TTS API accepts only text, a free-form description and an optional voice
 * WAV, so neither dimension exists to label by; `voice_source` is the bounded
 * stand-in.
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
    // One gauge per VoiceSource, indexed by its enum value.
    std::array<prometheus::Gauge*, tts::VOICE_SOURCE_COUNT> codec_tokens{};
  };

  prometheus::Family<prometheus::Gauge>* alive_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* step_age_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* output_age_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* codec_tokens_family_{nullptr};

  std::unordered_map<int, WorkerGauges> gauges_;
};

}  // namespace tt::worker
