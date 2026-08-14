// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstddef>
#include <cstdint>

#include "runtime/worker/worker_metrics_shm.hpp"

namespace tt::worker::tts {

/**
 * Scratch-area index convention for the TTS runner family (tagged in shared
 * memory as MetricsLayout::TTS_RUNNER).
 *
 * Both writer (worker-side BlazeTtsRunner via SingleProcessWorkerMetrics) and
 * reader (main-side TtsWorkerMetricsRenderer) include this header so they
 * agree on what each scratch slot means.
 *
 * Indices are append-only and are a namespace of their own: they are NOT
 * interchangeable with sp_pipeline's, even where the numbers coincide. The
 * heartbeat cells deliberately mirror sp_pipeline's 0/1 so the two layouts
 * read the same way in a debugger, but every reader/writer resolves the index
 * through its own layout header.
 */

constexpr size_t SCRATCH_STEP_EPOCH_MS = 0;
constexpr size_t SCRATCH_LAST_OUTPUT_EPOCH_MS = 1;

/**
 * How the voice for a request was specified. The TTS API has no voice ID, so
 * this is the coarsest honest breakdown of "which voice path produced these
 * tokens" that stays bounded — a cloned voice (VoiceSample) runs a different
 * amount of work per token than the default speaker.
 *
 * Values are scratch-index offsets; append-only, never renumber.
 */
enum class VoiceSource : uint8_t {
  Default = 0,      // no description, no voice sample
  Description = 1,  // free-form description prompt
  VoiceSample = 2,  // cloned from an uploaded voice WAV
};

constexpr size_t VOICE_SOURCE_COUNT = 3;

/**
 * Cumulative codec (acoustic) tokens emitted by this worker, one counter per
 * VoiceSource. Only the per-source cells exist — the total is
 * `sum without (voice_source)` in PromQL, so there is no aggregate cell to
 * drift out of sync with its parts.
 */
constexpr size_t CODEC_TOKENS_BASE = 2;

inline size_t codecTokensIdx(VoiceSource source) {
  return CODEC_TOKENS_BASE + static_cast<size_t>(source);
}

/** Prometheus `voice_source` label value for a source. */
inline const char* voiceSourceLabel(VoiceSource source) {
  switch (source) {
    case VoiceSource::Default:
      return "default";
    case VoiceSource::Description:
      return "description";
    case VoiceSource::VoiceSample:
      return "voice_sample";
  }
  return "default";
}

// Indices CODEC_TOKENS_BASE + VOICE_SOURCE_COUNT .. 31 reserved for future
// aggregates (audio seconds emitted, real-time factor, ...).
constexpr size_t SCRATCH_RESERVED_END = 32;

static_assert(CODEC_TOKENS_BASE + VOICE_SOURCE_COUNT <= SCRATCH_RESERVED_END,
              "tts codec-token cells overflow the reserved aggregate region");
static_assert(SCRATCH_RESERVED_END <= WORKER_SCRATCH_U64_COUNT,
              "tts layout exceeds scratch capacity");

}  // namespace tt::worker::tts
