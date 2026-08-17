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

/**
 * Vocoder batch size, bucketed. The vocoder stage turns generated acoustic
 * tokens back into PCM, and how many streams it reconstructs together is what
 * decides whether waveform reconstruction keeps up — so the audio counters are
 * attributed per bucket rather than only per worker.
 *
 * The engine does not report the batch it formed (`engine_tts::AudioOut`
 * carries only uid / chunk_index / samples_bf16), so the runner derives it: the
 * number of distinct slots whose chunks came out of one drainAudioOutputs()
 * sweep. The engine vocodes a batch and pushes one AudioOut per stream in it,
 * so a sweep observes one batch's worth. It is a proxy, not ground truth — a
 * batch can straddle two sweeps under load, which shows up as two smaller
 * buckets rather than one larger one.
 *
 * Buckets rather than a raw count keep the label bounded at 6 values
 * regardless of TTS_MAX_BATCH_SIZE / PM_MAX_USERS (which alone would admit
 * 128). Values are scratch-index offsets; append-only, never renumber.
 */
enum class BatchBucket : uint8_t {
  B1 = 0,
  B2 = 1,
  B3_4 = 2,
  B5_8 = 3,
  B9_16 = 4,
  B17Plus = 5,
};

constexpr size_t BATCH_BUCKET_COUNT = 6;

/** Bucket for a vocode batch of `streams` concurrent streams. */
inline BatchBucket batchBucketOf(size_t streams) {
  if (streams <= 1) return BatchBucket::B1;
  if (streams == 2) return BatchBucket::B2;
  if (streams <= 4) return BatchBucket::B3_4;
  if (streams <= 8) return BatchBucket::B5_8;
  if (streams <= 16) return BatchBucket::B9_16;
  return BatchBucket::B17Plus;
}

/** Prometheus `batch` label value for a bucket. */
inline const char* batchBucketLabel(BatchBucket bucket) {
  switch (bucket) {
    case BatchBucket::B1:
      return "1";
    case BatchBucket::B2:
      return "2";
    case BatchBucket::B3_4:
      return "3-4";
    case BatchBucket::B5_8:
      return "5-8";
    case BatchBucket::B9_16:
      return "9-16";
    case BatchBucket::B17Plus:
      return "17+";
  }
  return "1";
}

/**
 * Cumulative PCM frames (samples per channel) the vocoder reconstructed, one
 * counter per BatchBucket. Frames rather than raw sample words so the series
 * is channel-count independent: audio seconds = frames / sample_rate_hz.
 */
constexpr size_t AUDIO_FRAMES_BASE = 5;

inline size_t audioFramesIdx(BatchBucket bucket) {
  return AUDIO_FRAMES_BASE + static_cast<size_t>(bucket);
}

/**
 * Cumulative audio chunks emitted, same bucketing. Paired with the frame
 * counter it gives mean frames-per-chunk, which separates "fewer chunks" from
 * "shorter chunks" when audio throughput drops.
 */
constexpr size_t VOCODER_CHUNKS_BASE = 11;

inline size_t vocoderChunksIdx(BatchBucket bucket) {
  return VOCODER_CHUNKS_BASE + static_cast<size_t>(bucket);
}

/**
 * Output sample rate the runner is configured to emit, published once at
 * runner construction. Lets the reader convert frames to audio seconds
 * without reaching into config, and makes the rate visible in its own right.
 */
constexpr size_t SCRATCH_AUDIO_SAMPLE_RATE_HZ = 17;

/**
 * Epoch ms of the last vocoded chunk, distinct from
 * SCRATCH_LAST_OUTPUT_EPOCH_MS (last codec token). Two separate staleness
 * clocks are what localize a stall: tokens still advancing while this one ages
 * means the vocoder is the bottleneck, both ageing together points upstream at
 * token generation.
 */
constexpr size_t SCRATCH_LAST_VOCODE_EPOCH_MS = 18;

// Indices 19..31 reserved for future aggregates (real-time factor, vocode
// latency, ...).
constexpr size_t SCRATCH_RESERVED_END = 32;

static_assert(CODEC_TOKENS_BASE + VOICE_SOURCE_COUNT <= AUDIO_FRAMES_BASE,
              "tts codec-token cells overlap the audio-frame region");
static_assert(AUDIO_FRAMES_BASE + BATCH_BUCKET_COUNT <= VOCODER_CHUNKS_BASE,
              "tts audio-frame cells overlap the vocoder-chunk region");
static_assert(VOCODER_CHUNKS_BASE + BATCH_BUCKET_COUNT <=
                  SCRATCH_AUDIO_SAMPLE_RATE_HZ,
              "tts vocoder-chunk cells overlap the sample-rate cell");
static_assert(SCRATCH_LAST_VOCODE_EPOCH_MS < SCRATCH_RESERVED_END,
              "tts vocode cells overflow the reserved aggregate region");
static_assert(SCRATCH_RESERVED_END <= WORKER_SCRATCH_U64_COUNT,
              "tts layout exceeds scratch capacity");

}  // namespace tt::worker::tts
