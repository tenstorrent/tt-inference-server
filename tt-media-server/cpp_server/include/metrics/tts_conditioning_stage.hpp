// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstddef>
#include <cstdint>

namespace tt::metrics {

/**
 * A stage of TTS request preparation — building conditioning from text, or
 * from a voice sample — as opposed to synthesis itself. Short utterances can
 * be dominated by these rather than by generating audio, which is only visible
 * if they are timed apart from the request as a whole.
 *
 * The four stages are where that work actually happens, and which ones run
 * depends on the request:
 *   TextConditioning   main process, requests with no voice sample: tokenizer
 *                      lookup + prompt compilation. Named for its input, not
 *                      for one of its steps — it covers that whole path, which
 *                      is why PromptCompile does not also fire on it.
 *   VoiceNormalization main process, requests with a voice sample: validation,
 *                      downmix to mono, resample to the encoder's rate.
 *   VoiceEncode        worker process, voice-sample requests only: encoding the
 *                      reference WAV into speech IDs on device. Skipped (and
 *                      so reported as not run) when the voice-sample cache
 *                      already holds the IDs.
 *   PromptCompile      worker process, voice-sample requests only: prompt
 *                      compilation once the speech IDs exist.
 *
 * This lives in its own header, free of prometheus includes, so the TTS service
 * can name a stage without every consumer of its header inheriting a dependency
 * on the metrics backend.
 *
 * Values are array indices; append-only, never renumber.
 */
enum class TtsConditioningStage : uint8_t {
  TextConditioning = 0,
  VoiceNormalization = 1,
  VoiceEncode = 2,
  PromptCompile = 3,
};

constexpr size_t TTS_CONDITIONING_STAGE_COUNT = 4;

/** Prometheus `stage` label value for a conditioning stage. */
const char* ttsConditioningStageLabel(TtsConditioningStage stage);

}  // namespace tt::metrics
