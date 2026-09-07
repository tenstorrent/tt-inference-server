// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/tts_request_preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "utils/tts_prompt_compiler.hpp"
#include "utils/tts_tokenizer.hpp"

namespace tt::services {

namespace {

int16_t clampToI16(double value) {
  const double min = static_cast<double>(std::numeric_limits<int16_t>::min());
  const double max = static_cast<double>(std::numeric_limits<int16_t>::max());
  return static_cast<int16_t>(std::clamp(std::round(value), min, max));
}

std::vector<int16_t> downmixToMono(const std::vector<int16_t>& samples,
                                   uint16_t channels) {
  if (channels == 1) {
    return samples;
  }
  if (channels == 0) {
    throw std::invalid_argument("voice sample channel count must be > 0");
  }
  if (samples.size() % channels != 0) {
    throw std::invalid_argument(
        "voice sample PCM length must be divisible by channel count");
  }

  const size_t frames = samples.size() / channels;
  std::vector<int16_t> mono;
  mono.reserve(frames);
  for (size_t frame = 0; frame < frames; ++frame) {
    int64_t sum = 0;
    for (uint16_t channel = 0; channel < channels; ++channel) {
      sum += samples[frame * channels + channel];
    }
    mono.push_back(clampToI16(static_cast<double>(sum) / channels));
  }
  return mono;
}

std::vector<int16_t> resampleLinear(const std::vector<int16_t>& mono,
                                    uint32_t sourceRate, uint32_t targetRate) {
  if (sourceRate == 0 || targetRate == 0) {
    throw std::invalid_argument("voice sample rate must be > 0");
  }
  if (sourceRate == targetRate || mono.empty()) {
    return mono;
  }

  const double ratio = static_cast<double>(targetRate) / sourceRate;
  const auto outputSize =
      static_cast<size_t>(std::max(1.0, std::ceil(mono.size() * ratio)));
  std::vector<int16_t> out;
  out.reserve(outputSize);

  for (size_t i = 0; i < outputSize; ++i) {
    const double sourcePos = static_cast<double>(i) * sourceRate / targetRate;
    const size_t left =
        std::min(static_cast<size_t>(sourcePos), mono.size() - 1);
    const size_t right = std::min(left + 1, mono.size() - 1);
    const double frac = sourcePos - left;
    const double sample =
        (1.0 - frac) * mono[left] + frac * static_cast<double>(mono[right]);
    out.push_back(clampToI16(sample));
  }
  return out;
}

}  // namespace

TtsRequestPreprocessor::TtsRequestPreprocessor(config::TtsConfig config)
    : config(std::move(config)) {}

tt::domain::tts::TtsTask TtsRequestPreprocessor::process(
    const tt::domain::tts::TtsRequest& request) const {
  tt::domain::tts::TtsTask task;
  task.task_id = request.task_id;
  task.text = request.text;
  task.description = request.description;

  if (request.voiceSample.has_value()) {
    tt::utils::tts_prompt_compiler::validatePromptInputs(request.text,
                                                         request.description);
    auto normalized = normalizeVoiceSample(*request.voiceSample);
    task.voiceWavPcm = std::move(normalized.wavPcm);
  } else {
    const auto& tokenizer =
        tt::utils::tts_tokenizer::tokenizerForPath(config.tokenizerPath);
    task.promptTokens = tt::utils::tts_prompt_compiler::compilePromptTokens(
        tokenizer, request.text, request.description, /*promptSpeechIds=*/{},
        config.bosToken);
  }

  return task;
}

tt::domain::tts::VoiceSample TtsRequestPreprocessor::normalizeVoiceSample(
    const tt::domain::tts::VoiceSample& sample) const {
  if (sample.wavPcm.empty()) {
    throw std::invalid_argument("voice sample PCM data must not be empty");
  }

  auto mono = downmixToMono(sample.wavPcm, sample.channels);
  auto normalized =
      resampleLinear(mono, sample.sampleRateHz, config.voiceSampleRateHz);

  tt::domain::tts::VoiceSample out;
  out.wavPcm = std::move(normalized);
  out.sampleRateHz = config.voiceSampleRateHz;
  out.channels = config.voiceChannels;
  return out;
}

}  // namespace tt::services
