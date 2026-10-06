// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "config/runner_config.hpp"
#include "domain/tts/tts_types.hpp"
#include "services/qwen3_tts_request_validator.hpp"

namespace tt::services {

/** Converts client-facing TTS requests into worker-boundary TTS tasks.
 *
 * The preprocessor owns request validation, text prompt tokenization, and
 * voice-reference PCM normalization before the task crosses into worker IPC.
 * Qwen3-TTS templates its own prompt in the worker, so for it the
 * preprocessor validates the request against the served release and passes
 * the raw fields through instead of compiling TTS-2 prompt tokens.
 */
class TtsRequestPreprocessor {
 public:
  explicit TtsRequestPreprocessor(config::TtsConfig config);

  tt::domain::tts::TtsTask process(
      const tt::domain::tts::TtsRequest& request) const;

 private:
  tt::domain::tts::TtsTask processQwen3(
      const tt::domain::tts::TtsRequest& request) const;
  tt::domain::tts::VoiceSample normalizeVoiceSample(
      const tt::domain::tts::VoiceSample& sample) const;

  config::TtsConfig config;
  // Set only for TT_QWEN3_TTS.
  std::shared_ptr<const qwen3_tts::RequestValidator> qwen3Validator;
};

}  // namespace tt::services
