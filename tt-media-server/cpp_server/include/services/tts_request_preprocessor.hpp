// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "config/runner_config.hpp"
#include "domain/tts/tts_types.hpp"

namespace tt::services {

/** Converts client-facing TTS requests into worker-boundary TTS tasks.
 *
 * The preprocessor owns request validation, text prompt tokenization, and
 * voice-reference PCM normalization before the task crosses into worker IPC.
 */
class TtsRequestPreprocessor {
 public:
  explicit TtsRequestPreprocessor(config::TtsConfig config);

  tt::domain::tts::TtsTask process(
      const tt::domain::tts::TtsRequest& request) const;

 private:
  tt::domain::tts::VoiceSample normalizeVoiceSample(
      const tt::domain::tts::VoiceSample& sample) const;

  config::TtsConfig config;
};

}  // namespace tt::services
