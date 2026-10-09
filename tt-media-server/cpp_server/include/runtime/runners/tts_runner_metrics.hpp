// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>

#include "ipc/tts_ipc.hpp"
#include "runtime/worker/tts_metrics_layout.hpp"

/** Metric attribution shared by the TTS runners (BlazeTtsRunner,
 *  Qwen3TtsRunner), so every TTS runner labels and times requests the same
 *  way. */
namespace tt::runners::tts {

/** Microseconds elapsed since `start`, saturating at uint32 (~71 min) so a
 *  pathological stall cannot wrap the IPC field into a small value. */
inline uint32_t elapsedUsSince(std::chrono::steady_clock::time_point start) {
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  if (us <= 0) return 0;
  return static_cast<uint32_t>(
      std::min<int64_t>(us, std::numeric_limits<uint32_t>::max()));
}

/** Bounded metrics dimension for "which voice produced these tokens". A
 *  cloned voice costs more per token than the default speaker, and the TTS
 *  API exposes no voice ID to label by. */
inline tt::worker::tts::VoiceSource voiceSourceOf(
    const ipc::tts::TtsIpcTask& task) {
  if (!task.voiceWavPcm.empty()) {
    return tt::worker::tts::VoiceSource::VoiceSample;
  }
  if (task.description.has_value() && !task.description->empty()) {
    return tt::worker::tts::VoiceSource::Description;
  }
  return tt::worker::tts::VoiceSource::Default;
}

}  // namespace tt::runners::tts
