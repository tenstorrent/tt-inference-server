// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/qwen3_tts/qwen3_tts_runner_support.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tt::runners::qwen3_tts {

std::vector<ipc::tts::TtsAudioChunkMessage> splitIntoChunks(
    uint32_t taskId, const std::vector<int16_t>& samples, size_t chunkSamples,
    uint32_t sampleRateHz) {
  if (chunkSamples == 0) {
    throw std::invalid_argument("chunkSamples must be > 0");
  }
  std::vector<ipc::tts::TtsAudioChunkMessage> chunks;
  chunks.reserve((samples.size() + chunkSamples - 1) / chunkSamples);
  for (size_t begin = 0; begin < samples.size(); begin += chunkSamples) {
    const size_t end = std::min(samples.size(), begin + chunkSamples);
    ipc::tts::TtsAudioChunkMessage message;
    message.task_id = taskId;
    message.chunkIndex = static_cast<uint32_t>(chunks.size());
    message.sampleRateHz = sampleRateHz;
    message.channels = 1;
    message.samplesPcm16.assign(samples.begin() + begin, samples.begin() + end);
    chunks.push_back(std::move(message));
  }
  return chunks;
}

void CancelledTaskSet::add(uint32_t taskId) {
  if (capacity == 0 || !ids.insert(taskId).second) return;
  order.push_back(taskId);
  while (order.size() > capacity) {
    ids.erase(order.front());
    order.pop_front();
  }
}

bool CancelledTaskSet::take(uint32_t taskId) {
  if (ids.erase(taskId) == 0) return false;
  order.erase(std::find(order.begin(), order.end(), taskId));
  return true;
}

}  // namespace tt::runners::qwen3_tts
