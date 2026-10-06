// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_set>
#include <vector>

#include "ipc/tts_ipc.hpp"

/** The Python-free parts of the Qwen3-TTS runner, split out so they can be
 *  unit tested without an interpreter. */
namespace tt::runners::qwen3_tts {

/**
 * Split one utterance's PCM16 samples into audio-queue messages of at most
 * `chunkSamples` samples each, indexed from 0. Empty input yields no message.
 */
std::vector<ipc::tts::TtsAudioChunkMessage> splitIntoChunks(
    uint32_t taskId, const std::vector<int16_t>& samples, size_t chunkSamples,
    uint32_t sampleRateHz);

/**
 * Remembers recently cancelled task ids. The parent broadcasts every cancel to
 * every worker, possibly before the task is even dequeued, so the set is
 * bounded: the oldest ids fall out once it is full.
 */
class CancelledTaskSet {
 public:
  explicit CancelledTaskSet(size_t capacity) : capacity(capacity) {}

  void add(uint32_t taskId);
  bool contains(uint32_t taskId) const { return ids.contains(taskId); }
  /** Forget `taskId`, returning whether it was there. */
  bool take(uint32_t taskId);
  size_t size() const { return ids.size(); }

 private:
  size_t capacity;
  std::deque<uint32_t> order;
  std::unordered_set<uint32_t> ids;
};

}  // namespace tt::runners::qwen3_tts
