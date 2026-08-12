// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#define XXH_INLINE_ALL
#include "xxhash.h"

namespace tt::utils {

class VoiceSampleCache {
 public:
  using Samples = std::vector<int16_t>;
  using SpeechIds = std::vector<uint32_t>;

  explicit VoiceSampleCache(size_t capacity) : capacity(capacity) {}

  bool exists(const Samples& samples) const {
    return entries.find(samples) != entries.end();
  }

  SpeechIds get(const Samples& samples) {
    auto entry = entries.find(samples);
    if (entry == entries.end()) {
      throw std::out_of_range("Voice sample cache entry does not exist");
    }
    entriesByRecency.splice(entriesByRecency.begin(), entriesByRecency,
                            entry->second);
    return entry->second->speechIds;
  }

  void add(const Samples& samples, SpeechIds speechIds) {
    if (capacity == 0) return;
    auto existing = entries.find(samples);
    if (existing != entries.end()) {
      existing->second->speechIds = std::move(speechIds);
      entriesByRecency.splice(entriesByRecency.begin(), entriesByRecency,
                              existing->second);
      return;
    }
    if (entries.size() == capacity) {
      entries.erase(entriesByRecency.back().samples);
      entriesByRecency.pop_back();
    }
    entriesByRecency.push_front({samples, std::move(speechIds)});
    entries.emplace(samples, entriesByRecency.begin());
  }

 private:
  struct SamplesHash {
    size_t operator()(const Samples& samples) const noexcept {
      if (samples.empty()) return 0;
      return XXH64(samples.data(), samples.size() * sizeof(int16_t), 0);
    }
  };
  struct Entry {
    Samples samples;
    SpeechIds speechIds;
  };

  size_t capacity;
  std::list<Entry> entriesByRecency;
  std::unordered_map<Samples, std::list<Entry>::iterator, SamplesHash> entries;
};

}  // namespace tt::utils
