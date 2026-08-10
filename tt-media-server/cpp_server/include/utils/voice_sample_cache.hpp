// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

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
    touch(entry->second);
    return entry->second.speechIds;
  }

  void add(const Samples& samples, SpeechIds speechIds) {
    if (capacity == 0) return;
    auto existing = entries.find(samples);
    if (existing != entries.end()) {
      existing->second.speechIds = std::move(speechIds);
      touch(existing->second);
      return;
    }
    if (entries.size() == capacity) {
      Entry* leastRecentlyUsed = heap.front();
      entries.erase(leastRecentlyUsed->samples);
      heap.front() = heap.back();
      heap.pop_back();
      if (!heap.empty()) {
        heap.front()->heapIndex = 0;
        siftDown(0);
      }
    }
    auto [entry, inserted] = entries.emplace(
        samples,
        Entry{samples, std::move(speechIds), ++accessOrder, heap.size()});
    (void)inserted;
    heap.push_back(&entry->second);
    siftUp(entry->second.heapIndex);
  }

 private:
  struct SamplesHash {
    size_t operator()(const Samples& samples) const noexcept {
      size_t hash = samples.size();
      for (int16_t sample : samples) {
        hash ^= static_cast<uint16_t>(sample) + 0x9e3779b9 + (hash << 6) +
                (hash >> 2);
      }
      return hash;
    }
  };
  struct Entry {
    Samples samples;
    SpeechIds speechIds;
    uint64_t accessOrder;
    size_t heapIndex;
  };
  void touch(Entry& entry) {
    entry.accessOrder = ++accessOrder;
    siftDown(entry.heapIndex);
  }
  void siftUp(size_t index) {
    while (index > 0) {
      const size_t parent = (index - 1) / 2;
      if (heap[parent]->accessOrder <= heap[index]->accessOrder) return;
      swapHeapEntries(parent, index);
      index = parent;
    }
  }
  void siftDown(size_t index) {
    while (true) {
      const size_t left = index * 2 + 1;
      const size_t right = left + 1;
      size_t smallest = index;
      if (left < heap.size() &&
          heap[left]->accessOrder < heap[smallest]->accessOrder)
        smallest = left;
      if (right < heap.size() &&
          heap[right]->accessOrder < heap[smallest]->accessOrder)
        smallest = right;
      if (smallest == index) return;
      swapHeapEntries(index, smallest);
      index = smallest;
    }
  }
  void swapHeapEntries(size_t first, size_t second) {
    std::swap(heap[first], heap[second]);
    heap[first]->heapIndex = first;
    heap[second]->heapIndex = second;
  }

  size_t capacity;
  uint64_t accessOrder = 0;
  std::unordered_map<Samples, Entry, SamplesHash> entries;
  std::vector<Entry*> heap;
};

}  // namespace tt::utils
