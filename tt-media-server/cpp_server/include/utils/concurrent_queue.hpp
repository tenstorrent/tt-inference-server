// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <mutex>
#include <vector>

#include "profiling/tracy.hpp"

namespace tt::utils {

template <typename T>
class ConcurrentQueue {
 public:
  ConcurrentQueue() = default;
  ~ConcurrentQueue() = default;

  void push(const T& value) {
    std::lock_guard lock(mutex);
    pending.push_back(value);
  }

  std::vector<T> drain() {
    std::lock_guard lock(mutex);
    std::vector<T> out;
    out.swap(pending);
    return out;
  }

  size_t size() {
    std::lock_guard lock(mutex);
    return pending.size();
  }

  ConcurrentQueue(const ConcurrentQueue&) = delete;
  ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;

 private:
  std::vector<T> pending;
  TRACY_LOCKABLE(std::mutex, mutex);
};

namespace detail {

// Fixed size avoids -Winterference-size
// (std::hardware_destructive_interference_size is not stable across
// compiler/tuning); 64 matches typical x86/ARM cache lines.
inline constexpr size_t CACHE_LINE_SIZE = 64;

inline size_t nextPowerOfTwo(size_t n) { return std::bit_ceil(n); }
inline constexpr std::memory_order RELAXED = std::memory_order_relaxed;
inline constexpr std::memory_order ACQUIRE = std::memory_order_acquire;
inline constexpr std::memory_order RELEASE = std::memory_order_release;

}  // namespace detail

template <typename T>
class LockFreeSPSCQueue {
 public:
  LockFreeSPSCQueue(size_t capacity)
      : capacity(detail::nextPowerOfTwo(capacity + 1)),
        buffer(detail::nextPowerOfTwo(capacity + 1)) {
    mask = this->capacity - 1;
  }
  ~LockFreeSPSCQueue() = default;

  bool push(const T& value) {
    const size_t HEAD = head.load(detail::RELAXED);
    const size_t NEXT_HEAD = (HEAD + 1) & mask;

    if (NEXT_HEAD == tail.load(detail::ACQUIRE)) {
      return false;
    }

    buffer[HEAD] = value;

    head.store(NEXT_HEAD, detail::RELEASE);
    return true;
  }

  bool pop(T& value) {
    const size_t TAIL = tail.load(detail::RELAXED);

    if (TAIL == head.load(detail::ACQUIRE)) {
      return false;
    }

    value = std::move(buffer[TAIL]);

    tail.store((TAIL + 1) & mask, detail::RELEASE);
    return true;
  }

  size_t pushMany(const std::vector<T>& items) {
    const size_t HEAD = head.load(detail::RELAXED);
    const size_t TAIL = tail.load(detail::ACQUIRE);

    size_t available = (TAIL - HEAD - 1) & mask;
    size_t toPush = std::min(items.size(), available);

    if (toPush == 0) return 0;

    for (size_t i = 0; i < toPush; ++i) {
      buffer[(HEAD + i) & mask] = items[i];
    }

    head.store((HEAD + toPush) & mask, detail::RELEASE);
    return toPush;
  }

  size_t popMany(std::vector<T>& outItems, size_t maxItems) {
    const size_t TAIL = tail.load(detail::RELAXED);
    const size_t HEAD = head.load(detail::ACQUIRE);

    size_t occupied = (HEAD - TAIL) & mask;
    size_t toPop = std::min(maxItems, occupied);

    if (toPop == 0) return 0;

    for (size_t i = 0; i < toPop; ++i) {
      outItems.push_back(std::move(buffer[(TAIL + i) & mask]));
    }

    tail.store((TAIL + toPop) & mask, detail::RELEASE);
    return toPop;
  }

  size_t size() const { return (head.load() - tail.load()) & mask; }

  LockFreeSPSCQueue(const LockFreeSPSCQueue&) = delete;
  LockFreeSPSCQueue& operator=(const LockFreeSPSCQueue&) = delete;

 private:
  size_t capacity;
  size_t mask;
  std::vector<T> buffer;
  alignas(detail::CACHE_LINE_SIZE) std::atomic<size_t> head{0};
  alignas(detail::CACHE_LINE_SIZE) std::atomic<size_t> tail{0};
};

}  // namespace tt::utils
