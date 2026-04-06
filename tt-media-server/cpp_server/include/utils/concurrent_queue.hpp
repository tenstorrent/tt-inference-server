// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <mutex>
#include <vector>

#include "profiling/tracy.hpp"

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

#ifdef __cpp_lib_hardware_interference_size
static constexpr size_t CACHE_LINE_SIZE =
    std::hardware_destructive_interference_size;
#else
static constexpr size_t CACHE_LINE_SIZE = 64;
#endif
namespace {
inline size_t nextPowerOfTwo(size_t n) { return std::bit_ceil(n); }
constexpr std::memory_order RELAXED = std::memory_order_relaxed;
constexpr std::memory_order ACQUIRE = std::memory_order_acquire;
constexpr std::memory_order RELEASE = std::memory_order_release;
}  // namespace

template <typename T>
class LockFreeSpscQueue {
 public:
  LockFreeSpscQueue(size_t capacity)
      : capacity(nextPowerOfTwo(capacity + 1)),
        buffer(nextPowerOfTwo(capacity + 1)) {
    mask = this->capacity - 1;
  }
  ~LockFreeSpscQueue() = default;

  bool push(const T& value) {
    const size_t HEAD = head.load(RELAXED);
    const size_t NEXT_HEAD = (HEAD + 1) & mask;

    if (NEXT_HEAD == tail.load(ACQUIRE)) {
      return false;
    }

    buffer[HEAD] = value;

    head.store(NEXT_HEAD, RELEASE);
    return true;
  }

  bool pop(T& value) {
    const size_t TAIL = tail.load(RELAXED);

    if (TAIL == head.load(ACQUIRE)) {
      return false;
    }

    value = std::move(buffer[TAIL]);

    tail.store((TAIL + 1) & mask, RELEASE);
    return true;
  }

  size_t pushMany(const std::vector<T>& items) {
    const size_t HEAD = head.load(RELAXED);
    const size_t TAIL = tail.load(ACQUIRE);

    size_t available = (TAIL - HEAD - 1) & mask;
    size_t toPush = std::min(items.size(), available);

    if (toPush == 0) return 0;

    for (size_t i = 0; i < toPush; ++i) {
      buffer[(HEAD + i) & mask] = items[i];
    }

    head.store((HEAD + toPush) & mask, RELEASE);
    return toPush;
  }

  size_t popMany(std::vector<T>& outItems, size_t maxItems) {
    const size_t TAIL = tail.load(RELAXED);
    const size_t HEAD = head.load(ACQUIRE);

    size_t occupied = (HEAD - TAIL) & mask;
    size_t toPop = std::min(maxItems, occupied);

    if (toPop == 0) return 0;

    for (size_t i = 0; i < toPop; ++i) {
      outItems.push_back(std::move(buffer[(TAIL + i) & mask]));
    }

    tail.store((TAIL + toPop) & mask, RELEASE);
    return toPop;
  }

  size_t size() const { return (head.load() - tail.load()) & mask; }

  LockFreeSpscQueue(const LockFreeSpscQueue&) = delete;
  LockFreeSpscQueue& operator=(const LockFreeSpscQueue&) = delete;

 private:
  size_t capacity;
  size_t mask;
  std::vector<T> buffer;
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> head{0};
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail{0};
};
