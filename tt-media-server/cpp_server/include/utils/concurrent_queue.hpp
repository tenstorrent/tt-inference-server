// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#pragma once

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
    pending_.push_back(value);
  }

  std::vector<T> drain() {
    std::lock_guard lock(mutex);
    std::vector<T> out;
    out.swap(pending_);
    return out;
  }

  size_t size() {
    std::lock_guard lock(mutex);
    return pending_.size();
  }

  ConcurrentQueue(const ConcurrentQueue&) = delete;
  ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;

 private:
  std::vector<T> pending_;
  TRACY_LOCKABLE(std::mutex, mutex);
};
