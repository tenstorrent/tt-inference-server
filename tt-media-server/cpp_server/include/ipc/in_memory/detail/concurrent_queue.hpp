// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <utility>

namespace tt::ipc::in_memory {
namespace detail {

template <typename T>
class ConcurrentQueue {
 public:
  void push(T item) {
    {
      std::lock_guard<std::mutex> lock(mu);
      queue.push_back(std::move(item));
    }
    cv.notify_one();
  }

  bool tryPop(T& out) {
    std::lock_guard<std::mutex> lock(mu);
    if (queue.empty()) {
      return false;
    }
    out = std::move(queue.front());
    queue.pop_front();
    return true;
  }

  bool waitPop(T& out) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return shuttingDown || !queue.empty(); });
    if (queue.empty()) {
      return false;
    }
    out = std::move(queue.front());
    queue.pop_front();
    return true;
  }

  template <typename Rep, typename Period>
  bool waitPopFor(T& out, std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock<std::mutex> lock(mu);
    if (!cv.wait_for(lock, timeout, [&] { return shuttingDown || !queue.empty(); })) {
      return false;
    }
    if (queue.empty()) {
      return false;
    }
    out = std::move(queue.front());
    queue.pop_front();
    return true;
  }

  template <typename Container>
  void drainTo(Container& out) {
    std::lock_guard<std::mutex> lock(mu);
    while (!queue.empty()) {
      out.push_back(std::move(queue.front()));
      queue.pop_front();
    }
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mu);
    queue.clear();
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mu);
      shuttingDown = true;
    }
    cv.notify_all();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mu);
    return queue.empty();
  }

 private:
  mutable std::mutex mu;
  std::condition_variable cv;
  std::deque<T> queue;
  bool shuttingDown = false;
};

}  // namespace detail
}  // namespace tt::ipc::in_memory
