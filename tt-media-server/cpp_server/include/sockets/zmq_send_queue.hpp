// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>

namespace tt::sockets {

template <typename T>
class ZmqSendQueue {
 public:
  template <typename ShouldAccept>
  bool pushIf(std::shared_ptr<T> request, ShouldAccept shouldAccept) {
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      if (!shouldAccept()) {
        return false;
      }
      items.push_back(std::move(request));
    }
    hasItems = true;
    wakeCv.notify_one();
    return true;
  }

  bool tryPop(std::shared_ptr<T>& request) {
    std::lock_guard<std::mutex> lock(queueMutex);
    if (items.empty()) {
      hasItems = false;
      return false;
    }
    request = std::move(items.front());
    items.pop_front();
    return true;
  }

  template <typename Rep, typename Period, typename ShouldStop>
  void waitForWork(std::chrono::duration<Rep, Period> timeout,
                   ShouldStop shouldStop) {
    std::unique_lock<std::mutex> lock(wakeMutex);
    // Keep the predicate off queueMutex to avoid the ZMQ IO-thread TSan
    // deadlock false positive that this queue is designed around.
    wakeCv.wait_for(lock, timeout, [this, &shouldStop] {
      return hasItems.load() || shouldStop();
    });
  }

  void notifyStopped() {
    hasItems = false;
    wakeCv.notify_all();
  }

 private:
  std::mutex queueMutex;
  std::mutex wakeMutex;
  std::condition_variable wakeCv;
  std::atomic<bool> hasItems{false};
  std::deque<std::shared_ptr<T>> items;
};

}  // namespace tt::sockets
