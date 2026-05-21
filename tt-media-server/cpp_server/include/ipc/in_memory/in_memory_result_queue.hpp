// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>

#include "ipc/interface/result_queue.hpp"

namespace tt::ipc::in_memory {

class ResultQueue : public tt::ipc::IResultQueue {
 public:
  bool push(const tt::ipc::SharedToken& token) override {
    {
      std::lock_guard<std::mutex> lock(mu);
      queue.push_back(token);
    }
    cv.notify_one();
    return true;
  }

  bool tryPop(tt::ipc::SharedToken& out) override {
    std::lock_guard<std::mutex> lock(mu);
    if (queue.empty()) {
      return false;
    }
    out = queue.front();
    queue.pop_front();
    return true;
  }

  bool blockingPop(tt::ipc::SharedToken& out) override {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return shuttingDown || !queue.empty(); });
    if (queue.empty()) {
      return false;
    }
    out = queue.front();
    queue.pop_front();
    return !out.isDone();
  }

  void shutdown() override {
    {
      std::lock_guard<std::mutex> lock(mu);
      shuttingDown = true;
    }
    cv.notify_all();
  }

  bool waitPopFor(tt::ipc::SharedToken& out, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu);
    if (!cv.wait_for(lock, timeout, [&] { return !queue.empty(); })) {
      return false;
    }
    out = queue.front();
    queue.pop_front();
    return true;
  }

 private:
  std::mutex mu;
  std::condition_variable cv;
  std::deque<tt::ipc::SharedToken> queue;
  bool shuttingDown = false;
};

}  // namespace tt::ipc::in_memory
