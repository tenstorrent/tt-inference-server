// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "profiling/tracy.hpp"

namespace tt::utils {

class ThreadPool {
 public:
  explicit ThreadPool(size_t numThreads) : stop_(false) {
    for (size_t i = 0; i < numThreads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void submit(std::function<void()> task) {
    {
      std::lock_guard lock(mutex_);
      tasks_.push(std::move(task));
    }
    cv_.notify_one();
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  TRACY_LOCKABLE(std::mutex, mutex_);
  std::condition_variable_any cv_;
  bool stop_;
};

}  // namespace tt::utils
