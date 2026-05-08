// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Decorator around any ITaskQueue that records every pushed Sequence.
// Thread-safe. waitForNext() blocks until a Sequence is available.

#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "domain/llm/sequence.hpp"
#include "ipc/task_queue.hpp"

namespace tt::test {

class RecordingTaskQueue : public ipc::ITaskQueue {
 public:
  explicit RecordingTaskQueue(std::shared_ptr<ipc::ITaskQueue> inner)
      : inner_(std::move(inner)) {}

  // ITaskQueue — record a serialized clone, then forward.
  void push(const domain::llm::Sequence& seq) override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      // Sequence is not copyable; round-trip through serialize/deserialize.
      std::ostringstream os;
      seq.serialize(os);
      std::istringstream is(os.str());
      recorded_.push_back(
          std::make_unique<domain::llm::Sequence>(
              domain::llm::Sequence::deserialize(is)));
    }
    cv_.notify_one();
    inner_->push(seq);
  }

  // Delegate non-recording methods to inner queue.
  std::unique_ptr<domain::llm::Sequence> tryPop() override {
    return inner_->tryPop();
  }
  std::unique_ptr<domain::llm::Sequence> receive() override {
    return inner_->receive();
  }
  bool empty() const override { return inner_->empty(); }

  // Block until a Sequence is available, then return it.
  std::unique_ptr<domain::llm::Sequence> waitForNext(
      std::chrono::milliseconds timeout = std::chrono::seconds(10)) {
    std::unique_lock<std::mutex> lock(mu_);
    bool ok = cv_.wait_for(lock, timeout, [&] { return !recorded_.empty(); });
    if (!ok) {
      throw std::runtime_error(
          "RecordingTaskQueue: timed out waiting for task queue push");
    }
    auto seq = std::move(recorded_.front());
    recorded_.pop_front();
    return seq;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return recorded_.size();
  }

 private:
  std::shared_ptr<ipc::ITaskQueue> inner_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::unique_ptr<domain::llm::Sequence>> recorded_;
};

}  // namespace tt::test
