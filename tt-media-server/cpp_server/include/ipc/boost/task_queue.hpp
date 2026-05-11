// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>

#include "config/runner_config.hpp"
#include "ipc/boost/memory_queue.hpp"
#include "ipc/interface/task_queue.hpp"

namespace tt::ipc::boost {

/**
 * ITaskQueue implementation backed by the generic boost MemoryQueue.
 */
class TaskQueue : public tt::ipc::ITaskQueue {
 public:
  static constexpr size_t MAX_SEQUENCE_NON_TOKEN_BYTES = 4096;
  static constexpr size_t MAX_MSG_SIZE =
      tt::config::LLMConfig::MAX_INPUT_TOKENS * sizeof(int64_t) +
      MAX_SEQUENCE_NON_TOKEN_BYTES;

  using Queue = MemoryQueue<tt::domain::llm::Sequence, MAX_MSG_SIZE>;

  /** Create a new queue (main process). */
  TaskQueue(const std::string& name, int capacity)
      : queue_(std::make_unique<Queue>(name, capacity)) {}

  /** Open an existing queue (worker process). */
  explicit TaskQueue(const std::string& name)
      : queue_(Queue::openExisting(name)) {}

  void push(const tt::domain::llm::Sequence& seq) override {
    queue_->push(seq);
  }

  std::unique_ptr<tt::domain::llm::Sequence> tryPop() override {
    tt::domain::llm::Sequence seq(0, 1, {});
    if (!queue_->tryPop(seq)) return nullptr;
    return std::make_unique<tt::domain::llm::Sequence>(std::move(seq));
  }

  std::unique_ptr<tt::domain::llm::Sequence> receive() override {
    tt::domain::llm::Sequence seq(0, 1, {});
    queue_->receive(seq);
    return std::make_unique<tt::domain::llm::Sequence>(std::move(seq));
  }

  bool empty() const override { return queue_->empty(); }

  static void remove(const std::string& name) { Queue::remove(name); }

 private:
  std::unique_ptr<Queue> queue_;
};

}  // namespace tt::ipc::boost
