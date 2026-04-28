// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>

#include "config/runner_config.hpp"
#include "ipc/boost_ipc_queue.hpp"
#include "ipc/task_queue.hpp"

namespace tt::ipc {

/**
 * ITaskQueue implementation backed by the generic BoostIpcMemoryQueue.
 */
class BoostIpcTaskQueue : public tt::ipc::ITaskQueue {
 public:
  static constexpr size_t MAX_SEQUENCE_NON_TOKEN_BYTES = 4096;
  static constexpr size_t MAX_MSG_SIZE =
      tt::config::LLMConfig::MAX_INPUT_TOKENS * sizeof(int64_t) +
      MAX_SEQUENCE_NON_TOKEN_BYTES;

  using Queue = BoostIpcMemoryQueue<tt::domain::Sequence, MAX_MSG_SIZE>;

  /** Create a new queue (main process). */
  BoostIpcTaskQueue(const std::string& name, int capacity)
      : queue_(std::make_unique<Queue>(name, capacity)) {}

  /** Open an existing queue (worker process). */
  explicit BoostIpcTaskQueue(const std::string& name)
      : queue_(Queue::openExisting(name)) {}

  void push(const tt::domain::Sequence& seq) override { queue_->push(seq); }

  std::unique_ptr<tt::domain::Sequence> tryPop() override {
    tt::domain::Sequence seq(0, 1, {});
    if (!queue_->tryPop(seq)) return nullptr;
    return std::make_unique<tt::domain::Sequence>(std::move(seq));
  }

  std::unique_ptr<tt::domain::Sequence> receive() override {
    tt::domain::Sequence seq(0, 1, {});
    queue_->receive(seq);
    return std::make_unique<tt::domain::Sequence>(std::move(seq));
  }

  bool empty() const override { return queue_->empty(); }

  static void remove(const std::string& name) { Queue::remove(name); }

 private:
  std::unique_ptr<Queue> queue_;
};

}  // namespace tt::ipc
