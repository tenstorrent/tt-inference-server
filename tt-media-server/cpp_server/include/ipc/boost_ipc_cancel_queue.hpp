// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>

#include "ipc/boost_ipc_queue.hpp"
#include "ipc/cancel_queue.hpp"

namespace tt::ipc {

/**
 * ICancelQueue implementation backed by the generic BoostIpcMemoryQueue.
 * One queue per worker.
 */
class BoostIpcCancelQueue : public ICancelQueue {
 public:
  using Queue = BoostIpcMemoryQueue<uint32_t, sizeof(uint32_t)>;

  /** Create a new queue (main process). */
  BoostIpcCancelQueue(const std::string& name, size_t capacity)
      : queue_(std::make_unique<Queue>(name, static_cast<int>(capacity))) {}

  /** Open an existing queue (worker process). */
  explicit BoostIpcCancelQueue(const std::string& name)
      : queue_(Queue::openExisting(name)) {}

  void push(uint32_t taskId) override {
    if (!queue_->tryPush(taskId)) {
      TT_LOG_WARN("[CancelQueue] Queue full, dropping cancel for task_id={}",
                  taskId);
    }
  }

  void tryPopAll(std::vector<uint32_t>& out) override {
    queue_->tryPopAll(out);
  }

  void remove() override { queue_->remove(); }

  static void removeByName(const std::string& name) { Queue::remove(name); }

 private:
  std::unique_ptr<Queue> queue_;
};

}  // namespace tt::ipc
