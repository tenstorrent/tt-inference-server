// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>

#include "ipc/boost/memory_queue.hpp"
#include "ipc/interface/cancel_queue.hpp"

namespace tt::ipc::boost {

/**
 * ICancelQueue implementation backed by the generic boost MemoryQueue.
 * One queue per worker.
 */
class CancelQueue : public tt::ipc::ICancelQueue {
 public:
  using Queue = MemoryQueue<uint32_t, sizeof(uint32_t)>;

  /** Create a new queue (main process). */
  CancelQueue(const std::string& name, size_t capacity)
      : queue_(std::make_unique<Queue>(name, static_cast<int>(capacity))) {}

  /** Open an existing queue (worker process). */
  explicit CancelQueue(const std::string& name)
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

}  // namespace tt::ipc::boost
