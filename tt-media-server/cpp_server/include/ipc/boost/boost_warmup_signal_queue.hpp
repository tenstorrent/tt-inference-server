// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>

#include "ipc/boost/memory_queue.hpp"
#include "ipc/interface/warmup_signal_queue.hpp"

namespace tt::ipc::boost {

/**
 * IWarmupSignalQueue implementation backed by the generic boost MemoryQueue.
 */
class WarmupSignalQueue : public tt::ipc::IWarmupSignalQueue {
 public:
  using Queue = MemoryQueue<int64_t, sizeof(int64_t)>;

  /** Create queue (main process side). */
  WarmupSignalQueue(const std::string& name, size_t capacity)
      : queue_(std::make_unique<Queue>(name, static_cast<int>(capacity))) {}

  /** Open existing queue (worker side). */
  explicit WarmupSignalQueue(const std::string& name)
      : queue_(Queue::openExisting(name)) {}

  void sendReady(int workerId) override {
    queue_->push(static_cast<int64_t>(workerId));
  }

  int receive() override {
    int64_t payload = 0;
    queue_->receive(payload);
    return static_cast<int>(payload);
  }

  void remove() override { queue_->remove(); }

 private:
  std::unique_ptr<Queue> queue_;
};

}  // namespace tt::ipc::boost
