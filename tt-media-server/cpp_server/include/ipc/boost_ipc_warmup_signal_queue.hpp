// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>

#include "ipc/boost_ipc_queue.hpp"
#include "ipc/warmup_signal_queue.hpp"

namespace tt::ipc {

constexpr const char* WARMUP_SIGNALS_QUEUE_NAME = "tt_warmup_signals";

/**
 * IWarmupSignalQueue implementation backed by the generic BoostIpcMemoryQueue.
 */
class BoostIpcWarmupSignalQueue : public IWarmupSignalQueue {
 public:
  using Queue = BoostIpcMemoryQueue<int64_t, sizeof(int64_t)>;

  /** Create queue (main process side). */
  BoostIpcWarmupSignalQueue(const std::string& name, size_t capacity)
      : queue_(std::make_unique<Queue>(name, static_cast<int>(capacity))) {}

  /** Open existing queue (worker side). */
  explicit BoostIpcWarmupSignalQueue(const std::string& name)
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

  static void remove(const std::string& name) { Queue::remove(name); }

 private:
  std::unique_ptr<Queue> queue_;
};

}  // namespace tt::ipc
