// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>

#include "ipc/boost_ipc_queue.hpp"
#include "ipc/result_queue.hpp"

namespace tt::ipc {

constexpr int RESULT_QUEUE_CAPACITY = 65536;

/**
 * IResultQueue implementation backed by BoostIpcMemoryQueue.
 * Replaces the custom POSIX-shm / futex TokenRingBuffer with the same
 * Boost.Interprocess transport used by the task and cancel queues.
 */
class BoostIpcResultQueue : public IResultQueue {
 public:
  using Queue = BoostIpcMemoryQueue<SharedToken, sizeof(SharedToken)>;

  /** Create a new queue (main process). */
  BoostIpcResultQueue(const std::string& name, int capacity)
      : queue_(std::make_unique<Queue>(name, capacity)) {}

  /** Open an existing queue (worker process). */
  explicit BoostIpcResultQueue(const std::string& name)
      : queue_(Queue::openExisting(name)) {}

  bool push(const SharedToken& token) override {
    return queue_->tryPush(token);
  }

  bool tryPop(SharedToken& out) override { return queue_->tryPop(out); }

  /**
   * Block until a token is available or shutdown is signaled.
   * Uses Boost's fully-blocking receive() -- zero CPU burn, instant wake
   * on data.  Shutdown is signalled by pushing a poison-pill token
   * (FLAG_DONE) which causes this method to return false.
   */
  bool blockingPop(SharedToken& out) override {
    queue_->receive(out);
    return !out.isDone();
  }

  /** Push a poison-pill token so any thread blocked in blockingPop wakes. */
  void shutdown() override {
    SharedToken pill{};
    pill.flags = SharedToken::FLAG_DONE;
    queue_->push(pill);
  }

  void remove() override { queue_->remove(); }

 private:
  std::unique_ptr<Queue> queue_;
};

}  // namespace tt::ipc
