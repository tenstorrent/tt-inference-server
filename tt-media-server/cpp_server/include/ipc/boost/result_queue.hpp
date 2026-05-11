// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>

#include "ipc/boost/memory_queue.hpp"
#include "ipc/interface/result_queue.hpp"

namespace tt::ipc::boost {

constexpr int RESULT_QUEUE_CAPACITY = 65536;

/**
 * IResultQueue implementation backed by the generic boost MemoryQueue.
 * Replaces the custom POSIX-shm / futex TokenRingBuffer with the same
 * Boost.Interprocess transport used by the task and cancel queues.
 */
class ResultQueue : public tt::ipc::IResultQueue {
 public:
  using Queue = MemoryQueue<tt::ipc::SharedToken, sizeof(tt::ipc::SharedToken)>;

  /** Create a new queue (main process). */
  ResultQueue(const std::string& name, int capacity)
      : queue_(std::make_unique<Queue>(name, capacity)) {}

  /** Open an existing queue (worker process). */
  explicit ResultQueue(const std::string& name)
      : queue_(Queue::openExisting(name)) {}

  bool push(const tt::ipc::SharedToken& token) override {
    return queue_->tryPush(token);
  }

  bool tryPop(tt::ipc::SharedToken& out) override {
    return queue_->tryPop(out);
  }

  /**
   * Block until a token is available or shutdown is signaled.
   * Uses Boost's fully-blocking receive() -- zero CPU burn, instant wake
   * on data.  Shutdown is signalled by pushing a poison-pill token
   * (FLAG_DONE) which causes this method to return false.
   */
  bool blockingPop(tt::ipc::SharedToken& out) override {
    queue_->receive(out);
    return !out.isDone();
  }

  /** Push a poison-pill token so any thread blocked in blockingPop wakes. */
  void shutdown() override {
    tt::ipc::SharedToken pill{};
    pill.flags = tt::ipc::SharedToken::FLAG_DONE;
    queue_->push(pill);
  }

  void remove() override { queue_->remove(); }

 private:
  std::unique_ptr<Queue> queue_;
};

}  // namespace tt::ipc::boost
