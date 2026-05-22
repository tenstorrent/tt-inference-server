// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>

#include "ipc/in_memory/detail/concurrent_queue.hpp"
#include "ipc/interface/result_queue.hpp"

namespace tt::ipc::in_memory {

class ResultQueue : public tt::ipc::IResultQueue {
 public:
  bool push(const tt::ipc::SharedToken& token) override {
    queue.push(token);
    return true;
  }

  bool tryPop(tt::ipc::SharedToken& out) override { return queue.tryPop(out); }

  bool blockingPop(tt::ipc::SharedToken& out) override {
    if (!queue.waitPop(out)) {
      return false;
    }
    return !out.isDone();
  }

  void shutdown() override { queue.shutdown(); }

  bool waitPopFor(tt::ipc::SharedToken& out,
                  std::chrono::milliseconds timeout) {
    return queue.waitPopFor(out, timeout);
  }

 private:
  tt::ipc::in_memory::detail::ConcurrentQueue<tt::ipc::SharedToken> queue;
};

}  // namespace tt::ipc::in_memory
