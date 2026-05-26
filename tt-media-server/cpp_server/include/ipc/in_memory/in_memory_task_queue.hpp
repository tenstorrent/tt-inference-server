// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <sstream>

#include "domain/llm/sequence.hpp"
#include "ipc/in_memory/detail/concurrent_queue.hpp"
#include "ipc/interface/task_queue.hpp"

namespace tt::ipc::in_memory {

/**
 * In-process ITaskQueue backed by a deque of owned pointers.
 * Avoids requiring Sequence to be copy-constructible.
 * Useful for unit testing without IPC overhead.
 */
class TaskQueue : public tt::ipc::ITaskQueue {
 public:
  void push(const tt::domain::llm::Sequence& seq) override {
    std::ostringstream os;
    seq.serialize(os);
    std::istringstream is(os.str());
    queue.push(std::make_unique<tt::domain::llm::Sequence>(
        tt::domain::llm::Sequence::deserialize(is)));
  }

  std::unique_ptr<tt::domain::llm::Sequence> tryPop() override {
    std::unique_ptr<tt::domain::llm::Sequence> seq;
    if (!queue.tryPop(seq)) return nullptr;
    return seq;
  }

  std::unique_ptr<tt::domain::llm::Sequence> receive() override {
    std::unique_ptr<tt::domain::llm::Sequence> seq;
    // Keep in-memory test semantics non-blocking to avoid deadlocks in loops
    // that expect polling behavior (e.g. scheduler/runner tests).
    if (!queue.tryPop(seq)) return nullptr;
    return seq;
  }

  bool empty() const override { return queue.empty(); }

 private:
  tt::utils::BlockingQueue<std::unique_ptr<tt::domain::llm::Sequence>> queue;
};

}  // namespace tt::ipc::in_memory
