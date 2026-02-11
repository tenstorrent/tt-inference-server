// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "llm_engine/engine/task_queue.hpp"

#include <deque>

namespace llm_engine {

/**
 * In-process ITaskQueue backed by a simple deque.
 * Stores copies of sequences; try_pop returns a heap-allocated copy.
 * Useful for unit testing without IPC overhead.
 */
class InMemoryTaskQueue : public ITaskQueue {
 public:
  void push(const Sequence& seq) override { queue_.push_back(seq); }

  Sequence* try_pop() override {
    if (queue_.empty()) return nullptr;
    auto* seq = new Sequence(std::move(queue_.front()));
    queue_.pop_front();
    return seq;
  }

  bool empty() const override { return queue_.empty(); }

 private:
  std::deque<Sequence> queue_;
};

}  // namespace llm_engine
