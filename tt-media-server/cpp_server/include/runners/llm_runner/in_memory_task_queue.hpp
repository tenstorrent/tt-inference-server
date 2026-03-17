// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <deque>
#include <memory>
#include <sstream>

#include "runners/llm_runner/task_queue.hpp"

namespace llm_engine {

/**
 * In-process ITaskQueue backed by a deque of owned pointers.
 * Avoids requiring Sequence to be copy-constructible.
 * Useful for unit testing without IPC overhead.
 */
class InMemoryTaskQueue : public ITaskQueue {
 public:
  void push(const Sequence& seq) override {
    std::ostringstream os;
    seq.serialize(os);
    std::istringstream is(os.str());
    queue.push_back(std::unique_ptr<Sequence>(Sequence::deserialize(is)));
  }

  Sequence* tryPop() {
    if (queue.empty()) return nullptr;
    Sequence* seq = queue.front().release();
    queue.pop_front();
    return seq;
  }

  Sequence* receive() override {
    if (queue.empty()) return nullptr;
    Sequence* seq = queue.front().release();
    queue.pop_front();
    return seq;
  }

  bool empty() const override { return queue.empty(); }

 private:
  std::deque<std::unique_ptr<Sequence>> queue;
};

}  // namespace llm_engine
