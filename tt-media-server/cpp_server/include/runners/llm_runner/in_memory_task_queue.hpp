// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <deque>
#include <memory>
#include <sstream>

#include "runners/llm_runner/task_queue.hpp"

namespace tt::runners::llm_engine {

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
    queue_.push_back(std::make_unique<Sequence>(Sequence::deserialize(is)));
  }

  std::unique_ptr<Sequence> tryPop() override {
    if (queue_.empty()) return nullptr;
    auto seq = std::move(queue_.front());
    queue_.pop_front();
    return std::move(seq);
  }

  std::unique_ptr<Sequence> receive() override {
    if (queue_.empty()) return nullptr;
    auto seq = std::move(queue_.front());
    queue_.pop_front();
    return std::move(seq);
  }

  bool empty() const override { return queue_.empty(); }

 private:
  std::deque<std::unique_ptr<Sequence>> queue_;
};

}  // namespace tt::runners::llm_engine
