// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <sstream>
#include <deque>

#include "ipc/task_queue.hpp"
#include "domain/sequence.hpp"

namespace tt::runners::llm_engine {

/**
 * In-process ITaskQueue backed by a deque of owned pointers.
 * Avoids requiring Sequence to be copy-constructible.
 * Useful for unit testing without IPC overhead.
 */
class InMemoryTaskQueue : public tt::ipc::ITaskQueue {
 public:
  void push(const tt::domain::Sequence& seq) override {
    std::ostringstream os;
    seq.serialize(os);
    std::istringstream is(os.str());
    queue.push_back(std::make_unique<tt::domain::Sequence>(tt::domain::Sequence::deserialize(is)));
  }

  std::unique_ptr<tt::domain::Sequence> tryPop() override {
    if (queue.empty()) return nullptr;
    auto seq = std::move(queue.front());
    queue.pop_front();
    return seq;
  }

  std::unique_ptr<tt::domain::Sequence> receive() override {
    if (queue.empty()) return nullptr;
    auto seq = std::move(queue.front());
    queue.pop_front();
    return seq;
  }

  bool empty() const override { return queue.empty(); }

 private:
  std::deque<std::unique_ptr<tt::domain::Sequence>> queue;
};

}  // namespace tt::runners::llm_engine
