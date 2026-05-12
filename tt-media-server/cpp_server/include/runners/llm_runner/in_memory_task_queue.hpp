// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <deque>
#include <memory>
#include <sstream>

#include "domain/llm/sequence.hpp"
#include "ipc/task_queue.hpp"

namespace tt::runners::llm_engine {

using namespace tt::domain::llm;

/**
 * In-process ITaskQueue backed by a deque of owned pointers.
 * Avoids requiring Sequence to be copy-constructible.
 * Useful for unit testing without IPC overhead.
 */
class InMemoryTaskQueue : public tt::ipc::ITaskQueue {
 public:
  void push(const tt::domain::llm::Sequence& seq) override {
    std::ostringstream os;
    seq.serialize(os);
    std::istringstream is(os.str());
    queue.push_back(std::make_unique<tt::domain::llm::Sequence>(
        tt::domain::llm::Sequence::deserialize(is)));
  }

  std::unique_ptr<tt::domain::llm::Sequence> tryPop() override {
    if (queue.empty()) return nullptr;
    auto seq = std::move(queue.front());
    queue.pop_front();
    return seq;
  }

  std::unique_ptr<tt::domain::llm::Sequence> receive() override {
    if (queue.empty()) return nullptr;
    auto seq = std::move(queue.front());
    queue.pop_front();
    return seq;
  }

  bool empty() const override { return queue.empty(); }

 private:
  std::deque<std::unique_ptr<tt::domain::llm::Sequence>> queue;
};

}  // namespace tt::runners::llm_engine
