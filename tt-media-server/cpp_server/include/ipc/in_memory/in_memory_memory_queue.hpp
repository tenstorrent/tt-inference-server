// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "ipc/interface/memory_queue.hpp"
#include "utils/concurrent_queue.hpp"

namespace tt::ipc::in_memory {

class MemoryRequestQueue : public tt::ipc::IMemoryRequestQueue {
 public:
  void push(const tt::domain::ManageMemoryTask& task) override {
    queue.push(task);
  }

  bool tryPop(tt::domain::ManageMemoryTask& out) override {
    return queue.tryPop(out);
  }

 private:
  tt::utils::BlockingQueue<tt::domain::ManageMemoryTask> queue;
};

class MemoryResultQueue : public tt::ipc::IMemoryResultQueue {
 public:
  void push(const tt::domain::ManageMemoryResult& result) override {
    queue.push(result);
  }

  bool waitPop(tt::domain::ManageMemoryResult& out) override {
    return queue.waitPop(out);
  }

 private:
  tt::utils::BlockingQueue<tt::domain::ManageMemoryResult> queue;
};

}  // namespace tt::ipc::in_memory
