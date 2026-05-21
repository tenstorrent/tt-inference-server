// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include "ipc/interface/cancel_queue.hpp"

namespace tt::ipc::in_memory {

class CancelQueue : public tt::ipc::ICancelQueue {
 public:
  void push(uint32_t taskId) override {
    std::lock_guard<std::mutex> lock(mu);
    items.push_back(taskId);
  }

  void tryPopAll(std::vector<uint32_t>& out) override {
    std::lock_guard<std::mutex> lock(mu);
    out.insert(out.end(), items.begin(), items.end());
    items.clear();
  }

  void remove() override {
    std::lock_guard<std::mutex> lock(mu);
    items.clear();
  }

 private:
  std::mutex mu;
  std::vector<uint32_t> items;
};

}  // namespace tt::ipc::in_memory
