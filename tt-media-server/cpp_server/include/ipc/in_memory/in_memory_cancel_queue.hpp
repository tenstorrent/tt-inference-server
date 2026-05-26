// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <vector>

#include "ipc/in_memory/detail/concurrent_queue.hpp"
#include "ipc/interface/cancel_queue.hpp"

namespace tt::ipc::in_memory {

class CancelQueue : public tt::ipc::ICancelQueue {
 public:
  void push(uint32_t taskId) override { queue.push(taskId); }

  void tryPopAll(std::vector<uint32_t>& out) override { queue.drainTo(out); }

  void remove() override { queue.clear(); }

 private:
  tt::ipc::in_memory::detail::ConcurrentQueue<uint32_t> queue;
};

}  // namespace tt::ipc::in_memory
