// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <thread>

#include "ipc/result_queue.hpp"
#include "utils/logger.hpp"

namespace tt::ipc {

inline void pushToken(IResultQueue& queue, uint32_t taskId, uint64_t tokenId,
                      bool finished) {
  SharedToken token{};
  token.task_id = taskId;
  token.token_id = tokenId;
  token.flags = finished ? SharedToken::FLAG_FINAL : 0u;
  if (finished) {
    TT_LOG_DEBUG("pushed final token for task_id={}", taskId);
  }
  while (!queue.push(token)) {
    std::this_thread::yield();
  }
}

inline void pushErrorToken(IResultQueue& queue, uint32_t taskId) {
  SharedToken token{};
  token.task_id = taskId;
  token.flags = SharedToken::FLAG_FINAL | SharedToken::FLAG_ERROR;
  while (!queue.push(token)) {
    std::this_thread::yield();
  }
}

}  // namespace tt::ipc
