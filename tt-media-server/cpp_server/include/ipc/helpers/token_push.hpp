// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <thread>

#include "ipc/interface/result_queue.hpp"
#include "utils/logger.hpp"

namespace tt::ipc::helpers {

inline void pushToken(tt::ipc::IResultQueue& queue, uint32_t taskId,
                      uint64_t tokenId, bool finished) {
  tt::ipc::SharedToken token{};
  token.task_id = taskId;
  token.token_id = tokenId;
  token.flags = finished ? tt::ipc::SharedToken::FLAG_FINAL : 0u;
  if (finished) {
    TT_LOG_DEBUG("pushed final token for task_id={}", taskId);
  }
  while (!queue.push(token)) {
    std::this_thread::yield();
  }
}

inline void pushErrorToken(tt::ipc::IResultQueue& queue, uint32_t taskId) {
  tt::ipc::SharedToken token{};
  token.task_id = taskId;
  token.flags =
      tt::ipc::SharedToken::FLAG_FINAL | tt::ipc::SharedToken::FLAG_ERROR;
  while (!queue.push(token)) {
    std::this_thread::yield();
  }
}

}  // namespace tt::ipc::helpers
