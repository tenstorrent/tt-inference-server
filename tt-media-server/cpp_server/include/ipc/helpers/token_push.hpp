// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <thread>

#include "ipc/interface/result_queue.hpp"
#include "utils/logger.hpp"

namespace tt::ipc::helpers {

inline void pushToken(tt::ipc::IResultQueue& queue, uint32_t taskId,
                      uint64_t tokenId, uint32_t flag = 0,
                      uint32_t specAccepts = 0, uint32_t specRejects = 0) {
  tt::ipc::SharedToken token{};
  token.task_id = taskId;
  token.token_id = tokenId;
  token.flags = flag;
  token.spec_accepts = specAccepts;
  token.spec_rejects = specRejects;
  if (flag & tt::ipc::SharedToken::FLAG_FINAL) {
    TT_LOG_DEBUG("pushed final token for task_id={}", taskId);
  }
  while (!queue.push(token)) {
    std::this_thread::yield();
  }
}

inline void pushErrorToken(tt::ipc::IResultQueue& queue, uint32_t taskId,
                           bool timeout = false) {
  tt::ipc::SharedToken token{};
  token.task_id = taskId;
  token.flags =
      tt::ipc::SharedToken::FLAG_FINAL | tt::ipc::SharedToken::FLAG_ERROR;
  if (timeout) {
    token.flags |= tt::ipc::SharedToken::FLAG_TIMEOUT;
  }
  while (!queue.push(token)) {
    std::this_thread::yield();
  }
}

}  // namespace tt::ipc::helpers
