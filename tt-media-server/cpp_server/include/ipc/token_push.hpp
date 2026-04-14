// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <thread>

#include "ipc/token_ring_buffer.hpp"
#include "utils/logger.hpp"

namespace tt::ipc {

template <size_t N>
void pushToken(TokenRingBuffer<N>& queue, uint32_t taskId, uint64_t tokenId,
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

template <size_t N>
void pushErrorToken(TokenRingBuffer<N>& queue, uint32_t taskId) {
  SharedToken token{};
  token.task_id = taskId;
  token.flags = SharedToken::FLAG_FINAL | SharedToken::FLAG_ERROR;
  while (!queue.push(token)) {
    std::this_thread::yield();
  }
}

}  // namespace tt::ipc
