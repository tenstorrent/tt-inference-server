// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/sp_prefill_runner/sp_prefill_model_runner.hpp"

#include "utils/logger.hpp"

namespace sp_prefill {

SpPrefillModelRunner::SpPrefillModelRunner()
    : shmNames(), deviceInput(shmNames.write), deviceOutput(shmNames.read) {
  deviceInput.open();
  deviceOutput.open();
}

SpPrefillModelRunner::~SpPrefillModelRunner() { exit(); }

std::optional<tt::runners::llm_engine::TokenResult>
SpPrefillModelRunner::forward(uint32_t taskId,
                              const std::vector<int64_t>& tokenIds) {
  TT_LOG_DEBUG(
      "SpPrefillModelRunner: Writing into shared memory input task_id={}, "
      "token count={}",
      taskId, tokenIds.size());
  deviceInput.write(taskId, tokenIds, 1);

  tt::ipc::ReadResult readBuf;
  TT_LOG_DEBUG(
      "SpPrefillModelRunner: Reading from shared memory output task_id={}",
      taskId);
  while (!stop.load(std::memory_order_relaxed)) {
    if (deviceOutput.tryRead(readBuf)) {
      uint64_t tokenId = readBuf.tokenIds.empty() ? 0 : readBuf.tokenIds[0];
      TT_LOG_DEBUG(
          "SpPrefillModelRunner: Read from shared memory output task_id={}, "
          "token_id={}, token count={}",
          taskId, tokenId, readBuf.tokenIds.size());
      return tt::runners::llm_engine::TokenResult(readBuf.taskId, tokenId);
    }
    TT_LOG_DEBUG("SpPrefillModelRunner: Shared memory read failed");
    std::this_thread::yield();
  }
  TT_LOG_DEBUG(
      "SpPrefillModelRunner: forward exiting without Shared memory response "
      "(stop)",
      taskId);
  return std::nullopt;
}

void SpPrefillModelRunner::exit() {
  stop.store(true, std::memory_order_relaxed);
  TT_LOG_INFO("[SpPrefillModelRunner] Model runner exit");
}

}  // namespace sp_prefill
