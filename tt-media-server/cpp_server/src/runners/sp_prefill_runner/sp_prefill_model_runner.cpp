// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_prefill_runner/sp_prefill_model_runner.hpp"

#include "utils/logger.hpp"

namespace sp_prefill {

SpPrefillModelRunner::SpPrefillModelRunner()
    : shmNames(), deviceInput(shmNames.write), deviceOutput(shmNames.read) {
  deviceInput.open();
  deviceOutput.open();
}

SpPrefillModelRunner::~SpPrefillModelRunner() { exit(); }

std::optional<llm_engine::TokenResult> SpPrefillModelRunner::forward(
    const std::string& taskId, const std::vector<int64_t>& tokenIds) {
  deviceInput.write(taskId, tokenIds, 1);

  tt::ipc::ReadResult readBuf;
  while (!stop.load(std::memory_order_relaxed)) {
    if (deviceOutput.tryRead(readBuf)) {
      llm_engine::TaskID tid = tt::domain::TaskIDGenerator::deserialize(
          readBuf.taskId.data(), tt::domain::TaskIDGenerator::K_SERIALIZED_SIZE);
      uint64_t tokenId = readBuf.tokenIds.empty() ? 0 : readBuf.tokenIds[0];
      return llm_engine::TokenResult(std::move(tid), tokenId);
    }
    std::this_thread::yield();
  }
  return std::nullopt;
}

void SpPrefillModelRunner::exit() {
  stop.store(true, std::memory_order_relaxed);
  TT_LOG_INFO("SpPrefillModelRunner: Model runner exit");
}

}  // namespace sp_prefill
