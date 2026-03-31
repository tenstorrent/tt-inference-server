// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_prefill_runner/sp_prefill_model_runner.hpp"

#include "utils/logger.hpp"

namespace sp_prefill {

SpPrefillModelRunner::SpPrefillModelRunner(PrefillCallback callback)
    : prefillCallback(std::move(callback)),
      shmNames(),
      deviceInput(shmNames.write),
      deviceOutput(shmNames.read) {
  deviceInput.open();
  deviceOutput.open();
}

SpPrefillModelRunner::~SpPrefillModelRunner() { exit(); }

void SpPrefillModelRunner::write(const std::string& taskId,
                                 const std::vector<int64_t>& tokenIds) {
  // For prefill, we send tokens and immediately read back the single result
  deviceInput.write(taskId, tokenIds, 1);

  // Synchronously wait for the single prefill token
  tt::ipc::ReadResult readBuf;
  while (!stop.load(std::memory_order_relaxed)) {
    if (deviceOutput.tryRead(readBuf)) {
      llm_engine::TaskID tid = llm_engine::TaskID::ipcDeserialize(
          readBuf.taskId.data(), llm_engine::TaskID::K_SERIALIZED_SIZE);
      uint64_t tokenId = readBuf.tokenIds.empty() ? 0 : readBuf.tokenIds[0];
      llm_engine::TokenResult result(std::move(tid), tokenId);
      prefillCallback(result);
      break;  // Got the token, done
    }
    std::this_thread::yield();
  }
}

void SpPrefillModelRunner::exit() {
  stop.store(true, std::memory_order_relaxed);
  TT_LOG_INFO("SpPrefillModelRunner: Model runner exit");
}

}  // namespace sp_prefill
