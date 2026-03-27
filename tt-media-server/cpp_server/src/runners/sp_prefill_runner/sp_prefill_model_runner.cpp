// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_prefill_runner/sp_prefill_model_runner.hpp"

#include "runners/llm_runner/debug.hpp"

namespace sp_prefill {

SpPrefillModelRunner::SpPrefillModelRunner(PrefillCallback callback)
    : prefillCallback(std::move(callback)),
      shmNames(),
      deviceInput(shmNames.write),
      deviceOutput(shmNames.read) {
  deviceInput.open();
  deviceOutput.open();
  readerThread = std::thread([this] { readerLoop(); });
}

SpPrefillModelRunner::~SpPrefillModelRunner() { exit(); }

void SpPrefillModelRunner::write(const std::string& taskId,
                                 const std::vector<int64_t>& tokenIds) {
  // For prefill, we always send tokens but don't need maxTokens since we only
  // get 1 token back
  deviceInput.write(taskId, tokenIds, 1);
}

void SpPrefillModelRunner::exit() {
  if (stop.exchange(true)) return;
  if (readerThread.joinable()) readerThread.join();
  LLM_ENGINE_LOG("sp_prefill") << "model runner exit" << std::endl;
}

void SpPrefillModelRunner::readerLoop() {
  sp_pipeline::ReadResult readBuf;
  while (!stop.load(std::memory_order_relaxed)) {
    if (deviceOutput.tryRead(readBuf)) {
      llm_engine::TaskID tid = llm_engine::TaskID::ipcDeserialize(
          readBuf.taskId.data(), llm_engine::TaskID::K_SERIALIZED_SIZE);
      uint64_t tokenId = readBuf.tokenIds.empty() ? 0 : readBuf.tokenIds[0];
      llm_engine::TokenResult result(std::move(tid), tokenId);
      prefillCallback(result);
    } else {
      std::this_thread::yield();
    }
  }
}

}  // namespace sp_prefill
