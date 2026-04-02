// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_model_runner.hpp"

#include "runners/llm_runner/debug.hpp"

namespace sp_pipeline {

SpPipelineModelRunner::SpPipelineModelRunner(DecodeCallback callback)
    : decodeCallback(std::move(callback)),
      shmNames(),
      deviceInput(shmNames.write),
      deviceOutput(shmNames.read) {
  deviceInput.open();
  deviceOutput.open();
  readerThread = std::thread([this] { readerLoop(); });
}

SpPipelineModelRunner::~SpPipelineModelRunner() { exit(); }

void SpPipelineModelRunner::write(const std::string& taskId,
                                  const std::vector<int64_t>& tokenIds,
                                  uint32_t maxTokens, RequestPhase /*phase*/,
                                  bool fastMode) {
  // TODO: propagate phase to the shared-memory protocol for disaggregated mode.
  deviceInput.write(taskId, tokenIds, maxTokens, fastMode);
}

void SpPipelineModelRunner::exit() {
  if (stop.exchange(true)) return;
  if (readerThread.joinable()) readerThread.join();
  LLM_ENGINE_LOG("sp_pipeline") << "model runner exit" << std::endl;
}

void SpPipelineModelRunner::readerLoop() {
  tt::ipc::ReadResult readBuf;
  while (!stop.load(std::memory_order_relaxed)) {
    if (deviceOutput.tryRead(readBuf)) {
      llm_engine::TaskID tid = llm_engine::TaskID::ipcDeserialize(
          readBuf.taskId.data(), llm_engine::TaskID::K_SERIALIZED_SIZE);
      uint64_t tokenId = readBuf.tokenIds.empty() ? 0 : readBuf.tokenIds[0];
      llm_engine::TokenResult result(std::move(tid), tokenId);
      decodeCallback(result);
    } else {
      std::this_thread::yield();
    }
  }
}

}  // namespace sp_pipeline
