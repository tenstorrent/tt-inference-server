// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_model_runner.hpp"

#include "runners/llm_runner/debug.hpp"

namespace sp_pipeline {

SpPipelineModelRunner::SpPipelineModelRunner(DecodeCallback callback)
    : decode_callback_(std::move(callback)),
      shm_names_(),
      device_input_(shm_names_.write),
      device_output_(shm_names_.read) {
  device_input_.open();
  device_output_.open();
  reader_thread_ = std::thread([this] { readerLoop(); });
}

SpPipelineModelRunner::~SpPipelineModelRunner() { exit(); }

void SpPipelineModelRunner::write(const std::string& taskId,
                                  const std::vector<int64_t>& tokenIds,
                                  uint32_t maxTokens) {
  device_input_.write(taskId, tokenIds, maxTokens);
}

void SpPipelineModelRunner::exit() {
  if (stop_.exchange(true)) return;
  if (reader_thread_.joinable()) reader_thread_.join();
  LLM_ENGINE_LOG("sp_pipeline") << "model runner exit" << std::endl;
}

void SpPipelineModelRunner::readerLoop() {
  ReadResult readBuf;
  while (!stop_.load(std::memory_order_relaxed)) {
    if (device_output_.tryRead(readBuf)) {
      llm_engine::TaskID tid = llm_engine::TaskID::ipcDeserialize(
          readBuf.taskId.data(), llm_engine::TaskID::K_SERIALIZED_SIZE);
      uint64_t tokenId = readBuf.tokenIds.empty() ? 0 : readBuf.tokenIds[0];
      llm_engine::TokenResult result(std::move(tid), tokenId);
      decode_callback_(result);
    } else {
      std::this_thread::yield();
    }
  }
}

}  // namespace sp_pipeline
