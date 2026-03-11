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
  reader_thread_ = std::thread([this] { reader_loop(); });
}

SpPipelineModelRunner::~SpPipelineModelRunner() {
  exit();
}

void SpPipelineModelRunner::write(const std::string& task_id,
                                          const std::vector<int64_t>& token_ids,
                                          uint32_t max_tokens) {
  device_input_.write(task_id, token_ids, max_tokens);
}

void SpPipelineModelRunner::exit() {
  if (stop_.exchange(true)) return;
  if (reader_thread_.joinable()) reader_thread_.join();
  LLM_ENGINE_LOG("sp_pipeline") << "model runner exit" << std::endl;
}

void SpPipelineModelRunner::reader_loop() {
  ReadResult read_buf;
  while (!stop_.load(std::memory_order_relaxed)) {
    if (device_output_.try_read(read_buf)) {
      llm_engine::TaskID tid = llm_engine::TaskID::ipc_deserialize(
          read_buf.taskId.data(), llm_engine::TaskID::kSerializedSize);
      uint64_t token_id = read_buf.tokenIds.empty() ? 0 : read_buf.tokenIds[0];
      llm_engine::TokenResult result(std::move(tid), token_id);
      decode_callback_(result);
    } else {
      std::this_thread::yield();
    }
  }
}

}  // namespace sp_pipeline
