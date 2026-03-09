// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_model_runner.hpp"
#include "runners/llm_runner/debug.hpp"

namespace sp_pipeline {

// -- DecodeQueue --------------------------------------------------------------

void DecodeQueue::push(const llm_engine::TokenResult& result) {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.push_back(result);
}

std::vector<llm_engine::TokenResult> DecodeQueue::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<llm_engine::TokenResult> out;
  out.swap(pending_);
  return out;
}

// -- SpPipelineModelRunner ----------------------------------------------------

SpPipelineModelRunner::SpPipelineModelRunner(DecodeCallback callback)
    : decode_callback_(std::move(callback)),
      shm_names_(),
      device_input_(shm_names_.write),
      device_output_(shm_names_.read) {
  LLM_ENGINE_LOG("sp_pipeline") << "Using shared memory: C2P="
                                << shm_names_.write
                                << " P2C=" << shm_names_.read << std::endl;
  device_input_.open();
  device_output_.open();
  reader_thread_ = std::thread([this] { reader_loop(); });
}

SpPipelineModelRunner::~SpPipelineModelRunner() {
  exit();
}

void SpPipelineModelRunner::write_prefill(const std::string& task_id,
                                          const std::vector<int64_t>& token_ids,
                                          uint32_t max_tokens) {
  LLM_ENGINE_LOG("sp_pipeline") << "Writing to device: task_id=" << task_id
                                << " num_tokens=" << token_ids.size()
                                << " max_tokens=" << max_tokens << std::endl;
  device_input_.write(task_id, token_ids, max_tokens);
}

void SpPipelineModelRunner::exit() {
  if (stop_.exchange(true)) return;
  if (reader_thread_.joinable()) reader_thread_.join();
  LLM_ENGINE_LOG("sp_pipeline") << "model runner exit" << std::endl;
}

void SpPipelineModelRunner::reader_loop() {
  ReadResult read_buf;
  LLM_ENGINE_LOG("sp_pipeline") << "Reader loop started" << std::endl;
  while (!stop_.load(std::memory_order_relaxed)) {
    if (device_output_.try_read(read_buf)) {
      llm_engine::TokenResult result;
      result.task_id = llm_engine::TaskID::deserialize(
          read_buf.task_id.data(), llm_engine::TaskID::kSerializedSize);
      result.token_id = read_buf.token_ids.empty() ? 0 : read_buf.token_ids[0];

      LLM_ENGINE_LOG("sp_pipeline") << "Decoded token: task_id=" << result.task_id.id
                                    << " token_id=" << result.token_id << std::endl;
      decode_callback_(result);
    } else {
      std::this_thread::yield();
    }
  }
  LLM_ENGINE_LOG("sp_pipeline") << "Reader loop exited" << std::endl;
}

}  // namespace sp_pipeline
