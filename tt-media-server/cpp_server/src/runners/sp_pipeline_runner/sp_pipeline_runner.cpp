// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"
#include <cassert>
#include <cstring>
#include <iostream>

namespace tt::runners {

SpPipelineRunner::SpPipelineRunner(
    const sp_pipeline::SpPipelineConfig& config,
    ipc::TokenRingBuffer<65536>* result_queue,
    llm_engine::ITaskQueue* task_queue)
    : config_(config),
      stop_token_ids_(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      result_queue_(result_queue),
      task_queue_(task_queue) {
  auto decode_cb = [this](const llm_engine::TokenResult& result) {
    decode_queue_.push(result);
  };

  model_runner_ = std::make_unique<sp_pipeline::SpPipelineModelRunner>(std::move(decode_cb));
}

SpPipelineRunner::~SpPipelineRunner() {
  if (model_runner_) {
    model_runner_->exit();
  }
}

void SpPipelineRunner::run() {
  while (!stopped_.load(std::memory_order_relaxed)) {
    step();
  }
}

void SpPipelineRunner::stop() {
  stopped_.store(true, std::memory_order_relaxed);
}

void SpPipelineRunner::step() {
  drain_decode_results();

  llm_engine::Sequence* seq = task_queue_->try_pop();
  if (!seq) return;

  std::unique_ptr<llm_engine::Sequence> owned(seq);
  llm_engine::TaskID task_id = seq->task_id;

  uint32_t max_tokens = seq->sampling_params->max_tokens.value_or(65536);
  model_runner_->write(seq->task_id.id, seq->token_ids_, max_tokens);

  active_sequences_.emplace(task_id, std::move(owned));
}

void SpPipelineRunner::drain_decode_results() {
  for (const auto& dr : decode_queue_.drain()) {
    auto it = active_sequences_.find(dr.task_id);
    if (it == active_sequences_.end()) { // safeguard for too many decode results
      std::cout << "SpPipelineRunner: task_id not found in active_sequences_: " << dr.task_id << std::endl;
      continue;
    }
    llm_engine::Sequence* seq = it->second.get();

    if (dr.is_error) {
      push_error_token(dr.task_id);
      active_sequences_.erase(it);
      continue;
    }

    seq->append_token(static_cast<int64_t>(dr.token_id));

    bool is_stop = stop_token_ids_.count(static_cast<int64_t>(dr.token_id)) > 0;
    bool reached_max_tokens = seq->sampling_params->max_tokens.has_value() &&
    seq->num_completion_tokens() >= static_cast<size_t>(seq->sampling_params->max_tokens.value());
    bool finished =
        (!seq->sampling_params->ignore_eos && is_stop) ||
        reached_max_tokens;

    push_token(dr.task_id, dr.token_id, finished);

    if (finished) {
      active_sequences_.erase(it);
    }
  }
}

void SpPipelineRunner::push_token(const llm_engine::TaskID& task_id,
                                  uint64_t token_id, bool finished) {
  ipc::SharedToken shared{};
  shared.token_index = 0;
  shared.flags = finished ? ipc::SharedToken::FLAG_FINAL : 0u;
  shared.token_id = token_id;
  std::strncpy(shared.task_id, task_id.id.c_str(), sizeof(shared.task_id) - 1);
  shared.task_id[sizeof(shared.task_id) - 1] = '\0';
  result_queue_->push(shared);
}

void SpPipelineRunner::push_error_token(const llm_engine::TaskID& task_id) {
  ipc::SharedToken shared{};
  shared.token_index = 0;
  shared.flags = ipc::SharedToken::FLAG_FINAL | ipc::SharedToken::FLAG_ERROR;
  shared.token_id = 0;
  std::strncpy(shared.task_id, task_id.id.c_str(), sizeof(shared.task_id) - 1);
  shared.task_id[sizeof(shared.task_id) - 1] = '\0';
  result_queue_->push(shared);
}

}  // namespace tt::runners
