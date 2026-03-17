// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <thread>

#include "utils/logger.hpp"

namespace tt::runners {

SpPipelineRunner::SpPipelineRunner(const tt::config::LLMConfig& config,
                                   ipc::TokenRingBuffer<65536>* result_queue,
                                   llm_engine::ITaskQueue* task_queue)
    : config_(config),
      stop_token_ids_(config.stop_token_ids.begin(),
                      config.stop_token_ids.end()),
      result_queue_(result_queue),
      task_queue_(task_queue),
      max_in_flight_count_(config.max_in_flight_count) {
  auto decode_cb = [this](const llm_engine::TokenResult& result) {
    decode_queue_.push(result);
  };

  model_runner_ = std::make_unique<sp_pipeline::SpPipelineModelRunner>(
      std::move(decode_cb));
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

bool SpPipelineRunner::warmup() {
  // Create a warmup sequence with a single token
  llm_engine::SamplingParams warmup_params;
  warmup_params.max_tokens = 1;
  warmup_params.ignore_eos = true;

  std::vector<int64_t> warmup_tokens = {1};  // Single token
  llm_engine::TaskID warmup_task_id("warmup_task");

  auto warmup_seq = std::make_unique<llm_engine::Sequence>(
      warmup_task_id,
      1,  // block_size (doesn't matter for warmup)
      warmup_tokens, warmup_params);

  // Write the warmup sequence to the model runner
  model_runner_->write(warmup_seq->task_id.id, warmup_seq->token_ids_, 1);

  // Wait for the response token (with timeout)
  const int max_attempts = 1000;  // ~10 seconds with 10ms sleep
  int attempts = 0;
  bool received_token = false;

  while (attempts < max_attempts && !received_token) {
    // Drain decode queue to check for results
    for (const auto& dr : decode_queue_.drain()) {
      if (dr.task_id == warmup_task_id) {
        if (dr.is_error) {
          TT_LOG_ERROR("SpPipelineRunner: Warmup failed with error");
          return false;
        }
        received_token = true;
        break;
      }
    }

    if (!received_token) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      attempts++;
    }
  }

  if (!received_token) {
    TT_LOG_ERROR("SpPipelineRunner: Warmup timed out waiting for token");
    return false;
  }

  TT_LOG_INFO("SpPipelineRunner: Warmup successful");
  return true;
}

void SpPipelineRunner::stop() {
  stopped_.store(true, std::memory_order_relaxed);
}

void SpPipelineRunner::step() {
  drain_decode_results();

  // Check if we can accept more in-flight requests
  if (in_flight_count_ >= max_in_flight_count_) {
    return;
  }

  llm_engine::Sequence* seq = task_queue_->try_pop();
  if (!seq) return;

  std::unique_ptr<llm_engine::Sequence> owned(seq);
  llm_engine::TaskID task_id = seq->task_id;

  uint32_t max_tokens =
      seq->sampling_params->max_tokens.value_or(config_.MAX_INPUT_TOKENS);
  model_runner_->write(seq->task_id.id, seq->token_ids_, max_tokens);

  active_sequences_.emplace(task_id, std::move(owned));
  ++in_flight_count_;
}

void SpPipelineRunner::drain_decode_results() {
  for (const auto& dr : decode_queue_.drain()) {
    auto it = active_sequences_.find(dr.task_id);
    if (it ==
        active_sequences_.end()) {  // safeguard for too many decode results
      TT_LOG_WARN(
          "SpPipelineRunner: task_id not found in active_sequences_: {}",
          dr.task_id.id);
      continue;
    }
    llm_engine::Sequence* seq = it->second.get();

    if (dr.is_error) {
      push_error_token(dr.task_id);
      active_sequences_.erase(it);
      --in_flight_count_;
      continue;
    }

    seq->append_token(static_cast<int64_t>(dr.token_id));

    bool is_stop = stop_token_ids_.count(static_cast<int64_t>(dr.token_id)) > 0;
    bool reached_max_tokens =
        seq->sampling_params->max_tokens.has_value() &&
        seq->num_completion_tokens() >=
            static_cast<size_t>(seq->sampling_params->max_tokens.value());
    bool finished =
        (!seq->sampling_params->ignore_eos && is_stop) || reached_max_tokens;

    push_token(dr.task_id, dr.token_id, finished);

    if (finished) {
      active_sequences_.erase(it);
      --in_flight_count_;
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
