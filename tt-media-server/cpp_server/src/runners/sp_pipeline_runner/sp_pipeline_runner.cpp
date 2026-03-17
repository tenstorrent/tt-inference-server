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
                                   ipc::TokenRingBuffer<65536>* resultQueue,
                                   llm_engine::ITaskQueue* taskQueue)
    : config_(config),
      stop_token_ids_(config.stop_token_ids.begin(),
                      config.stop_token_ids.end()),
      result_queue_(resultQueue),
      task_queue_(taskQueue),
      decode_queue_(config.max_in_flight_count),
      max_in_flight_count_(config.max_in_flight_count) {
  auto decodeCb = [this](const llm_engine::TokenResult& result) {
    decode_queue_.push(result);
  };

  model_runner_ =
      std::make_unique<sp_pipeline::SpPipelineModelRunner>(std::move(decodeCb));
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
  llm_engine::SamplingParams warmupParams;
  warmupParams.max_tokens = 1;
  warmupParams.ignore_eos = true;

  std::vector<int64_t> warmupTokens = {1};  // Single token
  llm_engine::TaskID warmupTaskId("warmup_task");

  auto warmupSeq = std::make_unique<llm_engine::Sequence>(
      warmupTaskId,
      1,  // block_size (doesn't matter for warmup)
      warmupTokens, warmupParams);

  // Write the warmup sequence to the model runner
  model_runner_->write(warmupSeq->task_id.id, warmupSeq->token_ids_, 1);

  // Wait for the response token (with timeout)
  const int MAX_ATTEMPTS = 1000;  // ~10 seconds with 10ms sleep
  int attempts = 0;
  bool receivedToken = false;

  while (attempts < MAX_ATTEMPTS && !receivedToken) {
    std::vector<llm_engine::TokenResult> results;
    decode_queue_.popMany(results, max_in_flight_count_);
    for (const auto& dr : results) {
      if (dr.task_id == warmupTaskId) {
        if (dr.is_error) {
          TT_LOG_ERROR("SpPipelineRunner: Warmup failed with error");
          return false;
        }
        receivedToken = true;
        break;
      }
    }

    if (!receivedToken) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      attempts++;
    }
  }

  if (!receivedToken) {
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
  drainDecodeResults();

  // Check if we can accept more in-flight requests
  if (in_flight_count_ >= max_in_flight_count_) {
    return;
  }

  llm_engine::Sequence* seq = task_queue_->tryPop();
  if (!seq) return;

  std::unique_ptr<llm_engine::Sequence> owned(seq);
  llm_engine::TaskID taskId = seq->task_id;

  uint32_t maxTokens =
      seq->sampling_params->max_tokens.value_or(config_.MAX_INPUT_TOKENS);
  model_runner_->write(seq->task_id.id, seq->token_ids_, maxTokens);

  active_sequences_.emplace(taskId, std::move(owned));
  ++in_flight_count_;
}

void SpPipelineRunner::drainDecodeResults() {
  std::vector<llm_engine::TokenResult> results;
  decode_queue_.popMany(results, max_in_flight_count_);
  for (const auto& dr : results) {
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
      pushErrorToken(dr.task_id);
      active_sequences_.erase(it);
      --in_flight_count_;
      continue;
    }

    seq->appendToken(static_cast<int64_t>(dr.token_id));

    bool isStop = stop_token_ids_.count(static_cast<int64_t>(dr.token_id)) > 0;
    bool reachedMaxTokens =
        seq->sampling_params->max_tokens.has_value() &&
        seq->numCompletionTokens() >=
            static_cast<size_t>(seq->sampling_params->max_tokens.value());
    bool finished =
        (!seq->sampling_params->ignore_eos && isStop) || reachedMaxTokens;

    pushToken(dr.task_id, dr.token_id, finished);

    if (finished) {
      active_sequences_.erase(it);
      --in_flight_count_;
    }
  }
}

void SpPipelineRunner::pushToken(const llm_engine::TaskID& taskId,
                                 uint64_t tokenId, bool finished) {
  ipc::SharedToken shared{};
  shared.token_index = 0;
  shared.flags = finished ? ipc::SharedToken::FLAG_FINAL : 0u;
  shared.token_id = tokenId;
  std::strncpy(shared.task_id, taskId.id.c_str(), sizeof(shared.task_id) - 1);
  shared.task_id[sizeof(shared.task_id) - 1] = '\0';
  result_queue_->push(shared);
}

void SpPipelineRunner::pushErrorToken(const llm_engine::TaskID& taskId) {
  ipc::SharedToken shared{};
  shared.token_index = 0;
  shared.flags = ipc::SharedToken::FLAG_FINAL | ipc::SharedToken::FLAG_ERROR;
  shared.token_id = 0;
  std::strncpy(shared.task_id, taskId.id.c_str(), sizeof(shared.task_id) - 1);
  shared.task_id[sizeof(shared.task_id) - 1] = '\0';
  result_queue_->push(shared);
}

}  // namespace tt::runners
