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
    : config(config),
      stopTokenIds(config.stop_token_ids.begin(),
                   config.stop_token_ids.end()),
      resultQueue(resultQueue),
      taskQueue(taskQueue),
      maxInFlightCount(config.max_in_flight_count) {
  auto decodeCb = [this](const llm_engine::TokenResult& result) {
    decodeQueue.push(result);
  };

  modelRunner = std::make_unique<sp_pipeline::SpPipelineModelRunner>(
      std::move(decodeCb));
}

SpPipelineRunner::~SpPipelineRunner() {
  if (modelRunner) {
    modelRunner->exit();
  }
}

void SpPipelineRunner::run() {
  while (!stopped.load(std::memory_order_relaxed)) {
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
  modelRunner->write(warmupSeq->task_id.id, warmupSeq->token_ids_, 1);

  // Wait for the response token (with timeout)
  const int MAX_ATTEMPTS = 1000;  // ~10 seconds with 10ms sleep
  int attempts = 0;
  bool receivedToken = false;

  while (attempts < MAX_ATTEMPTS && !receivedToken) {
    // Drain decode queue to check for results
    for (const auto& dr : decodeQueue.drain()) {
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
  stopped.store(true, std::memory_order_relaxed);
}

void SpPipelineRunner::step() {
  drainDecodeResults();

  // Check if we can accept more in-flight requests
  if (inFlightCount >= maxInFlightCount) {
    return;
  }

  llm_engine::Sequence* seq = taskQueue->try_pop();
  if (!seq) return;

  std::unique_ptr<llm_engine::Sequence> owned(seq);
  llm_engine::TaskID taskId = seq->task_id;

  uint32_t maxTokens =
      seq->sampling_params->max_tokens.value_or(config.MAX_INPUT_TOKENS);
  modelRunner->write(seq->task_id.id, seq->token_ids_, maxTokens);

  activeSequences.emplace(taskId, std::move(owned));
  ++inFlightCount;
}

void SpPipelineRunner::drainDecodeResults() {
  for (const auto& dr : decodeQueue.drain()) {
    auto it = activeSequences.find(dr.task_id);
    if (it ==
        activeSequences.end()) {  // safeguard for too many decode results
      TT_LOG_WARN(
          "SpPipelineRunner: task_id not found in activeSequences: {}",
          dr.task_id.id);
      continue;
    }
    llm_engine::Sequence* seq = it->second.get();

    if (dr.is_error) {
      pushErrorToken(dr.task_id);
      activeSequences.erase(it);
      --inFlightCount;
      continue;
    }

    seq->appendToken(static_cast<int64_t>(dr.token_id));

    bool isStop = stopTokenIds.count(static_cast<int64_t>(dr.token_id)) > 0;
    bool reachedMaxTokens =
        seq->sampling_params->max_tokens.has_value() &&
        seq->numCompletionTokens() >=
            static_cast<size_t>(seq->sampling_params->max_tokens.value());
    bool finished =
        (!seq->sampling_params->ignore_eos && isStop) || reachedMaxTokens;

    pushToken(dr.task_id, dr.token_id, finished);

    if (finished) {
      activeSequences.erase(it);
      --inFlightCount;
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
  resultQueue->push(shared);
}

void SpPipelineRunner::pushErrorToken(const llm_engine::TaskID& taskId) {
  ipc::SharedToken shared{};
  shared.token_index = 0;
  shared.flags = ipc::SharedToken::FLAG_FINAL | ipc::SharedToken::FLAG_ERROR;
  shared.token_id = 0;
  std::strncpy(shared.task_id, taskId.id.c_str(), sizeof(shared.task_id) - 1);
  shared.task_id[sizeof(shared.task_id) - 1] = '\0';
  resultQueue->push(shared);
}

}  // namespace tt::runners
