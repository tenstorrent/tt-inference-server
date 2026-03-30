// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_prefill_runner/sp_prefill_runner.hpp"

#include <cassert>
#include <chrono>
#include <cstring>

#include "ipc/shared_memory.hpp"
#include "profiling/tracy.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

SpPrefillRunner::SpPrefillRunner(const config::LLMConfig& config,
                                 ipc::TokenRingBuffer<65536>* resultQueue,
                                 llm_engine::ITaskQueue* taskQueue)
    : config(config),
      stopTokenIds(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      resultQueue(resultQueue),
      taskQueue(taskQueue),
      prefillQueue(256) {  // Small queue since we only have 1 in flight
  auto prefillCb = [this](const llm_engine::TokenResult& result) {
    while (!prefillQueue.push(result)) {
      std::this_thread::yield();
    }
  };

  modelRunner = sp_prefill::makeModelRunner(config, std::move(prefillCb));
}

SpPrefillRunner::~SpPrefillRunner() {
  stop();
  if (modelRunner) {
    modelRunner->exit();
  }
}

void SpPrefillRunner::run() {
  while (!stopped.load(std::memory_order_relaxed)) {
    // Get next sequence from task queue
    auto* seq = taskQueue->tryPop();
    if (!seq) {
      std::this_thread::yield();
      continue;
    }

    // Use unique_ptr to ensure cleanup
    std::unique_ptr<llm_engine::Sequence> sequence(seq);
    TT_LOG_DEBUG("SpPrefillRunner: Starting prefill for task {}",
                 sequence->taskId.id);

    // Send prefill request
    modelRunner->write(sequence->taskId.id, sequence->tokenIds);

    // Wait synchronously for the single prefill token
    llm_engine::TokenResult result;
    while (!stopped.load(std::memory_order_relaxed)) {
      if (prefillQueue.pop(result)) {
        break;
      }
      std::this_thread::yield();
    }

    if (stopped.load(std::memory_order_relaxed)) {
      break;
    }

    // Process the result
    if (result.isError) {
      TT_LOG_WARN("SpPrefillRunner: Error token for task {}", result.taskId.id);
      pushErrorToken(result.taskId);
    } else {
      TT_LOG_DEBUG("SpPrefillRunner: Received prefill token {} for task {}",
                   result.tokenId, result.taskId.id);
      pushToken(result.taskId, result.tokenId, true);  // Always finished
    }

    // sequence automatically cleaned up at end of scope
  }
}

bool SpPrefillRunner::warmup() {
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

  modelRunner->write(warmupSeq->taskId.id, warmupSeq->tokenIds);

  // Wait for the response token (with timeout)
  const int maxAttempts = 1000;  // ~10 seconds with 10ms sleep
  int attempts = 0;
  bool receivedToken = false;

  while (attempts < maxAttempts && !receivedToken) {
    std::vector<llm_engine::TokenResult> results;
    prefillQueue.popMany(results, 256);
    for (const auto& dr : results) {
      if (dr.taskId == warmupTaskId) {
        if (dr.isError) {
          TT_LOG_ERROR("SpPrefillRunner: Warmup failed with error");
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
    TT_LOG_ERROR("SpPrefillRunner: Warmup timed out waiting for token");
    return false;
  }

  TT_LOG_INFO("SpPrefillRunner: Warmup successful");
  return true;
}

void SpPrefillRunner::stop() {
  TT_LOG_INFO("SpPrefillRunner: Stopping");
  stopped.store(true, std::memory_order_relaxed);
}

void SpPrefillRunner::pushToken(const llm_engine::TaskID& taskId,
                                uint64_t tokenId, bool finished) {
  ipc::SharedToken shared{};
  shared.token_index = 0;
  shared.flags = finished ? ipc::SharedToken::FLAG_FINAL : 0u;
  shared.token_id = tokenId;
  std::strncpy(shared.task_id, taskId.id.c_str(), sizeof(shared.task_id) - 1);
  shared.task_id[sizeof(shared.task_id) - 1] = '\0';
  resultQueue->push(shared);
}

void SpPrefillRunner::pushErrorToken(const llm_engine::TaskID& taskId) {
  ipc::SharedToken shared{};
  shared.token_index = 0;
  shared.flags = ipc::SharedToken::FLAG_FINAL | ipc::SharedToken::FLAG_ERROR;
  shared.token_id = 0;
  std::strncpy(shared.task_id, taskId.id.c_str(), sizeof(shared.task_id) - 1);
  shared.task_id[sizeof(shared.task_id) - 1] = '\0';
  resultQueue->push(shared);
}

}  // namespace tt::runners
