// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_prefill_runner/sp_prefill_runner.hpp"

#include "utils/logger.hpp"

namespace tt::runners {

SpPrefillRunner::SpPrefillRunner(const config::LLMConfig& config,
                                 ipc::TokenRingBuffer<65536>* resultQueue,
                                 llm_engine::ITaskQueue* taskQueue)
    : config(config), resultQueue(resultQueue), taskQueue(taskQueue) {
  modelRunner = sp_prefill::makeModelRunner(config);
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
    auto sequence = taskQueue->tryPop();
    if (!sequence) {
      std::this_thread::yield();
      continue;
    }
    TT_LOG_DEBUG("SpPrefillRunner: Starting prefill for task {}",
                 sequence->taskId);

    auto result = modelRunner->forward(sequence->taskId, sequence->tokenIds);

    if (!result) {
      break;  // stopped
    }

    if (result->isError) {
      TT_LOG_WARN("SpPrefillRunner: Error token for task {}", result->taskId);
      pushErrorToken(result->taskId);
    } else {
      TT_LOG_DEBUG("SpPrefillRunner: Received prefill token {} for task {}",
                   result->tokenId, result->taskId);
      pushToken(result->taskId, result->tokenId, true);  // Always finished
    }

    // sequence automatically cleaned up at end of scope
  }
}

bool SpPrefillRunner::warmup() {
  std::vector<int64_t> warmupTokens = {1};
  uint32_t warmupTaskId = 0;  // Use 0 for warmup task

  auto result = modelRunner->forward(warmupTaskId, warmupTokens);
  if (!result || result->isError) {
    TT_LOG_ERROR("SpPrefillRunner: Warmup failed");
    return false;
  }

  TT_LOG_INFO("SpPrefillRunner: Warmup successful");
  return true;
}

void SpPrefillRunner::stop() {
  TT_LOG_INFO("SpPrefillRunner: Stopping");
  stopped.store(true, std::memory_order_relaxed);
}

void SpPrefillRunner::pushToken(uint32_t taskId, uint64_t tokenId,
                                bool finished) {
  ipc::SharedToken shared{};
  shared.token_index = 0;
  shared.flags = finished ? ipc::SharedToken::FLAG_FINAL : 0u;
  shared.token_id = tokenId;
  shared.task_id = taskId;
  resultQueue->push(shared);
}

void SpPrefillRunner::pushErrorToken(uint32_t taskId) {
  ipc::SharedToken shared{};
  shared.token_index = 0;
  shared.flags = ipc::SharedToken::FLAG_FINAL | ipc::SharedToken::FLAG_ERROR;
  shared.token_id = 0;
  shared.task_id = taskId;
  resultQueue->push(shared);
}

}  // namespace tt::runners
