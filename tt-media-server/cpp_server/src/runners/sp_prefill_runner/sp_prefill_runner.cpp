// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_prefill_runner/sp_prefill_runner.hpp"

#include "ipc/token_push.hpp"
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
      ipc::pushErrorToken(*resultQueue, result->taskId);
    } else {
      TT_LOG_DEBUG("SpPrefillRunner: Received prefill token {} for task {}",
                   result->tokenId, result->taskId);
      ipc::pushToken(*resultQueue, result->taskId, result->tokenId, true);
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

}  // namespace tt::runners
