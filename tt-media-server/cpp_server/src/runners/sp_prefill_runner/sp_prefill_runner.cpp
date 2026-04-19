// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_prefill_runner/sp_prefill_runner.hpp"

#include <chrono>
#include <cstdlib>
#include <future>

#include "config/settings.hpp"
#include "ipc/token_push.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

SpPrefillRunner::SpPrefillRunner(const config::LLMConfig& config,
                                 ipc::IResultQueue* resultQueue,
                                 tt::runners::llm_engine::ITaskQueue* taskQueue)
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
  constexpr size_t MAX_CONSECUTIVE_ERRORS = 5;
  const auto timeoutMs = config::prefillTimeoutMs();

  while (!stopped.load(std::memory_order_relaxed)) {
    // Get next sequence from task queue
    auto sequence = taskQueue->tryPop();
    if (!sequence) {
      std::this_thread::yield();
      continue;
    }
    TT_LOG_DEBUG("[SpPrefillRunner] Starting prefill for task {}",
                 sequence->taskId);

    // Run forward with timeout
    auto startTime = std::chrono::steady_clock::now();
    auto future = std::async(std::launch::async, [this, &sequence]() {
      return modelRunner->forward(sequence->taskId, sequence->getTokenIds());
    });

    std::optional<tt::runners::llm_engine::TokenResult> result;
    auto status = future.wait_for(std::chrono::milliseconds(timeoutMs));

    if (status == std::future_status::timeout) {
      // Timeout occurred
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - startTime)
                         .count();

      TT_LOG_ERROR(
          "[SpPrefillRunner] Prefill timeout for task {} after {}ms (limit: "
          "{}ms, tokens: {})",
          sequence->taskId, elapsed, timeoutMs, sequence->getTokenIds().size());

      // Increment consecutive errors
      size_t errorCount =
          consecutiveErrors.fetch_add(1, std::memory_order_relaxed) + 1;

      TT_LOG_ERROR("[SpPrefillRunner] Consecutive errors: {}/{}", errorCount,
                   MAX_CONSECUTIVE_ERRORS);

      // Push error token
      ipc::pushErrorToken(*resultQueue, sequence->taskId);

      // Check if we should terminate
      if (errorCount >= MAX_CONSECUTIVE_ERRORS) {
        TT_LOG_CRITICAL(
            "[SpPrefillRunner] Max consecutive errors ({}) reached. "
            "Terminating worker process.",
            MAX_CONSECUTIVE_ERRORS);
        std::exit(1);  // Kill the worker process
      }

      continue;
    }

    // Success - get the result
    result = future.get();

    if (!result) {
      TT_LOG_DEBUG(
          "[SpPrefillRunner] forward returned without result for task {}",
          sequence->taskId);
      break;  // stopped
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - startTime)
                       .count();

    TT_LOG_DEBUG("[SpPrefillRunner] forward finished for task {} in {}ms",
                 result->taskId, elapsed);

    if (result->isError) {
      TT_LOG_WARN("[SpPrefillRunner] Error token for task {}", result->taskId);

      // Increment consecutive errors
      size_t errorCount =
          consecutiveErrors.fetch_add(1, std::memory_order_relaxed) + 1;

      TT_LOG_ERROR("[SpPrefillRunner] Consecutive errors: {}/{}", errorCount,
                   MAX_CONSECUTIVE_ERRORS);

      ipc::pushErrorToken(*resultQueue, result->taskId);

      // Check if we should terminate
      if (errorCount >= MAX_CONSECUTIVE_ERRORS) {
        TT_LOG_CRITICAL(
            "[SpPrefillRunner] Max consecutive errors ({}) reached. "
            "Terminating worker process.",
            MAX_CONSECUTIVE_ERRORS);
        std::exit(1);  // Kill the worker process
      }
    } else {
      TT_LOG_DEBUG(
          "[SpPrefillRunner] pushToken task_id={} token_id={} finished={}",
          result->taskId, result->tokenId, true);
      ipc::pushToken(*resultQueue, result->taskId, result->tokenId, true);
      // Reset consecutive error count on success
      consecutiveErrors.store(0, std::memory_order_relaxed);
    }

    // sequence automatically cleaned up at end of scope
  }
}

bool SpPrefillRunner::warmup() {
  std::vector<int64_t> warmupTokens = {1};
  uint32_t warmupTaskId = 0;  // Use 0 for warmup task

  TT_LOG_DEBUG("[SpPrefillRunner] warmup forward task_id={} token_count={}",
               warmupTaskId, warmupTokens.size());
  auto result = modelRunner->forward(warmupTaskId, warmupTokens);
  if (!result || result->isError) {
    TT_LOG_ERROR("[SpPrefillRunner] Warmup failed");
    return false;
  }

  TT_LOG_INFO("[SpPrefillRunner] Warmup successful");
  return true;
}

void SpPrefillRunner::stop() {
  TT_LOG_INFO("[SpPrefillRunner] Stopping");
  stopped.store(true, std::memory_order_relaxed);
}
}  // namespace tt::runners
