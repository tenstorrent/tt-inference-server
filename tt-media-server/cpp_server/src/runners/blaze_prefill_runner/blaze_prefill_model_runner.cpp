// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/sp_prefill_runner/blaze_prefill_model_runner.hpp"

#include <chrono>
#include <cstdlib>

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace blaze_prefill {

BlazePrefillModelRunner::BlazePrefillModelRunner()
    : shmNames(), deviceInput(shmNames.write), deviceOutput(shmNames.read) {
  deviceInput.open();
  deviceOutput.open();
}

BlazePrefillModelRunner::~BlazePrefillModelRunner() { exit(); }

std::optional<tt::runners::llm_engine::TokenResult>
BlazePrefillModelRunner::forward(uint32_t taskId,
                                  const std::vector<int64_t>& tokenIds) {
  const auto timeoutMs = tt::config::prefillTimeoutMs();
  auto startTime = std::chrono::steady_clock::now();

  TT_LOG_DEBUG(
      "BlazePrefillModelRunner: Writing into shared memory input task_id={}, "
      "token count={}",
      taskId, tokenIds.size());
  deviceInput.write(taskId, tokenIds, 1);

  tt::ipc::ReadResult readBuf;
  TT_LOG_DEBUG(
      "BlazePrefillModelRunner: Reading from shared memory output task_id={}",
      taskId);

  while (!stop.load(std::memory_order_relaxed)) {
    // Check for timeout
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - startTime)
                       .count();

    if (elapsed >= timeoutMs) {
      TT_LOG_ERROR(
          "BlazePrefillModelRunner: Prefill timeout for task {} after {}ms "
          "(limit: {}ms, tokens: {})",
          taskId, elapsed, timeoutMs, tokenIds.size());

      // Increment consecutive errors
      size_t errorCount =
          consecutiveErrors.fetch_add(1, std::memory_order_relaxed) + 1;

      TT_LOG_ERROR("BlazePrefillModelRunner: Consecutive errors: {}/{}",
                   errorCount, MAX_CONSECUTIVE_ERRORS);

      // Check if we should terminate
      if (errorCount >= MAX_CONSECUTIVE_ERRORS) {
        TT_LOG_CRITICAL(
            "BlazePrefillModelRunner: Max consecutive errors ({}) reached. "
            "Terminating worker process.",
            MAX_CONSECUTIVE_ERRORS);
        std::exit(1);  // Kill the worker process
      }

      // Return error token
      return tt::runners::llm_engine::TokenResult(taskId, 0, std::nullopt,
                                                  true);
    }

    if (deviceOutput.tryRead(readBuf)) {
      uint64_t tokenId = readBuf.tokenIds.empty() ? 0 : readBuf.tokenIds[0];
      TT_LOG_DEBUG(
          "BlazePrefillModelRunner: Read from shared memory output task_id={}, "
          "token_id={}, token count={}",
          taskId, tokenId, readBuf.tokenIds.size());

      auto result =
          tt::runners::llm_engine::TokenResult(readBuf.taskId, tokenId);

      // Check if it's an error token and handle consecutive errors
      if (result.isError) {
        size_t errorCount =
            consecutiveErrors.fetch_add(1, std::memory_order_relaxed) + 1;

        TT_LOG_ERROR(
            "BlazePrefillModelRunner: Error token received for task {}. "
            "Consecutive errors: {}/{}",
            taskId, errorCount, MAX_CONSECUTIVE_ERRORS);

        if (errorCount >= MAX_CONSECUTIVE_ERRORS) {
          TT_LOG_CRITICAL(
              "BlazePrefillModelRunner: Max consecutive errors ({}) reached. "
              "Terminating worker process.",
              MAX_CONSECUTIVE_ERRORS);
          std::exit(1);
        }
      } else {
        // Reset consecutive error count on success
        consecutiveErrors.store(0, std::memory_order_relaxed);
      }

      return result;
    }
    std::this_thread::yield();
  }

  TT_LOG_DEBUG(
      "BlazePrefillModelRunner: forward exiting without Shared memory response "
      "(stop)");
  return std::nullopt;
}

void BlazePrefillModelRunner::exit() {
  stop.store(true, std::memory_order_relaxed);
  TT_LOG_INFO("[BlazePrefillModelRunner] Model runner exit");
}

}  // namespace blaze_prefill
