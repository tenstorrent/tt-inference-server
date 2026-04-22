// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/sp_pipeline_runner/sp_pipeline_runner_demo.hpp"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>

#include "config/settings.hpp"
#include "ipc/token_push.hpp"
#include "profiling/tracy.hpp"
#include "services/memory_services/contiguous_memory_manager.hpp"
#include "utils/logger.hpp"
#include "worker/single_process_worker_metrics.hpp"

namespace tt::runners {

SpPipelineRunnerDemo::SpPipelineRunnerDemo(const config::LLMConfig& config,
                                           ipc::IResultQueue* resultQueue,
                                           tt::ipc::ITaskQueue* taskQueue)
    : config(config),
      stopTokenIds(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      resultQueue(resultQueue),
      taskQueue(taskQueue),
      decodeQueue(config.max_in_flight_count),
      maxInFlightCount(config.max_in_flight_count * 30),
      lastOutputTime(std::chrono::steady_clock::now()),
      outputHangTimeout(tt::config::outputHangTimeoutMs()) {
  if (tt::config::llmMode() == config::LLMMode::DECODE_ONLY ||
      tt::config::llmMode() == config::LLMMode::REGULAR) {
    memoryManager = std::make_unique<services::ContiguousMemoryManager>(64);
    memoryThread = std::thread([this] { memoryLoop(); });
  }

  auto decodeCb = [this](const tt::domain::TokenResult& result) {
    while (!decodeQueue.push(result)) {
      std::this_thread::yield();
    }
  };

  modelRunner = sp_pipeline::makeModelRunner(config, std::move(decodeCb));
}

SpPipelineRunnerDemo::~SpPipelineRunnerDemo() {
  stop();
  if (memoryThread.joinable()) {
    memoryThread.join();
  }
  if (modelRunner) {
    modelRunner->exit();
  }
}

void SpPipelineRunnerDemo::run() {
  while (!stopped.load(std::memory_order_relaxed)) {
    step();
  }
}

bool SpPipelineRunnerDemo::warmup() {
  // Create a warmup sequence with a single token
  tt::domain::SamplingParams warmupParams;
  warmupParams.max_tokens = 1;
  warmupParams.ignore_eos = true;

  std::vector<int64_t> warmupTokens = {1};  // Single token
  uint32_t warmupTaskId = 0;                // Use 0 for warmup task

  auto warmupSeq = std::make_unique<tt::domain::Sequence>(
      warmupTaskId,
      1,  // block_size (doesn't matter for warmup)
      warmupTokens, warmupParams);

  modelRunner->write(warmupSeq->taskId, warmupSeq->getTokenIds(), 1,
                     sp_pipeline::RequestPhase::PREFILL, false);

  // Wait for the response token (with timeout)
  const int maxAttempts = 1000;  // ~10 seconds with 10ms sleep
  int attempts = 0;
  bool receivedToken = false;

  while (attempts < maxAttempts && !receivedToken) {
    std::vector<tt::domain::TokenResult> results;
    decodeQueue.popMany(results, maxInFlightCount);
    for (const auto& dr : results) {
      if (dr.taskId == warmupTaskId) {
        if (dr.isError) {
          TT_LOG_ERROR("[SpPipelineRunnerDemo] Warmup failed with error");
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
    TT_LOG_ERROR("[SpPipelineRunnerDemo] Warmup timed out waiting for token");
    return false;
  }

  TT_LOG_INFO("[SpPipelineRunnerDemo] Warmup successful");
  return true;
}

void SpPipelineRunnerDemo::stop() {
  stopped.store(true, std::memory_order_relaxed);
}

void SpPipelineRunnerDemo::memoryLoop() {
  while (!stopped.load(std::memory_order_relaxed)) {
    auto task = memoryManager->getRequest();
    if (task) {
      memoryManager->handleRequest(*task);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void SpPipelineRunnerDemo::step() {
  tt::worker::SingleProcessWorkerMetrics::instance().updateStepHeartbeat();
  drainDecodeResults();
  checkOutputHang();

  if (inFlightCount >= maxInFlightCount) {
    return;
  }

  auto seq = taskQueue->tryPop();
  if (!seq) return;

  {
    ZoneScopedN("SpPipelineRunnerDemo::write_to_device");
    uint32_t taskId = seq->taskId;

    if (!seq->getSamplingParams().max_tokens.has_value()) {
      seq->getMutableSamplingParams().max_tokens =
          static_cast<int>(config::LLMConfig::MAX_INPUT_TOKENS);
    }

    if (inFlightCount == 0) {
      lastOutputTime = std::chrono::steady_clock::now();
    }

    tt::worker::SingleProcessWorkerMetrics::instance()
        .incrementActiveRequests();
    modelRunner->write(
        taskId, seq->getTokenIds(), seq->getSamplingParams().max_tokens.value(),
        sp_pipeline::RequestPhase::PREFILL, seq->getSamplingParams().fast_mode);

    activeSequences.emplace(taskId, std::move(seq));
    ++inFlightCount;
  }
}

void SpPipelineRunnerDemo::checkOutputHang() {
  if (inFlightCount == 0) {
    lastOutputTime = std::chrono::steady_clock::now();
    return;
  }
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - lastOutputTime);
  if (elapsed <= outputHangTimeout) {
    return;
  }
  TT_LOG_CRITICAL(
      "[SpPipelineRunnerDemo] Output hang detected: no model output for {} ms "
      "with {} active request(s) (threshold={} ms). Self-terminating worker "
      "so infrastructure can restart the server.",
      elapsed.count(), inFlightCount, outputHangTimeout.count());
  // See BlazeRunner::checkOutputHang for rationale on using abort() here.
  std::abort();
}

void SpPipelineRunnerDemo::drainDecodeResults() {
  std::vector<tt::domain::TokenResult> results;
  decodeQueue.popMany(results, maxInFlightCount);
  for (const auto& dr : results) {
    tt::worker::SingleProcessWorkerMetrics::instance().updateOutputHeartbeat();
    lastOutputTime = std::chrono::steady_clock::now();
    auto it = activeSequences.find(dr.taskId);
    if (it == activeSequences.end()) {
      TT_LOG_WARN(
          "[SpPipelineRunnerDemo] task_id not found in active_sequences_: {}",
          dr.taskId);
      continue;
    }
    tt::domain::Sequence* seq = it->second.get();

    if (dr.isError) {
      ipc::pushErrorToken(*resultQueue, dr.taskId);
      tt::worker::SingleProcessWorkerMetrics::instance()
          .decrementActiveRequests();
      activeSequences.erase(it);
      --inFlightCount;
      continue;
    }

    seq->appendToken(static_cast<int64_t>(dr.tokenId));

    bool isStop = stopTokenIds.count(static_cast<int64_t>(dr.tokenId)) > 0;
    bool reachedMaxTokens =
        seq->getSamplingParams().max_tokens.has_value() &&
        seq->numCompletionTokens() >=
            static_cast<size_t>(seq->getSamplingParams().max_tokens.value());
    bool finished =
        (!seq->getSamplingParams().ignore_eos && isStop) || reachedMaxTokens;

    ipc::pushToken(*resultQueue, dr.taskId, dr.tokenId, finished);

    if (finished) {
      tt::worker::SingleProcessWorkerMetrics::instance()
          .decrementActiveRequests();
      activeSequences.erase(it);
      --inFlightCount;
    }
  }
}

}  // namespace tt::runners
