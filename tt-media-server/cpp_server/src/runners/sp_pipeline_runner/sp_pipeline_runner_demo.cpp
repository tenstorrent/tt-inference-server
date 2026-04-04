// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_runner_demo.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

#include "config/settings.hpp"
#include "ipc/token_push.hpp"
#include "profiling/tracy.hpp"
#include "services/contiguous_memory_manager.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

SpPipelineRunnerDemo::SpPipelineRunnerDemo(
    const config::LLMConfig& config, ipc::TokenRingBuffer<65536>* resultQueue,
    llm_engine::ITaskQueue* taskQueue)
    : config(config),
      stopTokenIds(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      resultQueue(resultQueue),
      taskQueue(taskQueue),
      decodeQueue(config.max_in_flight_count),
      maxInFlightCount(config.max_in_flight_count * 30) {
  if (tt::config::llmMode() == config::LLMMode::DECODE_ONLY ||
      tt::config::llmMode() == config::LLMMode::REGULAR) {
    memoryManager = std::make_unique<services::ContiguousMemoryManager>();
    memoryThread = std::thread([this] { memoryLoop(); });
  }

  auto decodeCb = [this](const llm_engine::TokenResult& result) {
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
  llm_engine::SamplingParams warmupParams;
  warmupParams.max_tokens = 1;
  warmupParams.ignore_eos = true;

  std::vector<int64_t> warmupTokens = {1};  // Single token
  uint32_t warmupTaskId = 0;                // Use 0 for warmup task

  auto warmupSeq = std::make_unique<llm_engine::Sequence>(
      warmupTaskId,
      1,  // block_size (doesn't matter for warmup)
      warmupTokens, warmupParams);

  modelRunner->write(warmupSeq->taskId, warmupSeq->tokenIds, 1,
                     sp_pipeline::RequestPhase::PREFILL);

  // Wait for the response token (with timeout)
  const int maxAttempts = 1000;  // ~10 seconds with 10ms sleep
  int attempts = 0;
  bool receivedToken = false;

  while (attempts < maxAttempts && !receivedToken) {
    std::vector<llm_engine::TokenResult> results;
    decodeQueue.popMany(results, maxInFlightCount);
    for (const auto& dr : results) {
      if (dr.taskId == warmupTaskId) {
        if (dr.isError) {
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
  drainDecodeResults();

  if (inFlightCount >= maxInFlightCount) {
    return;
  }

  auto seq = taskQueue->tryPop();
  if (!seq) return;

  {
    ZoneScopedN("SpPipelineRunner::write_to_device");
    uint32_t taskId = seq->taskId;

    if (!seq->samplingParams->max_tokens.has_value()) {
      seq->samplingParams->max_tokens =
          static_cast<int>(config::LLMConfig::MAX_INPUT_TOKENS);
    }

    modelRunner->write(taskId, seq->tokenIds,
                       seq->samplingParams->max_tokens.value(),
                       sp_pipeline::RequestPhase::PREFILL);

    activeSequences.emplace(taskId, std::move(seq));
    ++inFlightCount;
  }
}

void SpPipelineRunnerDemo::drainDecodeResults() {
  std::vector<llm_engine::TokenResult> results;
  decodeQueue.popMany(results, maxInFlightCount);
  for (const auto& dr : results) {
    auto it = activeSequences.find(dr.taskId);
    if (it == activeSequences.end()) {  // safeguard for too many decode results
      TT_LOG_WARN(
          "SpPipelineRunner: task_id not found in active_sequences_: {}",
          dr.taskId);
      continue;
    }
    llm_engine::Sequence* seq = it->second.get();

    if (dr.isError) {
      ipc::pushErrorToken(*resultQueue, dr.taskId);
      activeSequences.erase(it);
      --inFlightCount;
      continue;
    }

    seq->appendToken(static_cast<int64_t>(dr.tokenId));

    bool isStop = stopTokenIds.count(static_cast<int64_t>(dr.tokenId)) > 0;
    bool reachedMaxTokens =
        seq->samplingParams->max_tokens.has_value() &&
        seq->numCompletionTokens() >=
            static_cast<size_t>(seq->samplingParams->max_tokens.value());
    bool finished =
        (!seq->samplingParams->ignore_eos && isStop) || reachedMaxTokens;

    ipc::pushToken(*resultQueue, dr.taskId, dr.tokenId, finished);

    if (finished) {
      activeSequences.erase(it);
      --inFlightCount;
    }
  }
}

}  // namespace tt::runners