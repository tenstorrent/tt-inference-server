// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "services/contiguous_memory_manager.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

SpPipelineRunner::SpPipelineRunner(const config::LLMConfig& config,
                                   ipc::TokenRingBuffer<65536>* resultQueue,
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

SpPipelineRunner::~SpPipelineRunner() {
  stop();
  if (memoryThread.joinable()) {
    memoryThread.join();
  }
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

  modelRunner->write(warmupSeq->taskId.id, warmupSeq->tokenIds, 1,
                     sp_pipeline::RequestPhase::PREFILL, false);

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

void SpPipelineRunner::stop() {
  stopped.store(true, std::memory_order_relaxed);
}

void SpPipelineRunner::memoryLoop() {
  while (!stopped.load(std::memory_order_relaxed)) {
    auto task = memoryManager->getRequest();
    if (task) {
      memoryManager->handleRequest(*task);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void SpPipelineRunner::step() {
  drainDecodeResults();

  if (inFlightCount >= maxInFlightCount) {
    return;
  }

  llm_engine::Sequence* seq = taskQueue->tryPop();
  if (!seq) return;

  {
    ZoneScopedN("SpPipelineRunner::write_to_device");
    std::unique_ptr<llm_engine::Sequence> owned(seq);
    llm_engine::TaskID taskId = seq->taskId;

    if (!seq->samplingParams->max_tokens.has_value()) {
      seq->samplingParams->max_tokens =
          static_cast<int>(config::LLMConfig::MAX_INPUT_TOKENS);
    }

    modelRunner->write(taskId.id, seq->tokenIds,
                       seq->samplingParams->max_tokens.value(),
                       sp_pipeline::RequestPhase::PREFILL, seq->fastMode);

    activeSequences.emplace(taskId, std::move(owned));
    ++inFlightCount;
  }
}

void SpPipelineRunner::drainDecodeResults() {
  std::vector<llm_engine::TokenResult> results;
  decodeQueue.popMany(results, maxInFlightCount);
  for (const auto& dr : results) {
    auto it = activeSequences.find(dr.taskId);
    if (it == activeSequences.end()) {  // safeguard for too many decode results
      TT_LOG_WARN(
          "SpPipelineRunner: task_id not found in active_sequences_: {}",
          dr.taskId.id);
      continue;
    }
    llm_engine::Sequence* seq = it->second.get();

    if (dr.isError) {
      pushErrorToken(dr.taskId);
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

    {
      pushToken(dr.taskId, dr.tokenId, finished);
    }

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
