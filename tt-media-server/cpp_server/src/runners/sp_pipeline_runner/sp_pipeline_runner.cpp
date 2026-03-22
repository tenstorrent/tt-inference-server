// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "profiling/tracy.hpp"
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
  auto decodeCb = [this](const llm_engine::TokenResult& result) {
    while (!decodeQueue.push(result)) {
      std::this_thread::yield();
    }
  };

  modelRunner = sp_pipeline::makeModelRunner(config, std::move(decodeCb));
}

SpPipelineRunner::~SpPipelineRunner() {
  if (modelRunner) {
    modelRunner->exit();
  }
}

void SpPipelineRunner::run() {
  while (!stopped.load(std::memory_order_relaxed)) {
    if (!modelRunner->isConnected()) {
      TT_LOG_WARN("SpPipelineRunner: pipeline disconnected, stopping runner");
      break;
    }
    step();
  }
}

bool SpPipelineRunner::warmup() {
  // Phase 1: Connect to pipeline shared memory.
  // Python creates shm before model loading, so this typically resolves in
  // seconds -- it only races on startup order.
  TT_LOG_INFO(
      "SpPipelineRunner: connecting to model pipeline shared memory...");
  modelRunner->connect();
  if (stopped.load(std::memory_order_relaxed)) return false;
  TT_LOG_INFO("SpPipelineRunner: connected to model pipeline");

  // Phase 2: Send one warmup token and wait for the pipeline response.
  // The pipeline may still be loading its model (10+ min), so the token sits
  // in the ring buffer until the pipeline enters its inference loop.
  const int TIMEOUT_SECONDS = [&]() {
    const char* s = std::getenv("TT_WARMUP_TIMEOUT_S");
    if (s) {
      try {
        return std::stoi(s);
      } catch (...) {
      }
    }
    return 1200;  // default 20 min
  }();

  const llm_engine::TaskID WARMUP_TASK_ID("warmup_task");
  const std::vector<int64_t> WARMUP_TOKENS = {1};

  modelRunner->write(WARMUP_TASK_ID.id, WARMUP_TOKENS, 1,
                     sp_pipeline::RequestPhase::PREFILL);
  TT_LOG_INFO(
      "SpPipelineRunner: warmup token sent, waiting for model pipeline "
      "response (timeout={}s)...",
      TIMEOUT_SECONDS);

  const auto DEADLINE =
      std::chrono::steady_clock::now() + std::chrono::seconds(TIMEOUT_SECONDS);
  auto lastLogTime = std::chrono::steady_clock::now();
  static constexpr auto LOG_INTERVAL = std::chrono::seconds(30);
  const auto WARMUP_START = std::chrono::steady_clock::now();

  while (!stopped.load(std::memory_order_relaxed)) {
    std::vector<llm_engine::TokenResult> results;
    decodeQueue.popMany(results, maxInFlightCount);
    for (const auto& dr : results) {
      if (dr.taskId == WARMUP_TASK_ID) {
        if (dr.isError) {
          TT_LOG_ERROR("SpPipelineRunner: warmup failed with error");
          return false;
        }
        TT_LOG_INFO("SpPipelineRunner: warmup successful");
        return true;
      }
    }

    const auto NOW = std::chrono::steady_clock::now();
    if (NOW > DEADLINE) {
      TT_LOG_ERROR("SpPipelineRunner: warmup timed out after {}s",
                   TIMEOUT_SECONDS);
      return false;
    }

    if (NOW - lastLogTime >= LOG_INTERVAL) {
      lastLogTime = NOW;
      const auto ELAPSED_S =
          std::chrono::duration_cast<std::chrono::seconds>(NOW - WARMUP_START)
              .count();
      TT_LOG_INFO(
          "SpPipelineRunner: waiting for model pipeline warmup response... "
          "(elapsed {}s)",
          ELAPSED_S);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  return false;
}

void SpPipelineRunner::stop() {
  stopped.store(true, std::memory_order_relaxed);
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

    modelRunner->write(taskId.id, seq->tokenIds,
                       seq->samplingParams->max_tokens.value(),
                       sp_pipeline::RequestPhase::PREFILL);

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
