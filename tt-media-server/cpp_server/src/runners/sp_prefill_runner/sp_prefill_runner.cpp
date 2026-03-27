// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_prefill_runner/sp_prefill_runner.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <thread>

#include "domain/manage_memory.hpp"
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
      prefillQueue(256),  // Small queue since we only have 1 in flight
      activeSequence(nullptr) {
  memoryThread = std::thread([this] { memoryLoop(); });

  auto prefillCb = [this](const llm_engine::TokenResult& result) {
    while (!prefillQueue.push(result)) {
      std::this_thread::yield();
    }
  };

  modelRunner = sp_prefill::makeModelRunner(config, std::move(prefillCb));
}

SpPrefillRunner::~SpPrefillRunner() {
  stop();
  if (memoryThread.joinable()) {
    memoryThread.join();
  }
  if (modelRunner) {
    modelRunner->exit();
  }
}

void SpPrefillRunner::run() {
  while (!stopped.load(std::memory_order_relaxed)) {
    step();
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
  const int MAX_ATTEMPTS = 1000;  // ~10 seconds with 10ms sleep
  int attempts = 0;
  bool receivedToken = false;

  while (attempts < MAX_ATTEMPTS && !receivedToken) {
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

void SpPrefillRunner::step() {
  ZoneScopedN("SpPrefillRunner::step");

  // Process incoming prefill results first
  drainPrefillResults();

  // If no active sequence, try to get one from the queue
  if (!activeSequence) {
    auto* seq = taskQueue->tryPop();
    if (!seq) {
      std::this_thread::yield();
      return;
    }

    activeSequence.reset(seq);
    TT_LOG_DEBUG("SpPrefillRunner: Starting prefill for task {}",
                 activeSequence->taskId.id);

    // Send prefill request
    modelRunner->write(activeSequence->taskId.id, activeSequence->tokenIds);
  }
}

void SpPrefillRunner::drainPrefillResults() {
  ZoneScopedN("SpPrefillRunner::drainPrefillResults");

  std::vector<llm_engine::TokenResult> results;
  prefillQueue.popMany(results, 256);

  for (const auto& dr : results) {
    if (dr.isError) {
      TT_LOG_WARN("SpPrefillRunner: Error token for task {}", dr.taskId.id);
      pushErrorToken(dr.taskId);

      // Clear active sequence if it matches
      if (activeSequence && activeSequence->taskId == dr.taskId) {
        activeSequence.reset();
      }
      continue;
    }

    TT_LOG_DEBUG("SpPrefillRunner: Received token {} for task {}",
                 dr.tokenId, dr.taskId.id);

    pushToken(dr.taskId, dr.tokenId, true);  // Always finished after prefill

    // Clear active sequence if it matches
    if (activeSequence && activeSequence->taskId == dr.taskId) {
      activeSequence.reset();
    }
  }
}

void SpPrefillRunner::memoryLoop() {
  tt::domain::ManageMemoryTask task{};
  std::vector<tt::domain::ManageMemoryTask> retryQueue;

  while (!stopped.load(std::memory_order_relaxed)) {
    if (!retryQueue.empty()) {
      auto result = memoryManager.handle_task(retryQueue.front());
      if (result.status != domain::ManageMemoryStatus::WAITING) {
        memoryResults.push(result);
        retryQueue.erase(retryQueue.begin());
      }
    } else if (memoryRequests.tryPop(task)) {
      auto result = memoryManager.handle_task(task);
      if (result.status == domain::ManageMemoryStatus::WAITING) {
        retryQueue.push_back(task);
      } else {
        memoryResults.push(result);
      }
    } else {
      std::this_thread::yield();
    }
  }
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
