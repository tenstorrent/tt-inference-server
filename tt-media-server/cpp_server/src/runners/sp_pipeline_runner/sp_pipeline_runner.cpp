// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <pipeline_manager/pipeline_manager_types.hpp>
#include <thread>

#include "domain/manage_memory.hpp"
#include "llm_runner/sequence.hpp"
#include "runners/sp_pipeline_runner/sp_pipeline_utils.hpp"
#include "utils/logger.hpp"
#include "ipc/token_push.hpp"

namespace tt::runners {
namespace utils = sp_pipeline_utils;

SpPipelineRunner::SpPipelineRunner(const config::LLMConfig& config,
                                   ipc::TokenRingBuffer<65536>* resultQueue,
                                   llm_engine::ITaskQueue* taskQueue)
    : config(config),
      stopTokenIds(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      resultQueue(resultQueue),
      taskQueue(taskQueue) {
  TT_LOG_INFO(
      "SpPipelineRunner: Constructing PipelineManager with SocketConfig...");
  pm::SocketConfig socketConfig{
      .h2d_socket_id = "h2d_socket",
      .d2h_socket_id = "d2h_socket",
      .connect_timeout_ms = 30000,
  };
  pm::ManagerParams managerParams{.max_users = 32};
  pipelineManager =
      std::make_unique<pm::PipelineManager>(socketConfig, managerParams);
  TT_LOG_INFO(
      "SpPipelineRunner: PipelineManager constructed, calling start()...");
  pipelineManager->start();
  TT_LOG_INFO(
      "SpPipelineRunner: PipelineManager started, creating MemoryManager...");
  memoryManager = std::make_unique<tt::services::SpPipelineMemoryManager>(
      *pipelineManager, [this](uint32_t slotId) { evictSlot(slotId); });
  TT_LOG_INFO("SpPipelineRunner: Constructor complete");
}

SpPipelineRunner::~SpPipelineRunner() {
  stop();
  if (pipelineManager) {
    pipelineManager->stop();
  }
}

void SpPipelineRunner::run() {
  while (!stopped.load(std::memory_order_relaxed)) {
    try {
      step();
    } catch (const std::exception& e) {
      TT_LOG_ERROR("SpPipelineRunner: Exception in run: {}", e.what());
      throw;
    }
  }
}

bool SpPipelineRunner::warmup() {
  llm_engine::SamplingParams warmupParams;
  warmupParams.max_tokens = 1;
  warmupParams.ignore_eos = true;

  std::vector<int64_t> warmupTokens = {1};
  uint32_t warmupTaskId = 0;

  auto warmupSeq = std::make_unique<llm_engine::Sequence>(
      warmupTaskId, 1, warmupTokens, warmupParams);

  TT_LOG_INFO("SpPipelineRunner: warmup - pushing ALLOCATE request...");
  pipelineManager->push_request(utils::makeAllocateRequest(0));

  TT_LOG_INFO("SpPipelineRunner: warmup - waiting for ALLOCATE response...");
  pm::PMResponse response{};
  while (!pipelineManager->try_pop_response(response)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  auto slotId = response.slot_id;
  TT_LOG_INFO("SpPipelineRunner: warmup - got slot_id={}", slotId);
  if (slotId == pm::INVALID_SLOT) {
    TT_LOG_ERROR("SpPipelineRunner: Warmup failed with error");
    return false;
  }

  TT_LOG_INFO("SpPipelineRunner: warmup - pushing SUBMIT request...");
  pipelineManager->push_request(utils::makeSubmitRequest(slotId, *warmupSeq));
  const int maxAttempts = 1000;
  int attempts = 0;
  bool receivedToken = false;
  auto output = pm::OutputMessage{};

  while (attempts < maxAttempts && !receivedToken) {
    receivedToken = pipelineManager->try_pop_output(output);
    if (receivedToken) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    attempts++;
  }

  if (!receivedToken) {
    TT_LOG_ERROR("SpPipelineRunner: Warmup timed out waiting for token");
    return false;
  }

  TT_LOG_INFO("SpPipelineRunner: Warmup successful");
  pipelineManager->push_request(utils::makeCancelRequest(slotId));
  return true;
}

void SpPipelineRunner::stop() {
  stopped.store(true, std::memory_order_relaxed);
}

void SpPipelineRunner::step() {
  auto memoryRequest = getMemoryRequest();
  if (memoryRequest.has_value()) {
    handleMemoryRequest(*memoryRequest);
  }
  auto response = getResponse();
  if (response.has_value()) {
    handleResponse(*response);
  }
  auto output = getOutput();
  if (output.has_value()) {
    handleOutput(*output);
  }
  auto request = getRequest();
  if (request) {
    handleRequest(std::move(request));
  }
}

std::optional<pm::PMResponse> SpPipelineRunner::getResponse() {
  pm::PMResponse response;
  if (pipelineManager->try_pop_response(response)) {
    return response;
  }
  return std::nullopt;
}

std::optional<pm::OutputMessage> SpPipelineRunner::getOutput() {
  pm::OutputMessage output;
  if (pipelineManager->try_pop_output(output)) {
    return output;
  }
  return std::nullopt;
}

std::unique_ptr<llm_engine::Sequence> SpPipelineRunner::getRequest() {
  auto req = taskQueue->tryPop();
  if (!req) return nullptr;
  return req;
}

inline void SpPipelineRunner::handleMemoryRequest(
    const tt::domain::ManageMemoryTask& request) {
  memoryManager->handleRequest(request);
}

inline void SpPipelineRunner::handleResponse(const pm::PMResponse& response) {
  memoryManager->handleResponse(response.request_id, response.slot_id);
}

inline std::optional<tt::domain::ManageMemoryTask>
SpPipelineRunner::getMemoryRequest() {
  return memoryManager->getRequest();
}

void SpPipelineRunner::handleOutput(const pm::OutputMessage& output) {
  auto it = running.find(output.slot_id);
  if (it == running.end()) {
    TT_LOG_ERROR("SpPipelineRunner: Output for unknown slot");
    return;
  }
  auto& seq = *it->second;
  seq.appendToken(output.token_id);
  bool finished = output.is_complete || stopTokenIds.count(output.token_id);
  ipc::pushToken(*resultQueue, seq.taskId, output.token_id, finished);
}

inline void SpPipelineRunner::evictSlot(uint32_t slotId) {
  running.erase(slotId);
}

void SpPipelineRunner::handleRequest(
    std::unique_ptr<llm_engine::Sequence> request) {
  auto slotId = request->getKVCacheAddress();
  assert(slotId != static_cast<uint64_t>(llm_engine::INVALID_KV_CACHE_ADDRESS));
  auto slot = static_cast<uint32_t>(slotId);

  auto it = running.find(slot);
  bool isNew = (it == running.end());

  if (!isNew && it->second->taskId != request->taskId) {
    TT_LOG_INFO(
        "SpPipelineRunner::handleRequest slot={} reused by new task {} "
        "(was task {}), treating as new SUBMIT",
        slot, request->taskId, it->second->taskId);
    pipelineManager->push_request(utils::makeCancelRequest(slot));
    running.erase(it);
    isNew = true;
  }

  if (isNew) {
    pipelineManager->push_request(utils::makeSubmitRequest(slotId, *request));
    running[slot] = std::move(request);
    return;
  }
  pipelineManager->push_request(utils::makeContinueRequest(slot, *request));
  running[slot] = std::move(request);
}

}  // namespace tt::runners
