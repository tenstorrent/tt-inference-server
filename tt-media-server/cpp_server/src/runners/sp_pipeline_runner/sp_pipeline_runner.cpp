// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <pipeline_manager/pipeline_manager_types.hpp>
#include <thread>

#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/token_push.hpp"
#include "llm_runner/sequence.hpp"
#include "runners/sp_pipeline_runner/sp_pipeline_utils.hpp"
#include "services/memory_services/sp_pipeline_memory_manager.hpp"
#include "utils/logger.hpp"

namespace tt::runners {
namespace utils = sp_pipeline_utils;

SpPipelineRunner::SpPipelineRunner(
    const config::LLMConfig& config, ipc::TokenRingBuffer<65536>* resultQueue,
    tt::runners::llm_engine::ITaskQueue* taskQueue)
    : config(config),
      stopTokenIds(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      resultQueue(resultQueue),
      taskQueue(taskQueue) {
  TT_LOG_INFO(
      "SpPipelineRunner: Constructing PipelineManager with SocketConfig...");
  pm::SocketConfig socketConfig{
      .h2d_socket_id = tt::config::h2dSocketId(),
      .d2h_socket_id = tt::config::d2hSocketId(),
      .connect_timeout_ms = tt::config::pmConnectTimeoutMs(),
      .use_deepseek_md_format = tt::config::useDeepseekMdFormat()};
  pm::ManagerParams managerParams{
      .max_users = static_cast<uint32_t>(tt::config::pmMaxUsers())};
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
  tt::runners::llm_engine::SamplingParams warmupParams;
  warmupParams.max_tokens = 1;
  warmupParams.ignore_eos = true;

  std::vector<int64_t> warmupTokens = {1};
  uint32_t warmupTaskId = 0;

  auto warmupSeq = std::make_unique<tt::runners::llm_engine::Sequence>(
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
    TT_LOG_ERROR("[SpPipelineRunner] Warmup timed out waiting for token");
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
    TT_LOG_DEBUG(
        "[SpPipelineRunner] step: got memoryRequest taskId={}, action={}",
        memoryRequest->taskId, static_cast<int>(memoryRequest->action));
    handleMemoryRequest(*memoryRequest);
  }
  auto response = getResponse();
  if (response.has_value()) {
    TT_LOG_DEBUG(
        "[SpPipelineRunner] step: got PMResponse request_id={}, slot_id={}",
        response->request_id, response->slot_id);
    handleResponse(*response);
  }
  auto output = getOutput();
  if (output.has_value()) {
    handleOutput(*output);
  }
  auto request = getRequest();
  if (request) {
    TT_LOG_DEBUG(
        "[SpPipelineRunner] step: got Sequence taskId={}, slotId={}, "
        "numPromptTokens={}, totalTokens={}",
        request->taskId, request->getKVCacheSlot(),
        request->getNumPromptTokens(), request->getTokenIds().size());
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

std::unique_ptr<tt::runners::llm_engine::Sequence>
SpPipelineRunner::getRequest() {
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
    TT_LOG_ERROR(
        "[SpPipelineRunner] handleOutput: output for unknown slot_id={}, "
        "token_id={}, is_complete={}",
        output.slot_id, output.token_id, output.is_complete);
    return;
  }
  auto& seq = *it->second;
  seq.appendToken(output.token_id);
  bool finished = output.is_complete || stopTokenIds.count(output.token_id);
  ipc::pushToken(*resultQueue, seq.taskId, output.token_id, finished);
}

inline void SpPipelineRunner::evictSlot(uint32_t slotId) {
  auto it = running.find(slotId);
  if (it != running.end()) {
    TT_LOG_DEBUG(
        "[SpPipelineRunner] evictSlot: slotId={}, was running taskId={}",
        slotId, it->second->taskId);
  } else {
    TT_LOG_DEBUG("[SpPipelineRunner] evictSlot: slotId={} (not in running map)",
                 slotId);
  }
  running.erase(slotId);
}

void SpPipelineRunner::handleRequest(
    std::unique_ptr<tt::runners::llm_engine::Sequence> request) {
  auto slotId = request->getKVCacheSlot();
  assert(slotId != tt::domain::INVALID_SLOT_ID);

  auto it = running.find(slotId);
  bool isNew = (it == running.end());

  TT_LOG_DEBUG(
      "[SpPipelineRunner] handleRequest: taskId={}, slotId={}, isNew={}, "
      "numPromptTokens={}, totalTokens={}, runningSlots={}",
      request->taskId, slotId, isNew, request->getNumPromptTokens(),
      request->getTokenIds().size(), running.size());

  if (!isNew && it->second->taskId != request->taskId) {
    TT_LOG_INFO(
        "SpPipelineRunner::handleRequest slot={} reused by new task {} "
        "(was task {}), treating as new SUBMIT",
        slotId, request->taskId, it->second->taskId);
    pipelineManager->push_request(utils::makeCancelRequest(slotId));
    running.erase(it);
    isNew = true;
  }

  if (isNew) {
    TT_LOG_DEBUG(
        "[SpPipelineRunner] handleRequest: SUBMIT taskId={}, slotId={}",
        request->taskId, slotId);
    pipelineManager->push_request(utils::makeSubmitRequest(slotId, *request));
    running[slotId] = std::move(request);
    return;
  }
  TT_LOG_DEBUG(
      "[SpPipelineRunner] handleRequest: CONTINUE taskId={}, slotId={}",
      request->taskId, slotId);
  pipelineManager->push_request(utils::makeContinueRequest(slotId, *request));
  running[slotId] = std::move(request);
}

}  // namespace tt::runners
