// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/sp_pipeline_runner/blaze_runner.hpp"

#include <cassert>
#include <cstring>
#include <pipeline_manager/pipeline_manager_types.hpp>
#include <thread>

#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/token_push.hpp"
#include "llm_runner/sequence.hpp"
#include "runners/sp_pipeline_runner/blaze_utils.hpp"
#include "services/memory_services/blaze_memory_manager.hpp"
#include "utils/logger.hpp"

namespace tt::runners {
namespace utils = blaze_utils;

BlazeRunner::BlazeRunner(const config::LLMConfig& config,
                         ipc::IResultQueue* resultQueue,
                         tt::runners::llm_engine::ITaskQueue* taskQueue)
    : config(config),
      stopTokenIds(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      resultQueue(resultQueue),
      taskQueue(taskQueue) {
  TT_LOG_INFO("BlazeRunner: Constructing PipelineManager with SocketConfig...");
  pm::SocketConfig socketConfig{
      .h2d_socket_id = tt::config::h2dSocketId(),
      .d2h_socket_id = tt::config::d2hSocketId(),
      .connect_timeout_ms = tt::config::pmConnectTimeoutMs(),
      .use_deepseek_md_format = tt::config::useDeepseekMdFormat()};
  // pm::MockConfig mock = {};
  pm::ManagerParams managerParams{
      .max_users = static_cast<uint32_t>(tt::config::pmMaxUsers())};
  pipelineManager =
      std::make_unique<pm::PipelineManager>(socketConfig, managerParams);
  TT_LOG_INFO("BlazeRunner: PipelineManager constructed, calling start()...");
  pipelineManager->start();
  TT_LOG_INFO(
      "BlazeRunner: PipelineManager started, creating MemoryManager...");
  memoryManager = std::make_unique<tt::services::BlazeMemoryManager>(
      *pipelineManager, [this](uint32_t slotId) { evictSlot(slotId); });
  TT_LOG_INFO("BlazeRunner: Constructor complete");
}

BlazeRunner::~BlazeRunner() {
  stop();
  if (pipelineManager) {
    pipelineManager->stop();
  }
}

void BlazeRunner::run() {
  while (!stopped.load(std::memory_order_relaxed)) {
    try {
      step();
    } catch (const std::exception& e) {
      TT_LOG_ERROR("BlazeRunner: Exception in run: {}", e.what());
      throw;
    }
  }
}

bool BlazeRunner::warmup() {
  tt::runners::llm_engine::SamplingParams warmupParams;
  warmupParams.max_tokens = 1;
  warmupParams.ignore_eos = true;

  std::vector<int64_t> warmupTokens = {1};
  uint32_t warmupTaskId = 0;

  auto warmupSeq = std::make_unique<tt::runners::llm_engine::Sequence>(
      warmupTaskId, 1, warmupTokens, warmupParams);

  TT_LOG_INFO("BlazeRunner: warmup - pushing ALLOCATE request...");
  pipelineManager->push_request(utils::makeAllocateRequest(0));

  TT_LOG_INFO("BlazeRunner: warmup - waiting for ALLOCATE response...");
  pm::PMResponse response{};
  while (!pipelineManager->try_pop_response(response)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  auto slotId = response.slot_id;
  TT_LOG_INFO("BlazeRunner: warmup - got slot_id={}", slotId);
  if (slotId == pm::INVALID_SLOT) {
    TT_LOG_ERROR("BlazeRunner: Warmup failed with error");
    return false;
  }

  TT_LOG_INFO("BlazeRunner: warmup - pushing SUBMIT request...");
  pipelineManager->push_request(utils::makeSubmitRequest(slotId, *warmupSeq));

  const auto timeout = std::chrono::milliseconds(tt::config::warmupTimeoutMs());
  const auto pollInterval = std::chrono::milliseconds(10);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  bool receivedToken = false;
  auto output = pm::OutputMessage{};

  while (std::chrono::steady_clock::now() < deadline) {
    if (pipelineManager->try_pop_output(output)) {
      receivedToken = true;
      break;
    }
    std::this_thread::sleep_for(pollInterval);
  }

  if (!receivedToken) {
    TT_LOG_ERROR("[BlazeRunner] Warmup timed out waiting for token after {} ms",
                 timeout.count());
    return false;
  }

  TT_LOG_INFO("BlazeRunner: Warmup successful");
  pipelineManager->push_request(utils::makeCancelRequest(slotId));
  return true;
}

void BlazeRunner::stop() { stopped.store(true, std::memory_order_relaxed); }

void BlazeRunner::step() {
  auto memoryRequest = getMemoryRequest();
  if (memoryRequest.has_value()) {
    TT_LOG_DEBUG("[BlazeRunner] step: got memoryRequest taskId={}, action={}",
                 memoryRequest->taskId,
                 static_cast<int>(memoryRequest->action));
    handleMemoryRequest(*memoryRequest);
  }
  auto response = getResponse();
  if (response.has_value()) {
    TT_LOG_DEBUG("[BlazeRunner] step: got PMResponse request_id={}, slot_id={}",
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
        "[BlazeRunner] step: got Sequence taskId={}, slotId={}, "
        "numPromptTokens={}, totalTokens={}",
        request->taskId, request->getKVCacheSlot(),
        request->getNumPromptTokens(), request->getTokenIds().size());
    handleRequest(std::move(request));
  }
}

std::optional<pm::PMResponse> BlazeRunner::getResponse() {
  pm::PMResponse response;
  if (pipelineManager->try_pop_response(response)) {
    return response;
  }
  return std::nullopt;
}

std::optional<pm::OutputMessage> BlazeRunner::getOutput() {
  pm::OutputMessage output;
  if (pipelineManager->try_pop_output(output)) {
    return output;
  }
  return std::nullopt;
}

std::unique_ptr<tt::runners::llm_engine::Sequence> BlazeRunner::getRequest() {
  auto req = taskQueue->tryPop();
  if (!req) return nullptr;
  return req;
}

inline void BlazeRunner::handleMemoryRequest(
    const tt::domain::ManageMemoryTask& request) {
  memoryManager->handleRequest(request);
}

inline void BlazeRunner::handleResponse(const pm::PMResponse& response) {
  memoryManager->handleResponse(response.request_id, response.slot_id);
}

inline std::optional<tt::domain::ManageMemoryTask>
BlazeRunner::getMemoryRequest() {
  return memoryManager->getRequest();
}

void BlazeRunner::handleOutput(const pm::OutputMessage& output) {
  auto it = running.find(output.slot_id);
  if (it == running.end()) {
    TT_LOG_ERROR(
        "[BlazeRunner] handleOutput: output for unknown slot_id={}, "
        "token_id={}, is_complete={}",
        output.slot_id, output.token_id, output.is_complete);
    return;
  }
  auto& seq = *it->second;
  seq.appendToken(output.token_id);
  bool hitStop = !seq.getSamplingParams().ignore_eos &&
                 stopTokenIds.count(output.token_id) > 0;
  bool finished = output.is_complete || hitStop;
  auto taskId = seq.taskId;
  ipc::pushToken(*resultQueue, taskId, output.token_id, finished);
}

inline void BlazeRunner::evictSlot(uint32_t slotId) {
  auto it = running.find(slotId);
  if (it != running.end()) {
    TT_LOG_DEBUG("[BlazeRunner] evictSlot: slotId={}, was running taskId={}",
                 slotId, it->second->taskId);
  } else {
    TT_LOG_DEBUG("[BlazeRunner] evictSlot: slotId={} (not in running map)",
                 slotId);
  }
  running.erase(slotId);
}

void BlazeRunner::handleRequest(
    std::unique_ptr<tt::runners::llm_engine::Sequence> request) {
  auto slotId = request->getKVCacheSlot();
  assert(slotId != tt::domain::INVALID_SLOT_ID);

  bool isNew = !request->isContinuation();

  TT_LOG_DEBUG(
      "[BlazeRunner] handleRequest: taskId={}, slotId={}, isNew={}, "
      "isContinuation={}, numPromptTokens={}, totalTokens={}, runningSlots={}",
      request->taskId, slotId, isNew, request->isContinuation(),
      request->getNumPromptTokens(), request->getTokenIds().size(),
      running.size());

  if (isNew) {
    if (request->getSamplingParams().hasGuidedDecoding()) {
      TT_LOG_WARN(
          "[BlazeRunner] task_id={} has response_format constraint but "
          "SP Pipeline does not support per-step guided decoding yet. "
          "Output may not conform to the requested schema.",
          request->taskId);
    }

    TT_LOG_DEBUG("[BlazeRunner] handleRequest: SUBMIT taskId={}, slotId={}",
                 request->taskId, slotId);
    pipelineManager->push_request(utils::makeSubmitRequest(slotId, *request));
    running[slotId] = std::move(request);
    return;
  } else {
    TT_LOG_DEBUG("[BlazeRunner] handleRequest: CONTINUE taskId={}, slotId={}",
                 request->taskId, slotId);
    pipelineManager->push_request(utils::makeContinueRequest(slotId, *request));
    running[slotId] = std::move(request);
  }
}

}  // namespace tt::runners
