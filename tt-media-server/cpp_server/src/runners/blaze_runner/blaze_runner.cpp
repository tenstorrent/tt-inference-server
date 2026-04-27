// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/sp_pipeline_runner/blaze_runner.hpp"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <pipeline_manager/pipeline_manager_types.hpp>
#include <thread>

#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "domain/sequence.hpp"
#include "ipc/token_push.hpp"
#include "runners/sp_pipeline_runner/blaze_utils.hpp"
#include "services/memory_services/blaze_memory_manager.hpp"
#include "utils/logger.hpp"
#include "worker/single_process_worker_metrics.hpp"

namespace tt::runners {
namespace utils = blaze_utils;

BlazeRunner::BlazeRunner(const config::LLMConfig& config,
                         ipc::IResultQueue* resultQueue,
                         tt::ipc::ITaskQueue* taskQueue)
    : config(config),
      stopTokenIds(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      resultQueue(resultQueue),
      taskQueue(taskQueue),
      lastOutputTime(std::chrono::steady_clock::now()),
      outputHangTimeout(tt::config::outputHangTimeoutMs()) {
  TT_LOG_INFO("BlazeRunner: Constructing PipelineManager with SocketConfig...");
  pm::SocketConfig socketConfig{
      .h2d_socket_id = tt::config::blazeSocketDescriptorPrefix() + "_h2d",
      .d2h_socket_id = tt::config::blazeSocketDescriptorPrefix() + "_d2h",
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
  tt::domain::SamplingParams warmupParams;
  warmupParams.max_tokens = 1;
  warmupParams.ignore_eos = true;

  std::vector<int64_t> warmupTokens = {1};
  uint32_t warmupTaskId = 0;

  auto warmupSeq = std::make_unique<tt::domain::Sequence>(
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
  tt::worker::SingleProcessWorkerMetrics::instance().updateStepHeartbeat();
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
  checkOutputHang();
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

std::unique_ptr<tt::domain::Sequence> BlazeRunner::getRequest() {
  if (requestToRetry) {
    return std::move(requestToRetry);
  }
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
  auto& metrics = tt::worker::SingleProcessWorkerMetrics::instance();
  metrics.updateOutputHeartbeat();
  lastOutputTime = std::chrono::steady_clock::now();
  auto it = slotContexts.find(output.slot_id);
  if (it == slotContexts.end()) {
    TT_LOG_ERROR(
        "[BlazeRunner] handleOutput: output for unknown slot_id={}, "
        "token_id={}, is_complete={}",
        output.slot_id, output.token_id, output.is_complete);
    return;
  }
  auto& context = it->second;
  bool hitStop = !context.ignoreEos && stopTokenIds.count(output.token_id) > 0;
  bool finished = output.is_complete || hitStop;
  auto taskId = context.taskId;
  ipc::pushToken(*resultQueue, taskId, output.token_id, finished);
  metrics.onOutputToken(output.slot_id);
  if (finished) {
    uint32_t specAccepts = pipelineManager->get_spec_accepts(output.slot_id) -
                           context.specAcceptsAtStart;
    uint32_t specRejects = pipelineManager->get_spec_rejects(output.slot_id) -
                           context.specRejectsAtStart;
    uint32_t specTotal = specAccepts + specRejects;
    double acceptRate = specTotal > 0 ? 100.0 * specAccepts / specTotal : 0.0;
    TT_LOG_INFO("slot {} turn: accepts={}/{} rate={:.1f}%", output.slot_id,
                specAccepts, specTotal, acceptRate);
    metrics.onTurnComplete(output.slot_id, specAccepts, specRejects);
    slotContexts.erase(output.slot_id);
    metrics.decrementActiveRequests();
  }
}

void BlazeRunner::checkOutputHang() {
  if (slotContexts.empty()) {
    lastOutputTime = std::chrono::steady_clock::now();
    return;
  }
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - lastOutputTime);
  if (elapsed <= outputHangTimeout) {
    return;
  }
  TT_LOG_CRITICAL(
      "[BlazeRunner] Output hang detected: no model output for {} ms with {} "
      "in-flight generation(s) (threshold={} ms). Self-terminating worker so "
      "infrastructure can restart the server.",
      elapsed.count(), slotContexts.size(), outputHangTimeout.count());
  // Use abort() so the existing fatalSignalHandler prints a visible
  // "killed by signal SIGABRT" line and the WorkerManager parent logs the
  // worker crash. Skipping destructors is acceptable here: the model/device
  // is already wedged and we want the worker gone ASAP.
  std::abort();
}

inline void BlazeRunner::evictSlot(uint32_t slotId) {
  auto it = slotContexts.find(slotId);
  if (it != slotContexts.end()) {
    TT_LOG_DEBUG("[BlazeRunner] evictSlot: slotId={}, had taskId={}", slotId,
                 it->second.taskId);
    slotContexts.erase(it);
    tt::worker::SingleProcessWorkerMetrics::instance()
        .decrementActiveRequests();
    return;
  }
  TT_LOG_DEBUG("[BlazeRunner] evictSlot: slotId={} (no slot context)", slotId);
}

void BlazeRunner::handleRequest(std::unique_ptr<tt::domain::Sequence> request) {
  auto slotId = request->getKVCacheSlot();
  assert(slotId != tt::domain::INVALID_SLOT_ID);
  assert(slotId < tt::config::pmMaxUsers());

  bool isNew = !request->isContinuation() && !request->isDisaggregated();
  if (isNew && request->getSamplingParams().hasGuidedDecoding()) {
    TT_LOG_WARN(
        "[BlazeRunner] task_id={} has response_format constraint but "
        "SP Pipeline does not support per-step guided decoding yet. "
        "Output may not conform to the requested schema.",
        request->taskId);
  }

  TT_LOG_DEBUG(
      "[BlazeRunner] handleRequest: taskId={}, slotId={}, isNew={}, "
      "isContinuation={}, numPromptTokens={}, totalTokens={}, runningSlots={}",
      request->taskId, slotId, isNew, request->isContinuation(),
      request->getNumPromptTokens(), request->getTokenIds().size(),
      slotContexts.size());
  tt_blaze::pipeline_manager::ISRequest req =
      isNew ? utils::makeSubmitRequest(slotId, *request)
            : utils::makeContinueRequest(slotId, *request);
  if (!pipelineManager->push_request(req)) {
    TT_LOG_DEBUG(
        "[BlazeRunner] handleRequest: failed to push request, taskId={}, "
        "slotId={}",
        request->taskId, slotId);
    requestToRetry = std::move(request);
    return;
  }
  if (slotContexts.empty()) {
    lastOutputTime = std::chrono::steady_clock::now();
  }
  slotContexts.insert_or_assign(
      slotId, blaze_utils::SlotContext{
                  request->taskId, request->getSamplingParams().ignore_eos,
                  pipelineManager->get_spec_accepts(slotId),
                  pipelineManager->get_spec_rejects(slotId)});
  auto& metrics = tt::worker::SingleProcessWorkerMetrics::instance();
  metrics.incrementActiveRequests();
  // Reset the slot's per-turn counters every time a request is bound to it
  // (fresh, continuation, or disaggregated). Skipping continuations would
  // leave the previous turn's OSL counter and first-token timestamp in place,
  // breaking TPOT/OSL exposure for any multi-turn slot.
  metrics.onTurnStart(slotId,
                      static_cast<uint32_t>(request->getTokenIds().size()));
}

}  // namespace tt::runners
