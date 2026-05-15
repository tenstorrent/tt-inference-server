// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/blaze_runner/blaze_runner.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <services/memory_services/blaze_memory_manager.hpp>

#include "config/settings.hpp"
#include "ipc/helpers/token_push.hpp"
#include "utils/logger.hpp"
#include "worker/single_process_worker_metrics.hpp"

namespace {
using namespace tt_llm_engine::scheduler::decode;
using namespace tt_llm_engine::pipeline;
PipelineConfig makePipelineConfig(const tt::config::LLMConfig& config) {
  switch (config.runner_type) {
    case tt::config::ModelRunnerType::PIPELINE_MANAGER:
      return SocketConfig{
          .h2d_socket_id = tt::config::blazeSocketDescriptorPrefix() + "_h2d",
          .d2h_socket_id = tt::config::blazeSocketDescriptorPrefix() + "_d2h",
          .connect_timeout_ms = tt::config::pmConnectTimeoutMs(),
          .use_deepseek_md_format = tt::config::useDeepseekMdFormat()};
    case tt::config::ModelRunnerType::MOCK_PIPELINE:
      return PipelineSimulatorConfig{
          .num_stages = 64,
          .stage_duration_us = 44,
          .decode_token_id = 12345,
      };
      /* spec decode config
       return PipelineSimulatorConfig{
          .num_stages = 64,
          .stage_duration_us = 44,
          .accept_rate = 0.9f,
          .safe_vocab_base = 1000,    // anything safely above your tokenizer's
      stop ids .safe_vocab_modulus = 64,   // any size >= 5; bigger = lower
      coincidental-stop chance
      };
       */
    default:
      throw std::runtime_error("Invalid blaze runner type");
  }
}
}  // namespace

namespace tt::runners {
namespace utils = blaze_utils;

BlazeRunner::BlazeRunner(const config::LLMConfig& config,
                         ipc::IResultQueue* resultQueue,
                         tt::ipc::ITaskQueue* taskQueue,
                         tt::ipc::ICancelQueue* cancelQueue)
    : config(config),
      stopTokenIds(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      resultQueue(resultQueue),
      taskQueue(taskQueue),
      cancelQueue(cancelQueue),
      lastOutputTime(std::chrono::steady_clock::now()),
      outputHangTimeout(tt::config::outputHangTimeoutMs()) {
  TT_LOG_INFO("BlazeRunner: Constructing DecodeScheduler with SocketConfig...");
  auto pipelineConfig = makePipelineConfig(config);
  ds::SchedulerParams managerParams{
      .max_users = static_cast<uint32_t>(tt::config::dsMaxUsers())};
  decodeScheduler =
      std::make_unique<ds::DecodeScheduler>(pipelineConfig, managerParams);
  TT_LOG_INFO("BlazeRunner: DecodeScheduler constructed, calling start()...");
  decodeScheduler->start();
  TT_LOG_INFO(
      "BlazeRunner: PipelineManager started, creating MemoryManager...");
  memoryManager = std::make_unique<tt::services::BlazeMemoryManager>(
      *decodeScheduler, [this](uint32_t slotId) { evictSlot(slotId); });
  TT_LOG_INFO("BlazeRunner: Constructor complete");
}

BlazeRunner::~BlazeRunner() {
  stop();
  if (decodeScheduler) {
    decodeScheduler->stop();
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
  tt::domain::llm::SamplingParams warmupParams;
  warmupParams.max_tokens = 1;
  warmupParams.ignore_eos = true;

  std::vector<int64_t> warmupTokens = {1};
  uint32_t warmupTaskId = 0;

  auto warmupSeq = std::make_unique<tt::domain::llm::Sequence>(
      warmupTaskId, 1, warmupTokens, warmupParams);

  constexpr uint32_t warmupAllocateRequestId = 0;
  constexpr uint32_t warmupEvictRequestId = 1;

  const auto timeout = std::chrono::milliseconds(tt::config::warmupTimeoutMs());
  const auto pollInterval = std::chrono::milliseconds(10);

  TT_LOG_INFO("BlazeRunner: warmup - pushing ALLOCATE request...");
  decodeScheduler->push_request(
      utils::makeAllocateRequest(warmupAllocateRequestId));

  TT_LOG_INFO("BlazeRunner: warmup - waiting for ALLOCATE response...");
  ds::SchedulerResponse response{};
  const auto allocateDeadline = std::chrono::steady_clock::now() + timeout;
  while (!decodeScheduler->try_pop_response(response)) {
    if (std::chrono::steady_clock::now() >= allocateDeadline) {
      TT_LOG_ERROR(
          "[BlazeRunner] Warmup timed out waiting for ALLOCATE response after "
          "{} ms",
          timeout.count());
      return false;
    }
    std::this_thread::sleep_for(pollInterval);
  }

  auto slotId = response.slot_id;
  TT_LOG_INFO("BlazeRunner: warmup - got slot_id={}", slotId);
  if (slotId == ds::INVALID_SLOT) {
    TT_LOG_ERROR("BlazeRunner: Warmup failed with error");
    return false;
  }

  TT_LOG_INFO("BlazeRunner: warmup - pushing SUBMIT request...");
  decodeScheduler->push_request(utils::makeSubmitRequest(slotId, *warmupSeq));

  const auto deadline = std::chrono::steady_clock::now() + timeout;
  bool receivedToken = false;
  auto output = ds::OutputMessage{};

  while (std::chrono::steady_clock::now() < deadline) {
    if (decodeScheduler->try_pop_output(output)) {
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

  TT_LOG_INFO("BlazeRunner: warmup - pushing EVICT request (slotId={})...",
              slotId);
  decodeScheduler->push_request(
      utils::makeEvictRequest(warmupEvictRequestId, slotId));
  ds::SchedulerResponse evictResponse{};
  const auto evictDeadline = std::chrono::steady_clock::now() + timeout;
  while (!decodeScheduler->try_pop_response(evictResponse)) {
    if (std::chrono::steady_clock::now() >= evictDeadline) {
      TT_LOG_ERROR(
          "[BlazeRunner] Warmup timed out waiting for EVICT ack after {} ms "
          "(slotId={})",
          timeout.count(), slotId);
      return false;
    }
    std::this_thread::sleep_for(pollInterval);
  }
  TT_LOG_INFO("BlazeRunner: warmup - got EVICT ack (slotId={})", slotId);
  TT_LOG_INFO("BlazeRunner: Warmup successful");
  return true;
}

void BlazeRunner::stop() { stopped.store(true, std::memory_order_relaxed); }

void BlazeRunner::step() {
  tt::worker::SingleProcessWorkerMetrics::instance().updateStepHeartbeat();
  drainAndHandleMemoryResponses();
  drainAndHandleOutputs();
  drainAndHandleCancelRequests();
  auto memoryRequest = getMemoryRequest();
  if (memoryRequest.has_value()) {
    TT_LOG_DEBUG("[BlazeRunner] step: got memoryRequest taskId={}, action={}",
                 memoryRequest->taskId,
                 static_cast<int>(memoryRequest->action));
    handleMemoryRequest(*memoryRequest);
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

void BlazeRunner::drainAndHandleMemoryResponses() {
  ds::SchedulerResponse response;
  size_t drained = 0;
  while (drained < tt::config::dsMaxUsers() &&
         decodeScheduler->try_pop_response(response)) {
    handleMemoryResponse(response);
    drained++;
  }
}

void BlazeRunner::drainAndHandleOutputs() {
  ds::OutputMessage output;
  size_t drained = 0;
  while (drained < tt::config::dsMaxUsers() &&
         decodeScheduler->try_pop_output(output)) {
    handleOutput(output);
    drained++;
  }
}

std::unique_ptr<tt::domain::llm::Sequence> BlazeRunner::getRequest() {
  if (requestToRetry) {
    return std::move(requestToRetry);
  }
  auto req = taskQueue->tryPop();
  if (!req) return nullptr;
  return req;
}

inline void BlazeRunner::handleMemoryRequest(
    const tt::domain::ManageMemoryTask& request) {
  if (request.action == tt::domain::MemoryManagementAction::DEALLOCATE) {
    auto it = pendingSubmits.find(request.slotId);
    if (it != pendingSubmits.end()) {
      TT_LOG_DEBUG(
          "[BlazeRunner] handleMemoryRequest: DEALLOCATE for slotId={} "
          "discarding pendingSubmit (expectedStopRequestId={}, hasSequence={})",
          request.slotId, it->second.expectedStopRequestId,
          static_cast<bool>(it->second.sequence));
      pendingSubmits.erase(it);
    }
  }
  memoryManager->handleRequest(request);
}

inline void BlazeRunner::handleMemoryResponse(
    const ds::SchedulerResponse& response) {
  if (auto it = pendingSubmits.find(response.slot_id);
      it != pendingSubmits.end() &&
      response.request_id == it->second.expectedStopRequestId) {
    auto pending = std::move(it->second);
    pendingSubmits.erase(it);
    TT_LOG_DEBUG(
        "[BlazeRunner] handleMemoryResponse: STOP ack received "
        "(slotId={}, taskId={}, hasDeferredSequence={})",
        response.slot_id, response.request_id,
        static_cast<bool>(pending.sequence));
    if (auto sit = slotContexts.find(response.slot_id);
        sit != slotContexts.end()) {
      slotContexts.erase(sit);
      tt::worker::SingleProcessWorkerMetrics::instance()
          .decrementActiveRequests();
    }
    if (pending.sequence) {
      handleRequest(std::move(pending.sequence));
    }
    return;
  }
  memoryManager->handleResponse(response.request_id, response.slot_id);
}

inline std::optional<tt::domain::ManageMemoryTask>
BlazeRunner::getMemoryRequest() {
  return memoryManager->getRequest();
}

void BlazeRunner::drainAndHandleCancelRequests() {
  std::vector<uint32_t> taskIds;
  cancelQueue->tryPopAll(taskIds);
  for (auto taskId : taskIds) {
    handleCancelRequest(taskId);
  }
}

inline void BlazeRunner::handleCancelRequest(uint32_t taskId) {
  if (requestToRetry && requestToRetry->taskId == taskId) {
    TT_LOG_DEBUG(
        "[BlazeRunner] handleCancelRequest: dropping queued retry "
        "(taskId={}, slotId={})",
        taskId, requestToRetry->getKVCacheSlot());
    requestToRetry = nullptr;
    return;
  }
  for (const auto& [slotId, ctx] : slotContexts) {
    if (ctx.taskId != taskId) continue;
    if (decodeScheduler->push_request(utils::makeStopRequest(taskId, slotId))) {
      TT_LOG_DEBUG(
          "[BlazeRunner] handleCancelRequest: pushed STOP "
          "(taskId={}, slotId={})",
          taskId, slotId);
      pendingSubmits.insert_or_assign(
          slotId, blaze_utils::PendingSubmit{.expectedStopRequestId = taskId,
                                             .sequence = nullptr});
    } else {
      TT_LOG_DEBUG(
          "[BlazeRunner] handleCancelRequest: scheduler rejected STOP "
          "(taskId={}, slotId={}); slot is already being evicted, request "
          "will be torn down by the eviction",
          taskId, slotId);
    }
    return;
  }
  TT_LOG_DEBUG(
      "[BlazeRunner] handleCancelRequest: no active slot for taskId={} "
      "(already finished or not yet allocated)",
      taskId);
}

void BlazeRunner::handleOutput(const ds::OutputMessage& output) {
  tt::worker::SingleProcessWorkerMetrics::instance().updateOutputHeartbeat();
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

  uint32_t specAccepts = 0;
  uint32_t specRejects = 0;

  context.tokensGenerated++;
  if (finished) {
    specAccepts = decodeScheduler->get_spec_accepts(output.slot_id) -
                  context.specAcceptsAtStart;
    specRejects = decodeScheduler->get_spec_rejects(output.slot_id) -
                  context.specRejectsAtStart;
    uint32_t specTotal = specAccepts + specRejects;
    double acceptRate = specTotal > 0 ? 100.0 * specAccepts / specTotal : 0.0;
    TT_LOG_INFO(
        "slot {} turn: accepts={}/{} rate={:.1f}% taskId={} token_id={} "
        "is_complete={} ignoreEos={} hitStop={} tokensGenerated={}",
        output.slot_id, specAccepts, specTotal, acceptRate, taskId,
        output.token_id, output.is_complete, context.ignoreEos, hitStop,
        context.tokensGenerated);
    slotContexts.erase(output.slot_id);
    tt::worker::SingleProcessWorkerMetrics::instance()
        .decrementActiveRequests();
  }

  ipc::helpers::pushToken(*resultQueue, taskId, output.token_id, finished,
                          specAccepts, specRejects);
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
  if (auto pit = pendingSubmits.find(slotId); pit != pendingSubmits.end()) {
    TT_LOG_DEBUG(
        "[BlazeRunner] evictSlot: slotId={} discarding pendingSubmit "
        "(expectedStopRequestId={}, hasSequence={})",
        slotId, pit->second.expectedStopRequestId,
        static_cast<bool>(pit->second.sequence));
    pendingSubmits.erase(pit);
  }
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

void BlazeRunner::handleRequest(
    std::unique_ptr<tt::domain::llm::Sequence> request) {
  auto slotId = request->getKVCacheSlot();
  assert(slotId != tt::domain::INVALID_SLOT_ID);
  assert(slotId < tt::config::dsMaxUsers());

  if (auto it = pendingSubmits.find(slotId); it != pendingSubmits.end()) {
    TT_LOG_DEBUG(
        "[BlazeRunner] handleRequest: deferring submit for slotId={} "
        "until STOP ack arrives (taskId={}, expectedStopRequestId={}, "
        "hadDeferredSequence={})",
        slotId, request->taskId, it->second.expectedStopRequestId,
        static_cast<bool>(it->second.sequence));
    it->second.sequence = std::move(request);
    return;
  }

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
  ds::ISRequest req = isNew ? utils::makeSubmitRequest(slotId, *request)
                            : utils::makeContinueRequest(slotId, *request);
  if (!decodeScheduler->push_request(req)) {
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
                  decodeScheduler->get_spec_accepts(slotId),
                  decodeScheduler->get_spec_rejects(slotId)});
  tt::worker::SingleProcessWorkerMetrics::instance().incrementActiveRequests();
}

}  // namespace tt::runners
