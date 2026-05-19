// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/blaze_runner/blaze_runner.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <services/memory_services/blaze_memory_manager.hpp>

#include "blaze_runner/blaze_types.hpp"
#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/helpers/token_push.hpp"
#include "runners/blaze_runner/blaze_utils.hpp"
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
namespace types = blaze_types;

BlazeRunner::BlazeRunner(const config::LLMConfig& config,
                         ipc::IResultQueue* resultQueue,
                         tt::ipc::ITaskQueue* taskQueue,
                         tt::ipc::ICancelQueue* cancelQueue)
    : config(config),
      stopTokenIds(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      resultQueue(resultQueue),
      taskQueue(taskQueue),
      cancelQueue(cancelQueue),
      slotManager(tt::config::dsMaxUsers()),
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
  memoryManager = std::make_unique<tt::services::BlazeMemoryManager>();
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
  auto memoryRequest = getMemoryRequest();
  if (memoryRequest.has_value()) {
    TT_LOG_DEBUG("[BlazeRunner] step: got memoryRequest taskId={}, action={}",
                 memoryRequest->taskId,
                 static_cast<int>(memoryRequest->action));
    handleMemoryRequest(*memoryRequest);
  }
  drainAndHandleCancelRequests();
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
  switch (request.action) {
    case tt::domain::MemoryManagementAction::ALLOCATE: {
      handleAllocateRequest(request);
      break;
    }
    case tt::domain::MemoryManagementAction::DEALLOCATE: {
      handleEvictRequest(request);
      break;
    }
    default: {
      TT_LOG_ERROR(
          "[BlazeRunner] handleMemoryRequest: unexpected action for taskId={}, "
          "action={}",
          request.taskId, static_cast<int>(request.action));
      assert(false && "unexpected action for taskId");
      break;
    }
  }
}

inline void BlazeRunner::handleEvictRequest(
    const tt::domain::ManageMemoryTask& request) {
  auto& slotContext = slotManager.getSlotContext(request.slotId);
  auto evictRequest = utils::makeEvictRequest(request.taskId, request.slotId);
  switch (slotContext.state) {
    case types::SlotState::IDLE:
    case types::SlotState::RUNNING: {
      if (!decodeScheduler->push_request(evictRequest)) {
        TT_LOG_DEBUG(
            "[BlazeRunner] handleEvictRequest: failed to push evict request, "
            "taskId={}, slotId={}",
            request.taskId, request.slotId);
        pendingMemoryRetry = request;
        return;
      }
      // Capture before setSlotState changes the state.
      bool wasRunning = (slotContext.state == types::SlotState::RUNNING);
      if (wasRunning) {
        slotManager.unbindTaskFromSlot(*slotContext.taskId);
        tt::worker::SingleProcessWorkerMetrics::instance()
            .decrementActiveRequests();
      }
      slotContext.pendingAckRequestId = request.taskId;
      slotManager.setSlotState(request.slotId,
                               types::SlotState::AWAITING_EVICT_ACK);
      break;
    }
    case types::SlotState::AWAITING_STOP_ACK: {
      // Eviction supersedes any deferred submit; abort it so the client
      // doesn't hang waiting for tokens that will never come.
      if (slotContext.deferredSubmit) {
        auto droppedTaskId = slotContext.deferredSubmit->taskId;
        slotContext.deferredSubmit.reset();
        ipc::helpers::pushToken(*resultQueue, droppedTaskId, 0,
                                ipc::SharedToken::FLAG_ABORT, 0, 0);
      }
      slotContext.deferredEvict = std::move(evictRequest);
      break;
    }
    case types::SlotState::AWAITING_EVICT_ACK:
      TT_LOG_WARN(
          "[BlazeRunner] handleEvictRequest: DEALLOCATE for slotId={} already "
          "evicting, ignoring",
          request.slotId);
      break;
    case types::SlotState::FREE:
      TT_LOG_ERROR(
          "[BlazeRunner] handleEvictRequest: DEALLOCATE for FREE slotId={}",
          request.slotId);
      assert(false && "DEALLOCATE on FREE slot");
      break;
  }
}

inline void BlazeRunner::handleAllocateRequest(
    const tt::domain::ManageMemoryTask& request) {
  auto allocateRequest = utils::makeAllocateRequest(request.taskId);
  if (!decodeScheduler->push_request(allocateRequest)) {
    TT_LOG_DEBUG(
        "[BlazeRunner] handleMemoryRequest: failed to push allocate "
        "request, "
        "requestId={}",
        request.taskId);
    pendingMemoryRetry = request;
    return;
  }
  pendingAllocates.insert(request.taskId);
}

inline void BlazeRunner::handleMemoryResponse(
    const ds::SchedulerResponse& response) {
  auto taskId = response.request_id;
  if (pendingAllocates.count(taskId) > 0) {
    pendingAllocates.erase(taskId);
    if (response.slot_id != ds::INVALID_SLOT) {
      slotManager.setSlotState(response.slot_id, types::SlotState::IDLE);
      memoryManager->replyAllocateSuccess(taskId, response.slot_id);
    } else {
      memoryManager->replyAllocateFailure(taskId);
    }
    return;
  }
  auto slotId = response.slot_id;
  auto& slotContext = slotManager.getSlotContext(slotId);
  if (slotContext.pendingAckRequestId != taskId) {
    TT_LOG_ERROR(
        "[BlazeRunner] handleMemoryResponse: unexpected taskId={} for "
        "slotId={}",
        taskId, slotId);
    return;
  }
  switch (slotContext.state) {
    case types::SlotState::AWAITING_EVICT_ACK:
      slotManager.clearSlotContext(slotId);
      break;
    case types::SlotState::AWAITING_STOP_ACK:
      slotManager.setSlotAsIdle(slotId);
      if (slotContext.deferredEvict.has_value()) {
        handleEvictRequest(tt::domain::ManageMemoryTask{
            .taskId = slotContext.deferredEvict->request_id,
            .action = tt::domain::MemoryManagementAction::DEALLOCATE,
            .slotId = slotContext.deferredEvict->slot_id,
        });
        slotContext.deferredEvict = std::nullopt;
      } else if (slotContext.deferredSubmit) {
        handleRequest(std::move(
            slotContext.deferredSubmit));  // move clears the deferred submit
      }
      break;
    default:
      TT_LOG_ERROR(
          "[BlazeRunner] handleMemoryResponse: unexpected state for taskId={}, "
          "slotId={}",
          taskId, slotId);
      assert(false && "unexpected state for taskId");
      break;
  }
}

inline std::optional<tt::domain::ManageMemoryTask>
BlazeRunner::getMemoryRequest() {
  if (pendingMemoryRetry.has_value()) {
    auto task = std::move(*pendingMemoryRetry);
    pendingMemoryRetry.reset();
    return task;
  }
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
    TT_LOG_DEBUG("[BlazeRunner] cancel for taskId={}: dropping queued retry",
                 taskId);
    requestToRetry.reset();
    return;
  }
  auto slot = slotManager.getSlotContextByTaskId(taskId);
  if (!slot) {
    TT_LOG_ERROR(
        "[BlazeRunner] handleCancelRequest: unexpected cancel request for "
        "taskId={}",
        taskId);
    return;
  }
  if (slot->state != types::SlotState::RUNNING) {
    TT_LOG_ERROR("[BlazeRunner] handleCancelRequest: taskId={} is not running",
                 taskId);
    return;
  }
  if (!decodeScheduler->push_request(utils::makeStopRequest(
          taskId, slot->slotId))) {  // what if this fails? Queue is full?
    TT_LOG_ERROR(
        "[BlazeRunner] handleCancelRequest: failed to push stop request, "
        "taskId={}",
        taskId);
    return;
  }
  slot->pendingAckRequestId = taskId;
  slotManager.setSlotState(slot->slotId, types::SlotState::AWAITING_STOP_ACK);
  slotManager.unbindTaskFromSlot(taskId);
  ipc::helpers::pushToken(*resultQueue, taskId, 0, ipc::SharedToken::FLAG_ABORT,
                          0, 0);
  tt::worker::SingleProcessWorkerMetrics::instance().decrementActiveRequests();
}

void BlazeRunner::handleOutput(const ds::OutputMessage& output) {
  tt::worker::SingleProcessWorkerMetrics::instance().updateOutputHeartbeat();
  lastOutputTime = std::chrono::steady_clock::now();
  auto& slotContext = slotManager.getSlotContext(output.slot_id);
  if (slotContext.state != types::SlotState::RUNNING) {
    TT_LOG_ERROR("[BlazeRunner] handleOutput: slotId={} is not running",
                 output.slot_id);
    return;
  }
  bool hitStop =
      !slotContext.ignoreEos && stopTokenIds.count(output.token_id) > 0;
  bool finished = output.is_complete || hitStop;
  auto taskId = slotContext.taskId.value();

  uint32_t specAccepts = 0;
  uint32_t specRejects = 0;

  slotContext.tokensGenerated++;
  if (finished) {
    specAccepts = decodeScheduler->get_spec_accepts(output.slot_id) -
                  slotContext.specAcceptsAtStart;
    specRejects = decodeScheduler->get_spec_rejects(output.slot_id) -
                  slotContext.specRejectsAtStart;
    uint32_t specTotal = specAccepts + specRejects;
    double acceptRate = specTotal > 0 ? 100.0 * specAccepts / specTotal : 0.0;
    TT_LOG_INFO(
        "slot {} turn: accepts={}/{} rate={:.1f}% taskId={} token_id={} "
        "is_complete={} ignoreEos={} hitStop={} tokensGenerated={}",
        output.slot_id, specAccepts, specTotal, acceptRate, taskId,
        output.token_id, output.is_complete, slotContext.ignoreEos, hitStop,
        slotContext.tokensGenerated);
    slotManager.setSlotAsIdle(output.slot_id);
    tt::worker::SingleProcessWorkerMetrics::instance()
        .decrementActiveRequests();
  }
  uint32_t flag = finished ? ipc::SharedToken::FLAG_FINAL : 0;
  ipc::helpers::pushToken(*resultQueue, taskId, output.token_id, flag,
                          specAccepts, specRejects);
}

void BlazeRunner::checkOutputHang() {
  auto runningCount = slotManager.activeRunningCount();
  if (runningCount == 0) {
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
      elapsed.count(), runningCount, outputHangTimeout.count());
  std::abort();
}

void BlazeRunner::handleRequest(
    std::unique_ptr<tt::domain::llm::Sequence> request) {
  auto slotId = request->getKVCacheSlot();
  assert(slotId != tt::domain::INVALID_SLOT_ID);
  assert(slotId < tt::config::dsMaxUsers());

  bool isNew = !request->isContinuation() && !request->isDisaggregated();
  if (isNew && request->getSamplingParams().hasGuidedDecoding()) {
    TT_LOG_WARN(
        "[BlazeRunner] task_id={} has response_format constraint but "
        "SP Pipeline does not support per-step guided decoding yet. "
        "Output may not conform to the requested schema.",
        request->taskId);
  }

  auto& slotContext = slotManager.getSlotContext(slotId);
  switch (slotContext.state) {
    case types::SlotState::IDLE: {
      TT_LOG_DEBUG(
          "[BlazeRunner] handleRequest: taskId={}, slotId={}, isNew={}, "
          "isContinuation={}, numPromptTokens={}, totalTokens={}, "
          "runningSlots={}",
          request->taskId, slotId, isNew, request->isContinuation(),
          request->getNumPromptTokens(), request->getTokenIds().size(),
          slotManager.activeRunningCount());
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
      if (slotManager.activeRunningCount() == 0) {
        lastOutputTime = std::chrono::steady_clock::now();
      }
      slotContext.ignoreEos = request->getSamplingParams().ignore_eos;
      slotContext.specAcceptsAtStart =
          decodeScheduler->get_spec_accepts(slotId);
      slotContext.specRejectsAtStart =
          decodeScheduler->get_spec_rejects(slotId);
      slotContext.taskId = request->taskId;
      slotContext.tokensGenerated = 0;
      slotManager.bindTaskToSlot(request->taskId, slotId);
      slotManager.setSlotState(slotId, types::SlotState::RUNNING);
      tt::worker::SingleProcessWorkerMetrics::instance()
          .incrementActiveRequests();
      break;
    }
    case types::SlotState::AWAITING_STOP_ACK: {
      TT_LOG_DEBUG("[BlazeRunner] ");
      slotContext.deferredSubmit = std::move(request);
      break;
    }
    case types::SlotState::AWAITING_EVICT_ACK: {
      TT_LOG_DEBUG("[BlazeRunner] ");
      ipc::helpers::pushToken(*resultQueue, request->taskId, 0,
                              ipc::SharedToken::FLAG_ABORT, 0, 0);
      break;
    }
    default: {
      TT_LOG_ERROR(
          "[BlazeRunner] handleRequest: unexpected state for taskId={}, "
          "slotId={}",
          request->taskId, slotId);
      assert(false && "unexpected state for taskId");
      break;
    }
  }
}

}  // namespace tt::runners
