// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_runner.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <services/memory_services/blaze_memory_manager.hpp>

#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/helpers/token_push.hpp"
#include "runtime/runners/blaze_runner/blaze_types.hpp"
#include "runtime/runners/blaze_runner/blaze_utils.hpp"
#include "runtime/worker/single_process_worker_metrics.hpp"
#include "utils/logger.hpp"

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
        TT_LOG_WARN(
            "[BlazeRunner] handleEvictRequest: scheduler queue full, deferring "
            "EVICT for taskId={}, slotId={} (state={})",
            request.taskId, request.slotId, types::toString(slotContext.state));
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
      TT_LOG_DEBUG(
          "[BlazeRunner] handleEvictRequest: pushed EVICT taskId={}, slotId={} "
          "(was {})",
          request.taskId, request.slotId, wasRunning ? "RUNNING" : "IDLE");
      break;
    }
    case types::SlotState::AWAITING_STOP_ACK: {
      // Eviction supersedes any deferred submit; abort it so the client
      // doesn't hang waiting for tokens that will never come.
      if (slotContext.deferredSubmit) {
        auto droppedTaskId = slotContext.deferredSubmit->taskId;
        slotContext.deferredSubmit.reset();
        TT_LOG_DEBUG(
            "[BlazeRunner] handleEvictRequest: superseding deferredSubmit for "
            "taskId={} on slotId={} (DEALLOCATE wins)",
            droppedTaskId, request.slotId);
        ipc::helpers::pushToken(*resultQueue, droppedTaskId, 0,
                                ipc::SharedToken::FLAG_ABORT, 0, 0);
      }
      TT_LOG_DEBUG(
          "[BlazeRunner] handleEvictRequest: latching deferredEvict on "
          "slotId={} (waiting for STOP ack)",
          request.slotId);
      slotContext.deferredEvict = std::move(evictRequest);
      break;
    }
    case types::SlotState::AWAITING_EVICT_ACK:
      TT_LOG_WARN(
          "[BlazeRunner] handleEvictRequest: duplicate DEALLOCATE for slotId={}"
          " (already AWAITING_EVICT_ACK), ignoring",
          request.slotId);
      break;
    case types::SlotState::FREE:
      TT_LOG_ERROR(
          "[BlazeRunner] handleEvictRequest: DEALLOCATE for FREE slotId={}, "
          "taskId={}",
          request.slotId, request.taskId);
      assert(false && "DEALLOCATE on FREE slot");
      break;
  }
}

inline void BlazeRunner::handleAllocateRequest(
    const tt::domain::ManageMemoryTask& request) {
  auto allocateRequest = utils::makeAllocateRequest(request.taskId);
  if (!decodeScheduler->push_request(allocateRequest)) {
    TT_LOG_WARN(
        "[BlazeRunner] handleAllocateRequest: scheduler queue full, deferring "
        "ALLOCATE for taskId={}",
        request.taskId);
    pendingMemoryRetry = request;
    return;
  }
  TT_LOG_DEBUG("[BlazeRunner] handleAllocateRequest: pushed ALLOCATE taskId={}",
               request.taskId);
  pendingAllocates.insert(request.taskId);
}

inline void BlazeRunner::handleMemoryResponse(
    const ds::SchedulerResponse& response) {
  auto taskId = response.request_id;
  // ALLOCATE acks have no slot bound on our side yet; route by the
  // pendingAllocates set rather than by slot state.
  if (pendingAllocates.count(taskId) > 0) {
    pendingAllocates.erase(taskId);
    handleAllocateAck(taskId, response.slot_id);
    return;
  }
  auto slotId = response.slot_id;
  auto& slot = slotManager.getSlotContext(slotId);
  if (slot.pendingAckRequestId != taskId) {
    TT_LOG_ERROR(
        "[BlazeRunner] handleMemoryResponse: stale ack taskId={} for slotId={} "
        "(state={}, expected pendingAckRequestId={})",
        taskId, slotId, types::toString(slot.state),
        slot.pendingAckRequestId.has_value()
            ? std::to_string(*slot.pendingAckRequestId)
            : "none");
    return;
  }
  switch (slot.state) {
    case types::SlotState::AWAITING_EVICT_ACK:
      handleEvictAck(slot);
      break;
    case types::SlotState::AWAITING_STOP_ACK:
      handleStopAck(slot);
      break;
    default:
      TT_LOG_WARN(
          "[BlazeRunner] handleMemoryResponse: ack taskId={} for slotId={} in "
          "unexpected state={}",
          taskId, slotId, types::toString(slot.state));
      assert(false && "ack for slot in unexpected state");
      break;
  }
}

inline void BlazeRunner::handleAllocateAck(uint32_t taskId, uint32_t slotId) {
  if (slotId == ds::INVALID_SLOT) {
    TT_LOG_WARN("[BlazeRunner] handleAllocateAck: ALLOCATE failed taskId={}",
                taskId);
    memoryManager->replyAllocateFailure(taskId);
    return;
  }
  TT_LOG_DEBUG("[BlazeRunner] handleAllocateAck: taskId={}, slotId={}", taskId,
               slotId);
  slotManager.setSlotState(slotId, types::SlotState::IDLE);
  memoryManager->replyAllocateSuccess(taskId, slotId);
}

inline void BlazeRunner::handleEvictAck(types::SlotContext& slot) {
  TT_LOG_DEBUG(
      "[BlazeRunner] handleEvictAck: taskId={}, slotId={}; clearing slot",
      slot.pendingAckRequestId.value_or(0), slot.slotId);
  slotManager.clearSlotContext(slot.slotId);
}

inline void BlazeRunner::handleStopAck(types::SlotContext& slot) {
  TT_LOG_DEBUG(
      "[BlazeRunner] handleStopAck: taskId={}, slotId={}; draining "
      "(deferredEvict={}, deferredSubmit={})",
      slot.pendingAckRequestId.value_or(0), slot.slotId,
      slot.deferredEvict.has_value(), static_cast<bool>(slot.deferredSubmit));
  slotManager.setSlotAsIdle(slot.slotId);
  handleDeferred(slot);
}

inline void BlazeRunner::handleDeferred(types::SlotContext& slot) {
  // Called right after a STOP ack drains the slot back to IDLE. Two possible
  // followups were latched during AWAITING_STOP_ACK:
  //   - deferredEvict: EVICT wins (it's destructive); also abort any
  //     deferredSubmit since the slot is about to disappear.
  //   - deferredSubmit (and no deferredEvict): replay the SUBMIT now that
  //     the slot is IDLE again.
  if (slot.deferredEvict.has_value()) {
    if (slot.deferredSubmit) {
      // Capture taskId before reset (deferredSubmit is a unique_ptr;
      // reading through it after .reset() is a null deref).
      auto droppedTaskId = slot.deferredSubmit->taskId;
      slot.deferredSubmit.reset();
      TT_LOG_DEBUG(
          "[BlazeRunner] handleDeferred: dropping deferredSubmit taskId={} on "
          "slotId={} (deferredEvict wins)",
          droppedTaskId, slot.slotId);
      ipc::helpers::pushToken(*resultQueue, droppedTaskId, 0,
                              ipc::SharedToken::FLAG_ABORT, 0, 0);
    }
    auto evictReq = std::move(*slot.deferredEvict);
    slot.deferredEvict = std::nullopt;
    handleEvictRequest(tt::domain::ManageMemoryTask{
        .taskId = evictReq.request_id,
        .action = tt::domain::MemoryManagementAction::DEALLOCATE,
        .slotId = evictReq.slot_id,
    });
  } else if (slot.deferredSubmit) {
    // move clears slot.deferredSubmit
    handleRequest(std::move(slot.deferredSubmit));
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
    TT_LOG_DEBUG(
        "[BlazeRunner] handleCancelRequest: dropping queued retry for "
        "taskId={}",
        taskId);
    requestToRetry.reset();
    return;
  }
  auto slot = slotManager.getSlotContextByTaskId(taskId);
  if (!slot) {
    // Common race: cancel arrived after the generation finished and the slot
    // was already returned to IDLE (or never bound, e.g. cancel before
    // submit). Not actionable.
    TT_LOG_DEBUG(
        "[BlazeRunner] handleCancelRequest: taskId={} has no bound slot "
        "(already finished or never started); ignoring",
        taskId);
    return;
  }
  if (slot->state != types::SlotState::RUNNING) {
    // Slot is mid STOP/EVICT ack — STOP already in flight or slot is being
    // torn down. No new STOP needed.
    TT_LOG_DEBUG(
        "[BlazeRunner] handleCancelRequest: taskId={}, slotId={} not RUNNING "
        "(state={}); ignoring",
        taskId, slot->slotId, types::toString(slot->state));
    return;
  }
  if (!decodeScheduler->push_request(
          utils::makeStopRequest(taskId, slot->slotId))) {
    // Best effort: leave the slot RUNNING; next user cancel (if any) will
    // retry. Scheduler back-pressure is rare and self-clearing.
    TT_LOG_WARN(
        "[BlazeRunner] handleCancelRequest: scheduler queue full, failed to "
        "push STOP for taskId={}, slotId={}",
        taskId, slot->slotId);
    return;
  }
  TT_LOG_DEBUG(
      "[BlazeRunner] handleCancelRequest: pushed STOP taskId={}, slotId={}",
      taskId, slot->slotId);
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
    // Common race: scheduler produced a token that crossed paths with a STOP
    // or EVICT we just sent. Safe to drop. ERROR for truly unexpected states
    // (FREE/IDLE) since those indicate a bookkeeping bug.
    if (slotContext.state == types::SlotState::AWAITING_STOP_ACK ||
        slotContext.state == types::SlotState::AWAITING_EVICT_ACK) {
      TT_LOG_DEBUG(
          "[BlazeRunner] handleOutput: dropping token for slotId={} (state={}, "
          "token_id={}, is_complete={}) — STOP/EVICT in flight",
          output.slot_id, types::toString(slotContext.state), output.token_id,
          output.is_complete);
    } else {
      TT_LOG_ERROR(
          "[BlazeRunner] handleOutput: unexpected token for slotId={} in "
          "state={} (token_id={}, is_complete={})",
          output.slot_id, types::toString(slotContext.state), output.token_id,
          output.is_complete);
    }
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
      // STOP in flight from a prior request that reused this slot. Latch the
      // SUBMIT; it will be re-driven from handleMemoryResponse once the STOP
      // ack arrives (unless a deferred EVICT supersedes it there).
      TT_LOG_DEBUG(
          "[BlazeRunner] handleRequest: latching deferredSubmit for taskId={} "
          "on slotId={} (waiting for STOP ack)",
          request->taskId, slotId);
      slotContext.deferredSubmit = std::move(request);
      break;
    }
    case types::SlotState::AWAITING_EVICT_ACK: {
      // EVICT in flight — slot is going away. Reject the SUBMIT.
      TT_LOG_WARN(
          "[BlazeRunner] handleRequest: dropping SUBMIT for taskId={} on "
          "slotId={} (slot is AWAITING_EVICT_ACK)",
          request->taskId, slotId);
      ipc::helpers::pushToken(*resultQueue, request->taskId, 0,
                              ipc::SharedToken::FLAG_ABORT, 0, 0);
      break;
    }
    default: {
      // FREE or RUNNING: SessionManager shouldn't route a SUBMIT here.
      TT_LOG_ERROR(
          "[BlazeRunner] handleRequest: SUBMIT for taskId={} on slotId={} in "
          "unexpected state={}",
          request->taskId, slotId, types::toString(slotContext.state));
      assert(false && "SUBMIT for slot in unexpected state");
      break;
    }
  }
}

}  // namespace tt::runners
