// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_prefill_runner.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>
#include <utility>

#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/helpers/token_push.hpp"
#include "runtime/runners/blaze_runner/blaze_slot_manager.hpp"
#include "runtime/runners/blaze_runner/blaze_utils.hpp"
#include "runtime/worker/single_process_worker_metrics.hpp"
#include "services/memory_services/memory_manager.hpp"
#include "utils/logger.hpp"

namespace tt::runners::blaze {
BlazePrefillRunner::BlazePrefillRunner(
    const config::LLMConfig& config, ipc::IResultQueue* resultQueue,
    tt::ipc::ITaskQueue* taskQueue, tt::ipc::ICancelQueue* stopQueue,
    std::unique_ptr<tt::services::MemoryManager> injectedMemoryManager)
    : config(config),
      resultQueue(resultQueue),
      taskQueue(taskQueue),
      stopQueue(stopQueue),
      slotManager(tt::config::psMaxUsers()),
      lastOutputTime(std::chrono::steady_clock::now()),
      outputHangTimeout(tt::config::outputHangTimeoutMs()),
      simPrefillMsPer5kTokens(tt::config::simulatePrefillMsPer5kTokens()) {
  TT_LOG_INFO(
      "BlazePrefillRunner: Constructing PrefillScheduler with SocketConfig...");
  auto pipelineConfig = utils::makePrefillPipelineConfig(config);
  ps::SchedulerParams managerParams{
      .layers_per_chunk = static_cast<uint32_t>(std::stoi(tt::config::prefillNumLayers())),
      .chunk_size = 5120,
  };
  managerParams.max_users = static_cast<uint32_t>(tt::config::psMaxUsers());
  auto ackChannelConfig = utils::makePrefillAckChannelConfig(config);
  prefillScheduler = std::make_unique<ps::PrefillScheduler>(
      pipelineConfig, ackChannelConfig, managerParams);
  TT_LOG_INFO(
      "BlazePrefillRunner: PrefillScheduler constructed, calling start()...");
  prefillScheduler->start();
  TT_LOG_INFO(
      "BlazePrefillRunner: PipelineManager started, creating MemoryManager...");
  memoryManager = injectedMemoryManager
                      ? std::move(injectedMemoryManager)
                      : std::make_unique<tt::services::MemoryManager>();
  TT_LOG_INFO("BlazePrefillRunner: Constructor complete");
}

BlazePrefillRunner::~BlazePrefillRunner() {
  stop();
  if (prefillScheduler) {
    prefillScheduler->stop();
  }
}

void BlazePrefillRunner::run() {
  while (!stopped.load(std::memory_order_relaxed)) {
    try {
      step();
      tt::worker::SingleProcessWorkerMetrics::instance().updateStepHeartbeat();
    } catch (const std::exception& e) {
      TT_LOG_ERROR("BlazePrefillRunner: Exception in run: {}", e.what());
      throw;
    }
  }
}

bool BlazePrefillRunner::warmup() {
  tt::domain::llm::SamplingParams warmupParams;
  warmupParams.max_tokens = 1;
  warmupParams.ignore_eos = true;

  std::vector<int64_t> warmupTokens(5120, 12345);
  uint32_t warmupTaskId = 0;

  auto warmupSeq = std::make_unique<tt::domain::llm::Sequence>(
      warmupTaskId, 1, warmupTokens, warmupParams);

  constexpr uint32_t warmupAllocateRequestId = 0;
  constexpr uint32_t warmupEvictRequestId = 1;

  const auto timeout = std::chrono::milliseconds(tt::config::warmupTimeoutMs());
  const auto pollInterval = std::chrono::milliseconds(10);

  TT_LOG_INFO("BlazePrefillRunner: warmup - pushing ALLOCATE request...");
  prefillScheduler->push_request(
      utils::makeAllocateRequest(warmupAllocateRequestId));

  TT_LOG_INFO("BlazePrefillRunner: warmup - waiting for ALLOCATE response...");
  ps::SchedulerResponse response{};
  const auto allocateDeadline = std::chrono::steady_clock::now() + timeout;
  while (!prefillScheduler->try_pop_response(response)) {
    if (std::chrono::steady_clock::now() >= allocateDeadline) {
      TT_LOG_ERROR(
          "[BlazePrefillRunner] Warmup timed out waiting for ALLOCATE response "
          "after "
          "{} ms",
          timeout.count());
      return false;
    }
    std::this_thread::sleep_for(pollInterval);
  }

  auto slotId = response.slot_id;
  TT_LOG_INFO("BlazePrefillRunner: warmup - got slot_id={}", slotId);
  if (slotId == ps::INVALID_SLOT) {
    TT_LOG_ERROR("BlazePrefillRunner: Warmup failed with error");
    return false;
  }

  TT_LOG_INFO("BlazePrefillRunner: warmup - pushing SUBMIT request...");
  prefillScheduler->push_request(utils::makeSubmitRequest(slotId, *warmupSeq));

  const auto deadline = std::chrono::steady_clock::now() + timeout;
  bool receivedToken = false;
  auto output = ps::OutputMessage{};

  while (std::chrono::steady_clock::now() < deadline) {
    if (prefillScheduler->try_pop_output(output)) {
      if (output.prefill_complete) {
        receivedToken = true;
        break;
      }
    }
    std::this_thread::sleep_for(pollInterval);
  }

  if (!receivedToken) {
    TT_LOG_ERROR(
        "[BlazePrefillRunner] Warmup timed out waiting for token after {} ms",
        timeout.count());
    return false;
  }

  TT_LOG_INFO(
      "BlazePrefillRunner: warmup - pushing EVICT request (slotId={})...",
      slotId);
  prefillScheduler->push_request(
      utils::makeEvictRequest(warmupEvictRequestId, slotId));
  ps::SchedulerResponse evictResponse{};
  const auto evictDeadline = std::chrono::steady_clock::now() + timeout;
  while (!prefillScheduler->try_pop_response(evictResponse)) {
    if (std::chrono::steady_clock::now() >= evictDeadline) {
      TT_LOG_ERROR(
          "[BlazePrefillRunner] Warmup timed out waiting for EVICT ack after "
          "{} ms "
          "(slotId={})",
          timeout.count(), slotId);
      return false;
    }
    std::this_thread::sleep_for(pollInterval);
  }
  TT_LOG_INFO("BlazePrefillRunner: warmup - got EVICT ack (slotId={})", slotId);
  TT_LOG_INFO("BlazePrefillRunner: Warmup successful");
  return true;
}

void BlazePrefillRunner::stop() {
  stopped.store(true, std::memory_order_relaxed);
}

void BlazePrefillRunner::step() {
  drainAndHandleMemoryResponses();
  drainAndHandleOutputs();
  auto memoryRequest = getMemoryRequest();
  if (memoryRequest.has_value()) {
    TT_LOG_DEBUG(
        "[BlazePrefillRunner] step: got memoryRequest taskId={}, action={}",
        memoryRequest->taskId, static_cast<int>(memoryRequest->action));
    handleMemoryRequest(*memoryRequest);
  }
  drainAndHandleStopRequests();
  auto request = getRequest();
  if (request) {
    TT_LOG_DEBUG(
        "[BlazePrefillRunner] step: got Sequence taskId={}, slotId={}, "
        "numPromptTokens={}, totalTokens={}",
        request->taskId, request->getPrefillKVCacheSlot(),
        request->getNumPromptTokens(), request->getTokenIds().size());
    handleRequest(std::move(request));
  }
  checkOutputHang();
}

void BlazePrefillRunner::drainAndHandleMemoryResponses() {
  ps::SchedulerResponse response;
  size_t drained = 0;
  size_t maxUsers = tt::config::psMaxUsers();
  while (drained < maxUsers && prefillScheduler->try_pop_response(response)) {
    handleMemoryResponse(response);
    drained++;
  }
}

void BlazePrefillRunner::drainAndHandleOutputs() {
  ps::OutputMessage output;
  size_t drained = 0;
  size_t maxUsers = tt::config::psMaxUsers();
  while (drained < maxUsers && prefillScheduler->try_pop_output(output)) {
    handleOutput(output);
    drained++;
  }
}

std::unique_ptr<tt::domain::llm::Sequence> BlazePrefillRunner::getRequest() {
  if (pendingRequests.pendingTask) {
    return std::move(pendingRequests.pendingTask);
  }
  auto req = taskQueue->tryPop();
  if (!req) return nullptr;
  return req;
}

inline void BlazePrefillRunner::handleMemoryRequest(
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
          "[BlazePrefillRunner] handleMemoryRequest: unexpected action for "
          "taskId={}, "
          "action={}",
          request.taskId, static_cast<int>(request.action));
      assert(false && "unexpected action for taskId");
      break;
    }
  }
}

inline void BlazePrefillRunner::handleEvictRequest(
    const tt::domain::ManageMemoryTask& request) {
  auto& slotContext = slotManager.getSlotContext(request.slotId);
  auto evictRequest = utils::makeEvictRequest(request.taskId, request.slotId);
  switch (slotContext.state) {
    case SlotState::IDLE:
    case SlotState::RUNNING: {
      if (!prefillScheduler->push_request(evictRequest)) {
        TT_LOG_WARN(
            "[BlazePrefillRunner] handleEvictRequest: scheduler queue full, "
            "deferring "
            "EVICT for taskId={}, slotId={} (state={})",
            request.taskId, request.slotId, toString(slotContext.state));
        pendingRequests.pendingMemoryTask = request;
        return;
      }
      if (pendingRequests.pendingTask &&
          pendingRequests.pendingTask->getPrefillKVCacheSlot() ==
              request.slotId) {
        auto droppedTaskId = pendingRequests.pendingTask->taskId;
        pendingRequests.pendingTask.reset();
        TT_LOG_DEBUG(
            "[BlazePrefillRunner] handleEvictRequest: dropping pending task "
            "for "
            "taskId={} on slotId={} (EVICT wins)",
            droppedTaskId, request.slotId);
        ipc::helpers::pushToken(
            *resultQueue, droppedTaskId, 0,
            ipc::SharedToken::FLAG_FINAL | ipc::SharedToken::FLAG_ABORT, 0, 0);
      }
      // Capture before setSlotState changes the state.
      bool wasRunning = (slotContext.state == SlotState::RUNNING);
      if (wasRunning) {
        slotManager.unbindTaskFromSlot(*slotContext.taskId);
        tt::worker::SingleProcessWorkerMetrics::instance()
            .decrementActiveRequests();
      }
      slotContext.pendingAckRequestId = request.taskId;
      slotManager.setSlotState(request.slotId, SlotState::AWAITING_EVICT_ACK);
      TT_LOG_DEBUG(
          "[BlazePrefillRunner] handleEvictRequest: pushed EVICT taskId={}, "
          "slotId={} "
          "(was {})",
          request.taskId, request.slotId, wasRunning ? "RUNNING" : "IDLE");
      break;
    }
    case SlotState::AWAITING_STOP_ACK: {
      // Eviction supersedes any deferred submit; terminate it so the client
      // doesn't hang waiting for tokens that will never come.
      // FINAL|ERROR (not ABORT): ABORT is for client-initiated cancels where
      // the controller has already set done=true and the stream is being
      // swallowed. Here the client is innocent — they're still waiting on the
      // SSE — so the runner has to close the stream itself with an error.
      if (slotContext.deferredContinue) {
        auto droppedTaskId = slotContext.deferredContinue->taskId;
        slotContext.deferredContinue.reset();
        TT_LOG_DEBUG(
            "[BlazePrefillRunner] handleEvictRequest: superseding "
            "deferredContinue "
            "for "
            "taskId={} on slotId={} (DEALLOCATE wins)",
            droppedTaskId, request.slotId);
        ipc::helpers::pushToken(
            *resultQueue, droppedTaskId, 0,
            ipc::SharedToken::FLAG_FINAL | ipc::SharedToken::FLAG_ABORT, 0, 0);
      }
      TT_LOG_DEBUG(
          "[BlazePrefillRunner] handleEvictRequest: latching deferredEvict on "
          "slotId={} (waiting for STOP ack)",
          request.slotId);
      slotContext.deferredEvict = std::move(evictRequest);
      break;
    }
    case SlotState::AWAITING_EVICT_ACK:
      TT_LOG_WARN(
          "[BlazePrefillRunner] handleEvictRequest: duplicate DEALLOCATE for "
          "slotId={}"
          " (already AWAITING_EVICT_ACK), ignoring",
          request.slotId);
      assert(false && "Double DEALLOCATE on same slot");
      break;
    case SlotState::FREE:
      TT_LOG_ERROR(
          "[BlazePrefillRunner] handleEvictRequest: DEALLOCATE for FREE "
          "slotId={}, "
          "taskId={}",
          request.slotId, request.taskId);
      assert(false && "DEALLOCATE on FREE slot");
      break;
  }
}

inline void BlazePrefillRunner::handleAllocateRequest(
    const tt::domain::ManageMemoryTask& request) {
  auto allocateRequest = utils::makeAllocateRequest(request.taskId);
  if (!prefillScheduler->push_request(allocateRequest)) {
    TT_LOG_WARN(
        "[BlazePrefillRunner] handleAllocateRequest: scheduler queue full, "
        "deferring "
        "ALLOCATE for taskId={}",
        request.taskId);
    pendingRequests.pendingMemoryTask = request;
    return;
  }
  TT_LOG_DEBUG(
      "[BlazePrefillRunner] handleAllocateRequest: pushed ALLOCATE taskId={}",
      request.taskId);
}

inline void BlazePrefillRunner::handleMemoryResponse(
    const ps::SchedulerResponse& response) {
  auto taskId = response.request_id;
  auto slotId = response.slot_id;
  auto action = response.request_type;

  switch (action) {
    case ps::RequestType::ALLOCATE: {
      handleAllocateAck(taskId, slotId);
      break;
    }
    case ps::RequestType::EVICT: {
      handleEvictAck(taskId, slotId);
      break;
    }
    case ps::RequestType::STOP: {
      handleStopAck(taskId, slotId);
      break;
    }
    default: {
      TT_LOG_ERROR(
          "[BlazePrefillRunner] handleMemoryResponse: unexpected action for "
          "taskId={}, "
          "action={}",
          taskId, static_cast<int>(action));
      assert(false && "unexpected action for taskId");
      break;
    }
  }
}

inline void BlazePrefillRunner::handleAllocateAck(uint32_t taskId,
                                                  uint32_t slotId) {
  // == Do we gain anything if we add a new state AWAITING_EVICT_ACK? ==
  if (slotId == ps::INVALID_SLOT) {
    TT_LOG_WARN(
        "[BlazePrefillRunner] handleAllocateAck: ALLOCATE failed taskId={}",
        taskId);
    memoryManager->replyAllocateFailure(taskId);
    return;
  }
  TT_LOG_DEBUG("[BlazePrefillRunner] handleAllocateAck: taskId={}, slotId={}",
               taskId, slotId);
  slotManager.setSlotState(slotId, SlotState::IDLE);
  memoryManager->replyAllocateSuccess(taskId, slotId);
}

inline SlotContext* BlazePrefillRunner::validateAck(uint32_t taskId,
                                                    uint32_t slotId,
                                                    const char* ackName) {
  auto& slot = slotManager.getSlotContext(slotId);
  if (slot.pendingAckRequestId != taskId) {
    // optional<uint32_t> compares as expected: nullopt != scalar is true,
    // so this catches both the no-pending-ack and mismatched-id cases.
    TT_LOG_ERROR(
        "[BlazePrefillRunner] {}: stale ack taskId={} for slotId={} (state={}, "
        "expected pendingAckRequestId={})",
        ackName, taskId, slotId, toString(slot.state),
        slot.pendingAckRequestId.has_value()
            ? std::to_string(*slot.pendingAckRequestId)
            : "none");
    return nullptr;
  }
  return &slot;
}

inline void BlazePrefillRunner::handleEvictAck(uint32_t taskId,
                                               uint32_t slotId) {
  auto* slot = validateAck(taskId, slotId, "handleEvictAck");
  if (!slot) return;
  TT_LOG_DEBUG(
      "[BlazePrefillRunner] handleEvictAck: taskId={}, slotId={}; clearing "
      "slot",
      taskId, slotId);
  slotManager.clearSlotContext(slot->slotId);
}

inline void BlazePrefillRunner::handleStopAck(uint32_t taskId,
                                              uint32_t slotId) {
  auto* slot = validateAck(taskId, slotId, "handleStopAck");
  if (!slot) return;
  TT_LOG_DEBUG(
      "[BlazePrefillRunner] handleStopAck: taskId={}, slotId={}; draining "
      "(deferredEvict={}, deferredSubmit={})",
      taskId, slotId, slot->deferredEvict.has_value(),
      static_cast<bool>(slot->deferredContinue));
  slotManager.setSlotAsIdle(slot->slotId);
  handleDeferred(*slot);
}

inline void BlazePrefillRunner::handleDeferred(SlotContext& slot) {
  // Called right after a STOP ack drains the slot back to IDLE. Two possible
  // followups were latched during AWAITING_STOP_ACK:
  //   - deferredEvict: EVICT wins (it's destructive); also abort any
  //     deferredSubmit since the slot is about to disappear.
  //   - deferredSubmit (and no deferredEvict): replay the SUBMIT now that
  //     the slot is IDLE again.
  if (slot.deferredEvict.has_value()) {
    if (slot.deferredContinue) {
      // Capture taskId before reset (deferredContinue is a unique_ptr;
      // reading through it after .reset() is a null deref).
      auto droppedTaskId = slot.deferredContinue->taskId;
      slot.deferredContinue.reset();
      TT_LOG_DEBUG(
          "[BlazePrefillRunner] handleDeferred: dropping deferredContinue "
          "taskId={} "
          "on "
          "slotId={} (deferredEvict wins)",
          droppedTaskId, slot.slotId);
      // FINAL|ERROR (not ABORT) — see handleEvictRequest's comment.
      ipc::helpers::pushToken(
          *resultQueue, droppedTaskId, 0,
          ipc::SharedToken::FLAG_FINAL | ipc::SharedToken::FLAG_ABORT, 0, 0);
    }
    auto evictReq = std::move(*slot.deferredEvict);
    slot.deferredEvict = std::nullopt;
    handleEvictRequest(tt::domain::ManageMemoryTask{
        .taskId = evictReq.request_id,
        .action = tt::domain::MemoryManagementAction::DEALLOCATE,
        .slotId = evictReq.slot_id,
    });
  } else if (slot.deferredContinue) {
    // move clears slot.deferredContinue
    handleRequest(std::move(slot.deferredContinue));
  }
}

inline std::optional<tt::domain::ManageMemoryTask>
BlazePrefillRunner::getMemoryRequest() {
  if (pendingRequests.pendingMemoryTask.has_value()) {
    auto task = std::move(*pendingRequests.pendingMemoryTask);
    pendingRequests.pendingMemoryTask = std::nullopt;
    return task;
  }
  return memoryManager->getRequest();
}

void BlazePrefillRunner::drainAndHandleStopRequests() {
  if (pendingRequests.pendingCancelTaskId.has_value()) {
    auto retryTaskId = *pendingRequests.pendingCancelTaskId;
    pendingRequests.pendingCancelTaskId = std::nullopt;
    TT_LOG_DEBUG(
        "[BlazePrefillRunner] drainAndHandleCancelRequests: retrying deferred "
        "cancel "
        "for taskId={}",
        retryTaskId);
    handleStopRequest(retryTaskId);
  }
  std::vector<uint32_t> taskIds;
  stopQueue->tryPopAll(taskIds);
  for (auto taskId : taskIds) {
    handleStopRequest(taskId);
  }
}

inline void BlazePrefillRunner::handleStopRequest(uint32_t taskId) {
  if (pendingRequests.pendingTask &&
      pendingRequests.pendingTask->taskId == taskId) {
    TT_LOG_DEBUG(
        "[BlazePrefillRunner] handleStopRequest: dropping queued retry for "
        "taskId={}",
        taskId);
    pendingRequests.pendingTask.reset();
    return;
  }
  auto slot = slotManager.getSlotContextByTaskId(taskId);
  if (!slot) {
    // Common race: cancel arrived after the generation finished and the slot
    // was already returned to IDLE (or never bound, e.g. cancel before
    // submit). Not actionable.
    TT_LOG_DEBUG(
        "[BlazePrefillRunner] handleCancelRequest: taskId={} has no bound slot "
        "(already finished or never started); ignoring",
        taskId);
    return;
  }
  if (slot->state != SlotState::RUNNING) {
    // Slot is mid STOP/EVICT ack — STOP already in flight or slot is being
    // torn down. No new STOP needed.
    TT_LOG_DEBUG(
        "[BlazePrefillRunner] handleCancelRequest: taskId={}, slotId={} not "
        "RUNNING "
        "(state={}); ignoring",
        taskId, slot->slotId, toString(slot->state));
    return;
  }
  if (!prefillScheduler->push_request(
          utils::makeStopRequest(taskId, slot->slotId))) {
    TT_LOG_WARN(
        "[BlazePrefillRunner] handleCancelRequest: scheduler queue full, "
        "deferring "
        "STOP for taskId={}, slotId={}",
        taskId, slot->slotId);
    pendingRequests.pendingCancelTaskId = taskId;
    return;
  }
  TT_LOG_DEBUG(
      "[BlazePrefillRunner] handleCancelRequest: pushed STOP taskId={}, "
      "slotId={}",
      taskId, slot->slotId);
  slot->pendingAckRequestId = taskId;
  slotManager.setSlotState(slot->slotId, SlotState::AWAITING_STOP_ACK);
  slotManager.unbindTaskFromSlot(taskId);
  ipc::helpers::pushToken(*resultQueue, taskId, 0, ipc::SharedToken::FLAG_ABORT,
                          0, 0);
  tt::worker::SingleProcessWorkerMetrics::instance().decrementActiveRequests();
}

void BlazePrefillRunner::handleOutput(const ps::OutputMessage& output) {
  tt::worker::SingleProcessWorkerMetrics::instance().updateOutputHeartbeat();
  lastOutputTime = std::chrono::steady_clock::now();
  auto& slotContext = slotManager.getSlotContext(output.slot_id);
  if (slotContext.state != SlotState::RUNNING) {
    if (slotContext.state == SlotState::AWAITING_STOP_ACK ||
        slotContext.state == SlotState::AWAITING_EVICT_ACK) {
      // Legitimate race: scheduler had this token in flight when our STOP /
      // EVICT got there. Drop and move on.
      TT_LOG_DEBUG(
          "[BlazePrefillRunner] handleOutput: dropping token for slotId={} "
          "(state={}, "
          "token_id={}, prefill_complete={}, ctx_exhausted={}) — STOP/EVICT in "
          "flight",
          output.slot_id, toString(slotContext.state), output.token_id,
          output.prefill_complete, output.ctx_exhausted);
      return;
    }
    // FREE/IDLE: scheduler is emitting tokens for a slot we don't believe is
    // active. That's a bookkeeping bug, not a race — fail loud.
    TT_LOG_ERROR(
        "[BlazePrefillRunner] handleOutput: unexpected token for slotId={} in "
        "state={} (token_id={}, prefill_complete={}, ctx_exhausted={})",
        output.slot_id, toString(slotContext.state), output.token_id,
        output.prefill_complete, output.ctx_exhausted);
    assert(false && "scheduler output for slot not RUNNING/AWAITING_*_ACK");
    return;
  }
  auto taskId = slotContext.taskId.value();
  if (output.ctx_exhausted) {
    slotManager.setSlotAsIdle(output.slot_id);
    tt::worker::SingleProcessWorkerMetrics::instance()
        .decrementActiveRequests();
    ipc::helpers::pushToken(
        *resultQueue, taskId, output.token_id,
        ipc::SharedToken::FLAG_ERROR | ipc::SharedToken::FLAG_FINAL, 0, 0);
    return;
  }
  bool finished = output.prefill_complete;

  slotContext.currentPosition = output.real_pos;
  if (finished) {
    slotManager.setSlotAsIdle(output.slot_id);
    tt::worker::SingleProcessWorkerMetrics::instance()
        .decrementActiveRequests();
  }
  uint32_t flag = finished ? ipc::SharedToken::FLAG_FINAL : 0;
  ipc::helpers::pushToken(*resultQueue, taskId, output.token_id, flag, 0, 0);
}

void BlazePrefillRunner::checkOutputHang() {
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
      "[BlazePrefillRunner] Output hang detected: no model output for {} ms "
      "with {} "
      "in-flight generation(s) (threshold={} ms). Self-terminating worker so "
      "infrastructure can restart the server.",
      elapsed.count(), runningCount, outputHangTimeout.count());
  std::abort();
}

void BlazePrefillRunner::handleRequest(
    std::unique_ptr<tt::domain::llm::Sequence> request) {
  auto slotId = request->getPrefillKVCacheSlot();
  assert(slotId != tt::domain::INVALID_SLOT_ID);
  assert(slotId < tt::config::psMaxUsers());

  auto& slotContext = slotManager.getSlotContext(slotId);
  switch (slotContext.state) {
    case SlotState::IDLE: {
      TT_LOG_DEBUG(
          "[BlazePrefillRunner] handleRequest: taskId={}, slotId={}, "
          "isContinuation={}, numPromptTokens={}, totalTokens={}, "
          "runningSlots={}",
          request->taskId, slotId, request->isContinuation(),
          request->getNumPromptTokens(), request->getTokenIds().size(),
          slotManager.activeRunningCount());
      // Debug: simulate real on-device prefill cost. Blocks the worker's step
      // loop (prefill is serialized), so the IPC task queue — and thus
      // tt_num_requests_in_flight — builds up under load. Off when 0.
      if (simPrefillMsPer5kTokens > 0) {
        const size_t prefillTokens = request->getNumPromptTokens();
        const std::chrono::duration<double, std::milli> delay{
            static_cast<double>(prefillTokens) * simPrefillMsPer5kTokens /
            5000.0};
        TT_LOG_DEBUG(
            "[BlazePrefillRunner] handleRequest: simulating prefill delay "
            "taskId={}, tokens={}, delayMs={}",
            request->taskId, prefillTokens, delay.count());
        std::this_thread::sleep_for(delay);
      }
      ps::ISRequest req = utils::makeSubmitRequest(
          slotId, *request, std::make_optional(request->getKVCacheSlot()));
      if (!prefillScheduler->push_request(req)) {
        TT_LOG_DEBUG(
            "[BlazePrefillRunner] handleRequest: failed to push request, "
            "taskId={}, "
            "slotId={}",
            request->taskId, slotId);
        pendingRequests.pendingTask = std::move(request);
        return;
      }
      if (slotManager.activeRunningCount() == 0) {
        lastOutputTime = std::chrono::steady_clock::now();
      }
      utils::initSlotForRun(slotContext, *request);
      slotManager.bindTaskToSlot(request->taskId, slotId);
      slotManager.setSlotState(slotId, SlotState::RUNNING);
      tt::worker::SingleProcessWorkerMetrics::instance()
          .incrementActiveRequests();
      break;
    }

    case SlotState::AWAITING_STOP_ACK: {
      if (slotContext.deferredContinue) {
        TT_LOG_WARN(
            "[BlazePrefillRunner] handleRequest: overwriting deferred "
            "taskId={} with "
            "taskId={} on slotId={} — the dropped task's stream will not "
            "finalize",
            slotContext.deferredContinue->taskId, request->taskId, slotId);
      }
      TT_LOG_DEBUG(
          "[BlazePrefillRunner] handleRequest: latching deferredSubmit for "
          "taskId={} "
          "on slotId={} (waiting for STOP ack)",
          request->taskId, slotId);
      slotContext.deferredContinue = std::move(request);
      break;
    }
    case SlotState::AWAITING_EVICT_ACK: {
      TT_LOG_WARN(
          "[BlazePrefillRunner] handleRequest: dropping SUBMIT for taskId={} "
          "on "
          "slotId={} (slot is AWAITING_EVICT_ACK)",
          request->taskId, slotId);
      ipc::helpers::pushToken(
          *resultQueue, request->taskId, 0,
          ipc::SharedToken::FLAG_FINAL | ipc::SharedToken::FLAG_ABORT, 0, 0);
      break;
    }
    default: {
      // FREE or RUNNING: SessionManager shouldn't route a SUBMIT here.
      TT_LOG_ERROR(
          "[BlazePrefillRunner] handleRequest: SUBMIT for taskId={} on "
          "slotId={} in "
          "unexpected state={}",
          request->taskId, slotId, toString(slotContext.state));
      assert(false && "SUBMIT for slot in unexpected state");
      break;
    }
  }
}

}  // namespace tt::runners::blaze
