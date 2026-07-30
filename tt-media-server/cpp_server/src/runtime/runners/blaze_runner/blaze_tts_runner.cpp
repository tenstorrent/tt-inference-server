// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_tts_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <utility>

#include "runtime/worker/single_process_worker_metrics.hpp"
#include "utils/logger.hpp"
#include "utils/tts_prompt_compiler.hpp"

namespace tt::runners::blaze {

namespace sched = tts_scheduler;

namespace {

sched::GenerationParams toSchedulerGeneration(
    const domain::tts::TtsGenerationParams& generation) {
  sched::GenerationParams out;
  out.ignoreEos = generation.ignoreEos;
  out.stopTokens = generation.stopTokenIds;
  return out;
}

}  // namespace

BlazeTtsRunner::BlazeTtsRunner(
    config::TtsConfig config, std::unique_ptr<sched::ITtsScheduler> ttsScheduler,
    ipc::tts::TtsTaskQueue* taskQueue,
    ipc::tts::TtsAudioChunkQueue* audioQueue, ipc::ICancelQueue* cancelQueue)
    : config(std::move(config)),
      scheduler(std::move(ttsScheduler)),
      taskQueue(taskQueue),
      audioQueue(audioQueue),
      cancelQueue(cancelQueue),
      outputHangTimeout(this->config.outputHangTimeoutMs) {
  if (!this->scheduler) {
    throw std::invalid_argument("BlazeTtsRunner: scheduler must not be null");
  }
  if (!this->taskQueue) {
    throw std::invalid_argument("BlazeTtsRunner: taskQueue must not be null");
  }
  if (!this->audioQueue) {
    throw std::invalid_argument("BlazeTtsRunner: audioQueue must not be null");
  }
  if (!this->cancelQueue) {
    throw std::invalid_argument("BlazeTtsRunner: cancelQueue must not be null");
  }

  slots.resize(std::max<size_t>(this->config.maxUsers, 1));
  for (uint32_t i = 0; i < slots.size(); ++i) {
    slots[i].slotId = i;
  }
  this->scheduler->start();
  lastOutputTime = std::chrono::steady_clock::now();
}

BlazeTtsRunner::~BlazeTtsRunner() {
  stop();
  shutdownScheduler();
}

bool BlazeTtsRunner::warmup() {
  TT_LOG_INFO("[BlazeTtsRunner] Warmup complete");
  return true;
}

void BlazeTtsRunner::stop() { stopped.store(true, std::memory_order_release); }

void BlazeTtsRunner::shutdownScheduler() {
  if (scheduler) {
    scheduler->stop();
  }
}

void BlazeTtsRunner::run() {
  TT_LOG_INFO("[BlazeTtsRunner] Entering TTS scheduler loop");
  while (!stopped.load(std::memory_order_acquire)) {
    step();
    tt::worker::SingleProcessWorkerMetrics::instance().updateStepHeartbeat();
  }
}

void BlazeTtsRunner::step() {
  drainVoiceEncodeResults();
  drainSchedulerResponses();
  drainTokenOutputs();
  drainAudioOutputs();
  drainControlMessages();
  drainDeferredStops();
  drainTasks();

  if (std::chrono::steady_clock::now() - lastOutputTime > outputHangTimeout) {
    TT_LOG_DEBUG("[BlazeTtsRunner] No TTS scheduler output for {} ms",
                 outputHangTimeout.count());
    lastOutputTime = std::chrono::steady_clock::now();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

void BlazeTtsRunner::drainSchedulerResponses() {
  sched::SchedulerResponse response;
  size_t drained = 0;
  while (drained < config.maxUsers && scheduler->tryPopResponse(response)) {
    handleSchedulerResponse(response);
    ++drained;
  }
}

void BlazeTtsRunner::drainTokenOutputs() {
  sched::TokenOutput output;
  size_t drained = 0;
  while (drained < config.maxUsers && scheduler->tryPopToken(output)) {
    handleTokenOutput(output);
    ++drained;
  }
}

void BlazeTtsRunner::drainAudioOutputs() {
  sched::AudioOutput output;
  size_t drained = 0;
  while (drained < config.audioQueueCapacity &&
         scheduler->tryPopAudio(output)) {
    handleAudioOutput(output);
    ++drained;
  }
  if (drained < config.audioQueueCapacity) {
    for (const auto& slot : slots) {
      maybeFinalizeCompletedSlot(slot.slotId);
    }
  }
}

void BlazeTtsRunner::drainVoiceEncodeResults() {
  sched::VoiceEncodeResult result;
  size_t drained = 0;
  while (drained < config.taskQueueCapacity &&
         scheduler->tryPopVoiceEncodeResult(result)) {
    handleVoiceEncodeResult(result);
    ++drained;
  }
}

void BlazeTtsRunner::drainControlMessages() {
  std::vector<uint32_t> taskIds;
  cancelQueue->tryPopAll(taskIds);
  for (uint32_t taskId : taskIds) {
    handleControl(taskId);
  }
}

void BlazeTtsRunner::drainDeferredStops() {
  if (deferredStopSlots.empty()) {
    return;
  }
  auto stops = std::move(deferredStopSlots);
  deferredStopSlots.clear();
  for (uint32_t slotId : stops) {
    requestStop(slotId);
  }
}

void BlazeTtsRunner::drainTasks() {
  ipc::tts::TtsIpcTask task;
  size_t drained = 0;
  while (drained < config.taskQueueCapacity && taskQueue->tryPop(task)) {
    if (task.isDone()) {
      stopped.store(true, std::memory_order_release);
      return;
    }
    handleTask(std::move(task));
    ++drained;
  }
}

void BlazeTtsRunner::handleTask(ipc::tts::TtsIpcTask task) {
  if (!task.voiceWavPcm.empty()) {
    sched::VoiceEncodeRequest request;
    request.requestId = task.task_id;
    request.wavPcm = std::move(task.voiceWavPcm);
    if (!scheduler->enqueueVoiceEncode(std::move(request))) {
      sendFinish(task.task_id, domain::tts::TtsFinishReason::Error,
                 "TTS scheduler voice encoder queue is full");
      return;
    }
    const uint32_t taskId = task.task_id;
    pendingVoiceEncodes.emplace(taskId, std::move(task));
    return;
  }

  allocateTask(std::move(task));
}

void BlazeTtsRunner::allocateTask(ipc::tts::TtsIpcTask task) {
  if (task.promptTokens.empty()) {
    sendFinish(task.task_id, domain::tts::TtsFinishReason::Error,
               "TTS compiled prompt is empty");
    return;
  }

  sched::SchedulerRequest request;
  request.type = sched::RequestType::ALLOCATE;
  request.requestId = task.task_id;
  request.taskId = task.task_id;
  if (!scheduler->pushRequest(request)) {
    TT_LOG_WARN("[BlazeTtsRunner] Scheduler queue full for ALLOCATE taskId={}",
                task.task_id);
    sendFinish(task.task_id, domain::tts::TtsFinishReason::Error,
               "TTS scheduler queue full during ALLOCATE");
    return;
  }
  pendingAllocations[task.task_id] = std::move(task);
}

void BlazeTtsRunner::handleControl(uint32_t taskId) {
  auto pendingVoice = pendingVoiceEncodes.find(taskId);
  if (pendingVoice != pendingVoiceEncodes.end()) {
    sendFinish(taskId, domain::tts::TtsFinishReason::Cancelled);
    pendingVoiceEncodes.erase(pendingVoice);
    return;
  }

  auto* slot = findSlotByTask(taskId);
  if (!slot) {
    auto pending = pendingAllocations.find(taskId);
    if (pending != pendingAllocations.end()) {
      sendFinish(pending->second.task_id,
                 domain::tts::TtsFinishReason::Cancelled);
      pendingAllocations.erase(pending);
    }
    return;
  }
  requestStop(slot->slotId);
}

void BlazeTtsRunner::handleVoiceEncodeResult(
    const sched::VoiceEncodeResult& result) {
  auto pending = pendingVoiceEncodes.find(result.requestId);
  if (pending == pendingVoiceEncodes.end()) {
    return;
  }

  auto task = std::move(pending->second);
  pendingVoiceEncodes.erase(pending);

  switch (result.status) {
    case sched::VoiceEncodeStatus::Completed:
      break;
    case sched::VoiceEncodeStatus::Cancelled:
      sendFinish(task.task_id, domain::tts::TtsFinishReason::Cancelled);
      return;
    case sched::VoiceEncodeStatus::Error:
      sendFinish(task.task_id, domain::tts::TtsFinishReason::Error,
                 "TTS voice sample encoding failed");
      return;
  }

  try {
    task.promptTokens = tt::utils::tts_prompt_compiler::compilePromptTokens(
        task.text, task.description, result.speechIds);
  } catch (const std::exception& e) {
    sendFinish(task.task_id, domain::tts::TtsFinishReason::Error, e.what());
    return;
  }

  allocateTask(std::move(task));
}

void BlazeTtsRunner::handleSchedulerResponse(
    const sched::SchedulerResponse& response) {
  lastOutputTime = std::chrono::steady_clock::now();
  switch (response.type) {
    case sched::RequestType::ALLOCATE:
      handleAllocateAck(response);
      break;
    case sched::RequestType::SUBMIT:
      handleSubmitAck(response);
      break;
    case sched::RequestType::CONTINUE:
      break;
    case sched::RequestType::STOP:
      handleStopAck(response);
      break;
    case sched::RequestType::EVICT:
      handleEvictAck(response);
      break;
  }
}

void BlazeTtsRunner::handleAllocateAck(
    const sched::SchedulerResponse& response) {
  auto pending = pendingAllocations.find(response.requestId);
  if (pending == pendingAllocations.end()) {
    if (response.slotId != sched::INVALID_SLOT) {
      sched::SchedulerRequest evict;
      evict.type = sched::RequestType::EVICT;
      evict.requestId = response.requestId;
      evict.taskId = response.taskId;
      evict.slotId = response.slotId;
      scheduler->pushRequest(evict);
    }
    return;
  }

  auto task = std::move(pending->second);
  pendingAllocations.erase(pending);

  if (response.errorCode != 0 || response.slotId == sched::INVALID_SLOT) {
    sendFinish(task.task_id, domain::tts::TtsFinishReason::Error,
               response.error);
    return;
  }

  auto* slot = findSlot(response.slotId);
  if (!slot) {
    sendFinish(task.task_id, domain::tts::TtsFinishReason::Error,
               "TTS scheduler returned slot outside configured max users");
    return;
  }

  slot->state = SlotState::RUNNING;
  slot->slotId = response.slotId;
  slot->task_id = task.task_id;
  slot->completionPending = false;

  sched::TtsSubmit submit;
  submit.requestId = task.task_id;
  submit.taskId = task.task_id;
  submit.slotId = response.slotId;
  submit.promptTokens = std::move(task.promptTokens);
  submit.generation = toSchedulerGeneration(task.generation);
  if (!scheduler->submit(submit)) {
    sendFinish(slot->task_id, domain::tts::TtsFinishReason::Error,
               "TTS scheduler queue full during SUBMIT");
    requestEvict(slot->slotId);
  }
}

void BlazeTtsRunner::handleSubmitAck(const sched::SchedulerResponse& response) {
  if (response.errorCode == 0) {
    return;
  }
  auto* slot = findSlot(response.slotId);
  if (slot && (slot->state == SlotState::AWAITING_STOP_ACK ||
               slot->state == SlotState::AWAITING_EVICT_ACK)) {
    TT_LOG_DEBUG(
        "[BlazeTtsRunner] Dropping SUBMIT error for slotId={} while teardown "
        "is in flight",
        response.slotId);
    return;
  }
  sendFinish(response.taskId, domain::tts::TtsFinishReason::Error,
             response.error);
  requestEvict(response.slotId);
}

void BlazeTtsRunner::handleStopAck(const sched::SchedulerResponse& response) {
  sendFinish(response.taskId, domain::tts::TtsFinishReason::Cancelled,
             response.error);
  requestEvict(response.slotId);
}

void BlazeTtsRunner::handleEvictAck(const sched::SchedulerResponse& response) {
  cleanupSlot(response.slotId);
}

void BlazeTtsRunner::handleTokenOutput(const sched::TokenOutput& output) {
  if (shouldDropOutput(output.slotId, "token")) {
    return;
  }
  TT_LOG_TRACE(
      "[BlazeTtsRunner] Drained speech token taskId={} slotId={} "
      "tokenId={} final={}",
      output.taskId, output.slotId, output.tokenId, output.final);
  if (output.final) {
    if (auto* slot = findSlot(output.slotId)) {
      slot->completionPending = scheduler->isComplete(output.slotId);
    }
  }
}

void BlazeTtsRunner::handleAudioOutput(const sched::AudioOutput& output) {
  lastOutputTime = std::chrono::steady_clock::now();
  if (shouldDropOutput(output.slotId, "audio")) {
    return;
  }
  if (!output.final) {
    ipc::tts::TtsAudioChunkMessage message;
    message.task_id = output.taskId;
    message.chunkIndex = output.chunkIndex;
    message.sampleRateHz = output.sampleRateHz;
    message.channels = output.channels;
    message.samplesBf16 = output.samplesBf16;
    if (!audioQueue->push(message)) {
      TT_LOG_ERROR("[BlazeTtsRunner] Audio queue full for taskId={}",
                   output.taskId);
      requestStop(output.slotId);
    }
    return;
  }

  sendFinish(output.taskId, output.finishReason, output.error);
  requestEvict(output.slotId);
}

void BlazeTtsRunner::sendFinish(uint32_t taskId,
                                domain::tts::TtsFinishReason reason,
                                std::string error) {
  audioQueue->push(
      ipc::tts::TtsAudioChunkMessage::finish(taskId, reason, std::move(error)));
}

void BlazeTtsRunner::maybeFinalizeCompletedSlot(uint32_t slotId) {
  auto* slot = findSlot(slotId);
  if (!slot || !slot->completionPending || slot->state != SlotState::RUNNING) {
    return;
  }
  if (!scheduler->isComplete(slotId) ||
      scheduler->getInFlightCount(slotId) != 0) {
    return;
  }

  slot->completionPending = false;
  sendFinish(slot->task_id, domain::tts::TtsFinishReason::Completed);
  requestEvict(slotId);
}

void BlazeTtsRunner::requestStop(uint32_t slotId) {
  auto* slot = findSlot(slotId);
  if (!slot || slot->state == SlotState::AWAITING_STOP_ACK ||
      slot->state == SlotState::AWAITING_EVICT_ACK) {
    return;
  }
  if (slot->state != SlotState::RUNNING) {
    TT_LOG_DEBUG(
        "[BlazeTtsRunner] Ignoring STOP for slotId={} because state is not "
        "RUNNING",
        slotId);
    return;
  }

  sched::SchedulerRequest request;
  request.type = sched::RequestType::STOP;
  request.requestId = slot->task_id;
  request.taskId = slot->task_id;
  request.slotId = slotId;
  if (scheduler->pushRequest(request)) {
    slot->state = SlotState::AWAITING_STOP_ACK;
    return;
  }
  if (std::find(deferredStopSlots.begin(), deferredStopSlots.end(), slotId) ==
      deferredStopSlots.end()) {
    deferredStopSlots.push_back(slotId);
  }
  TT_LOG_WARN(
      "[BlazeTtsRunner] Scheduler queue full; deferring STOP for "
      "slotId={}",
      slotId);
}

void BlazeTtsRunner::requestEvict(uint32_t slotId) {
  auto* slot = findSlot(slotId);
  if (!slot || slot->state == SlotState::AWAITING_EVICT_ACK) {
    return;
  }

  sched::SchedulerRequest request;
  request.type = sched::RequestType::EVICT;
  request.requestId = slot->task_id;
  request.taskId = slot->task_id;
  request.slotId = slotId;
  if (scheduler->pushRequest(request)) {
    slot->state = SlotState::AWAITING_EVICT_ACK;
  }
}

void BlazeTtsRunner::cleanupSlot(uint32_t slotId) {
  auto* slot = findSlot(slotId);
  if (!slot) {
    return;
  }
  *slot = SlotContext{};
  slot->slotId = slotId;
}

BlazeTtsRunner::SlotContext* BlazeTtsRunner::findSlot(uint32_t slotId) {
  if (slotId == sched::INVALID_SLOT || slotId >= slots.size()) {
    return nullptr;
  }
  return &slots[slotId];
}

BlazeTtsRunner::SlotContext* BlazeTtsRunner::findSlotByTask(uint32_t taskId) {
  for (auto& slot : slots) {
    if (slot.state != SlotState::FREE && slot.task_id == taskId) {
      return &slot;
    }
  }
  return nullptr;
}

bool BlazeTtsRunner::shouldDropOutput(uint32_t slotId,
                                      const char* outputType) const {
  if (slotId == sched::INVALID_SLOT || slotId >= slots.size()) {
    TT_LOG_ERROR("[BlazeTtsRunner] Unexpected {} output for invalid slotId={}",
                 outputType, slotId);
    return true;
  }
  const auto& slot = slots[slotId];
  if (slot.state == SlotState::AWAITING_STOP_ACK ||
      slot.state == SlotState::AWAITING_EVICT_ACK) {
    TT_LOG_DEBUG(
        "[BlazeTtsRunner] Dropping {} output for slotId={} during serialized "
        "STOP/EVICT teardown",
        outputType, slotId);
    return true;
  }
  if (slot.state != SlotState::RUNNING) {
    TT_LOG_ERROR(
        "[BlazeTtsRunner] Unexpected {} output for slotId={} in "
        "non-running state",
        outputType, slotId);
    return true;
  }
  return false;
}

}  // namespace tt::runners::blaze
