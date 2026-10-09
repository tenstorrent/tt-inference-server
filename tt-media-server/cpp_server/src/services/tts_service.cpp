// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/tts_service.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "utils/logger.hpp"
#include "utils/tts_traffic_log.hpp"

namespace tt::services {

namespace {

// Outcome label for a traffic record. Kept here rather than on the enum: it is
// a wire string that analysis scripts match on, so it should change only when
// someone means to change the capture format.
const char* finishReasonName(domain::tts::TtsFinishReason reason) {
  switch (reason) {
    case domain::tts::TtsFinishReason::Completed:
      return "completed";
    case domain::tts::TtsFinishReason::Cancelled:
      return "cancelled";
    case domain::tts::TtsFinishReason::Error:
      return "error";
  }
  return "unknown";
}

}  // namespace

TtsService::TtsService(config::TtsConfig config,
                       std::unique_ptr<tt::worker::WorkerManager> workerManager,
                       std::unique_ptr<tt::ipc::tts::TtsQueueSet> queueManager)
    : ttsConfig(std::move(config)),
      workerManager(std::move(workerManager)),
      queueManager(std::move(queueManager)),
      requestPreprocessor(ttsConfig) {
  if (!this->workerManager) {
    throw std::invalid_argument("TtsService: workerManager must not be null");
  }
  if (!this->queueManager || !this->queueManager->taskQueue) {
    throw std::invalid_argument("TtsService: queueManager must not be null");
  }
  TT_LOG_INFO(
      "[TtsService] Initialized worker-backed TTS service "
      "(runner={}, capacity={}, output_rate={}Hz, channels={}, workers={})",
      runnerInUse(), capacityLimit(), ttsConfig.audioSampleRateHz,
      ttsConfig.audioChannels, this->queueManager->audioQueues.size());
}

TtsService::~TtsService() { stop(); }

void TtsService::start() {
  if (running.exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  workerManager->start();
  audioThreads.reserve(queueManager->audioQueues.size());
  for (size_t workerIndex = 0; workerIndex < queueManager->audioQueues.size();
       ++workerIndex) {
    audioThreads.emplace_back(&TtsService::audioLoop, this, workerIndex);
  }
  TT_LOG_INFO("[TtsService] Started worker-backed service");
}

void TtsService::stop() {
  if (!running.exchange(false, std::memory_order_acq_rel)) {
    return;
  }

  if (queueManager && queueManager->taskQueue) {
    queueManager->taskQueue->shutdown();
  }
  if (queueManager) {
    for (auto& queue : queueManager->audioQueues) {
      queue->shutdown();
    }
  }

  workerManager->stop();
  for (auto& thread : audioThreads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  audioThreads.clear();

  std::vector<StreamCallback> callbacksToCancel;
  {
    std::lock_guard<std::mutex> lock(mutex);
    callbacksToCancel.reserve(callbacks.size());
    for (auto& [_, callback] : callbacks) {
      callbacksToCancel.push_back(std::move(callback));
    }
    callbacks.clear();
  }

  for (const auto& callback : callbacksToCancel) {
    if (callback) {
      callback(domain::tts::TtsFinishReason::Cancelled);
    }
  }
  TT_LOG_INFO("[TtsService] Stopped");
}

bool TtsService::isModelReady() const {
  return running.load(std::memory_order_acquire) && workerManager->isReady();
}

SystemStatus TtsService::getSystemStatus() const {
  SystemStatus status;
  status.modelReady = isModelReady();
  status.queueSize = currentQueueSize();
  status.maxQueueSize = capacityLimit();
  status.workerInfo = workerManager->getWorkerInfo();
  return status;
}

std::string TtsService::runnerInUse() const {
  switch (ttsConfig.runner_type) {
    case config::ModelRunnerType::TT_TTS:
      return "tt_tts";
    case config::ModelRunnerType::MOCK_SCHEDULER:
      return "mock_tts";
    default:
      return config::toClientRunnerName(ttsConfig.runner_type);
  }
}

uint32_t TtsService::outputSampleRateHz() const {
  return ttsConfig.audioSampleRateHz;
}

uint16_t TtsService::outputChannels() const { return ttsConfig.audioChannels; }

bool TtsService::generate(domain::tts::TtsRequest request,
                          StreamCallback callback) {
  if (!callback) {
    throw std::invalid_argument("TTS stream callback must not be null");
  }
  if (!isModelReady()) {
    return false;
  }

  auto task = prepareTask(request);
  TT_LOG_INFO(
      "[TtsService] Prepared TTS task task_id={} promptTokens={} "
      "voiceWavPcm={}",
      task.task_id, task.promptTokens.size(), task.voiceWavPcm.size());

  // Built before admission so a rejected request is still described by the
  // same fields as an accepted one; inflight is filled in under the lock.
  utils::tts_traffic::ArrivalInfo arrival;
  arrival.taskId = task.task_id;
  arrival.text = &request.text;
  arrival.description = request.description.has_value() ? &*request.description : nullptr;
  arrival.promptSpeechIds = request.promptSpeechIds.size();
  arrival.promptTokens = task.promptTokens.size();
  arrival.capacity = capacityLimit();

  {
    std::lock_guard<std::mutex> lock(mutex);
    arrival.inflightOnArrival = callbacks.size();
    if (callbacks.size() >= capacityLimit()) {
      utils::tts_traffic::onRejected(arrival, "rejected_capacity");
      throw QueueFullException{};
    }
    callbacks.emplace(task.task_id, std::move(callback));
  }
  utils::tts_traffic::onArrival(arrival);

  if (!queueManager->taskQueue->tryPush(
          tt::ipc::tts::TtsIpcTask::fromDomainTask(task))) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      callbacks.erase(task.task_id);
    }
    utils::tts_traffic::onFinish(task.task_id, "rejected_task_queue");
    throw QueueFullException{};
  }
  return true;
}

void TtsService::cancel(uint32_t taskId) {
  StreamCallback callback;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = callbacks.find(taskId);
    if (it != callbacks.end()) {
      callback = std::move(it->second);
      callbacks.erase(it);
    }
  }

  TT_LOG_DEBUG("[TtsService] Cancel requested for TTS task {}", taskId);
  if (queueManager) {
    for (auto& queue : queueManager->cancelQueues) {
      queue->push(taskId);
    }
  }
  if (callback) {
    // cancel() answers the client itself rather than going through
    // finishRequest, so the record has to be closed here or it leaks.
    utils::tts_traffic::onFinish(taskId, "cancelled");
    callback(domain::tts::TtsFinishReason::Cancelled);
  }
}

size_t TtsService::capacityLimit() const {
  const size_t taskCapacity = std::max<size_t>(ttsConfig.taskQueueCapacity, 1);
  const size_t userCapacity = std::max<size_t>(ttsConfig.maxUsers, 1);
  return std::min(taskCapacity, userCapacity);
}

size_t TtsService::currentQueueSize() const {
  std::lock_guard<std::mutex> lock(mutex);
  return callbacks.size();
}

domain::tts::TtsTask TtsService::prepareTask(
    const domain::tts::TtsRequest& request) {
  return requestPreprocessor.process(request);
}

void TtsService::audioLoop(size_t workerIndex) {
  TT_LOG_INFO("[TtsService] Audio drain thread started for worker {}",
              workerIndex);
  auto& queue = queueManager->audioQueues.at(workerIndex);
  tt::ipc::tts::TtsAudioChunkMessage message;
  while (running.load(std::memory_order_acquire) &&
         queue->blockingPop(message)) {
    if (message.isFinal()) {
      finishRequest(message.task_id, message.finishReason());
      continue;
    }
    deliverEvent(message.task_id, message.toDomainChunk());
  }
  TT_LOG_INFO("[TtsService] Audio drain thread stopped for worker {}",
              workerIndex);
}

bool TtsService::deliverEvent(uint32_t taskId,
                              const domain::tts::TtsEvent& event) {
  StreamCallback callback;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = callbacks.find(taskId);
    if (it == callbacks.end()) {
      return false;
    }
    callback = it->second;
  }
  if (const auto* chunk = std::get_if<domain::tts::TtsAudioChunk>(&event)) {
    utils::tts_traffic::onChunk(taskId, chunk->chunkIndex,
                                chunk->samplesBf16.size(), chunk->sampleRateHz);
  }
  callback(event);
  return true;
}

void TtsService::finishRequest(uint32_t taskId,
                               domain::tts::TtsFinishReason reason) {
  StreamCallback callback;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = callbacks.find(taskId);
    if (it == callbacks.end()) {
      return;
    }
    callback = std::move(it->second);
    callbacks.erase(it);
  }
  utils::tts_traffic::onFinish(taskId, finishReasonName(reason));
  if (callback) {
    callback(reason);
  }
}

}  // namespace tt::services
