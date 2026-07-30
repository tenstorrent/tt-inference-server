// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/tts_service.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utils/logger.hpp"

namespace tt::services {

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
      ttsConfig.audioChannels,
      this->queueManager->audioQueues.size());
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

  {
    std::lock_guard<std::mutex> lock(mutex);
    if (callbacks.size() >= capacityLimit()) {
      throw QueueFullException{};
    }
    callbacks.emplace(task.task_id, std::move(callback));
  }

  if (!queueManager->taskQueue->tryPush(
          tt::ipc::tts::TtsIpcTask::fromDomainTask(task))) {
    std::lock_guard<std::mutex> lock(mutex);
    callbacks.erase(task.task_id);
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
  if (callback) {
    callback(reason);
  }
}

}  // namespace tt::services
