// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/tts_service.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "metrics/metrics.hpp"
#include "utils/logger.hpp"

namespace tt::services {

namespace {

double secondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
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
    callbacksToCancel.reserve(inFlight.size());
    for (auto& [_, request] : inFlight) {
      callbacksToCancel.push_back(std::move(request.callback));
    }
    inFlight.clear();
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

  // The request clock starts before conditioning, so conditioning is measured
  // inside the request duration and their ratio is a true fraction. Which
  // main-process stage runs is decided by the same condition the preprocessor
  // branches on: a voice sample means PCM normalization, otherwise text
  // normalization and prompt compilation.
  const auto submittedAt = std::chrono::steady_clock::now();
  const auto conditioningStage =
      request.voiceSample.has_value()
          ? tt::metrics::TtsConditioningStage::VoiceNormalization
          : tt::metrics::TtsConditioningStage::TextNormalization;
  auto task = prepareTask(request);
  const double conditioningSeconds = secondsSince(submittedAt);

  TT_LOG_INFO(
      "[TtsService] Prepared TTS task task_id={} promptTokens={} "
      "voiceWavPcm={}",
      task.task_id, task.promptTokens.size(), task.voiceWavPcm.size());

  {
    std::lock_guard<std::mutex> lock(mutex);
    if (inFlight.size() >= capacityLimit()) {
      throw QueueFullException{};
    }
    inFlight.emplace(task.task_id,
                     InFlightRequest{std::move(callback), submittedAt,
                                     conditioningStage, conditioningSeconds});
  }

  if (!queueManager->taskQueue->tryPush(
          tt::ipc::tts::TtsIpcTask::fromDomainTask(task))) {
    std::lock_guard<std::mutex> lock(mutex);
    inFlight.erase(task.task_id);
    throw QueueFullException{};
  }
  return true;
}

void TtsService::cancel(uint32_t taskId) {
  StreamCallback callback;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = inFlight.find(taskId);
    if (it != inFlight.end()) {
      callback = std::move(it->second.callback);
      inFlight.erase(it);
    }
  }
  // Deliberately no timing observations here: a client abort ends the request
  // at an arbitrary point, so its duration would not describe engine time.
  // Dropping the entry also means the worker's later terminal message finds
  // nothing to observe, so a cancelled request contributes to neither the
  // conditioning numerator nor the duration denominator.

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
  return inFlight.size();
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
      finishRequest(message.task_id, message);
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
    auto it = inFlight.find(taskId);
    if (it == inFlight.end()) {
      return false;
    }
    callback = it->second.callback;
  }
  callback(event);
  return true;
}

void TtsService::finishRequest(
    uint32_t taskId, const tt::ipc::tts::TtsAudioChunkMessage& message) {
  InFlightRequest request;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = inFlight.find(taskId);
    if (it == inFlight.end()) {
      return;
    }
    request = std::move(it->second);
    inFlight.erase(it);
  }

  // Every conditioning stage and the request duration are observed here, at the
  // one point that sees a request through to the end, so all of them describe
  // the same set of requests. The worker's stages arrive as microseconds on the
  // terminal message; 0 means the stage did not run for this request (no voice
  // sample, or the voice-sample cache already held the speech IDs), and is
  // skipped rather than observed as a zero-length stage.
  auto& serverMetrics = tt::metrics::ServerMetrics::instance();
  serverMetrics.onTtsConditioning(request.conditioningStage,
                                  request.conditioningSeconds);
  if (message.voiceEncodeUs > 0) {
    serverMetrics.onTtsConditioning(
        tt::metrics::TtsConditioningStage::VoiceEncode,
        static_cast<double>(message.voiceEncodeUs) / 1e6);
  }
  if (message.promptCompileUs > 0) {
    serverMetrics.onTtsConditioning(
        tt::metrics::TtsConditioningStage::PromptCompile,
        static_cast<double>(message.promptCompileUs) / 1e6);
  }
  serverMetrics.onTtsRequestDuration(secondsSince(request.submittedAt));

  if (request.callback) {
    request.callback(message.finishReason());
  }
}

}  // namespace tt::services
