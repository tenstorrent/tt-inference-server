// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/tts/tts_types.hpp"
#include "ipc/tts_ipc.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/request_pipeline.hpp"
#include "services/tts_request_preprocessor.hpp"

namespace tt::services {

/** API-facing lifecycle owner for TTS generation requests. */
class TtsService : public IService {
 public:
  using StreamCallback = std::function<void(const domain::tts::TtsEvent&)>;

  TtsService(config::TtsConfig config,
             std::unique_ptr<tt::worker::WorkerManager> workerManager,
             std::unique_ptr<tt::ipc::tts::TtsQueueSet> queueManager);
  ~TtsService() override;

  void start() override;
  void stop() override;
  bool isModelReady() const override;
  SystemStatus getSystemStatus() const override;
  std::string runnerInUse() const override;
  uint32_t outputSampleRateHz() const;
  uint16_t outputChannels() const;

  bool generate(domain::tts::TtsRequest request, StreamCallback callback);
  void cancel(uint32_t taskId);

  /** Worker liveness source for WorkerMetricsAggregator (see LLMService). */
  tt::worker::WorkerManager* getWorkerManager() const {
    return workerManager.get();
  }

 private:
  size_t capacityLimit() const;
  size_t currentQueueSize() const;
  domain::tts::TtsTask prepareTask(const domain::tts::TtsRequest& request);

  void audioLoop(size_t workerIndex);
  bool deliverEvent(uint32_t taskId, const domain::tts::TtsEvent& event);
  void finishRequest(uint32_t taskId, domain::tts::TtsFinishReason reason);

  config::TtsConfig ttsConfig;
  std::unique_ptr<tt::worker::WorkerManager> workerManager;
  std::unique_ptr<tt::ipc::tts::TtsQueueSet> queueManager;
  TtsRequestPreprocessor requestPreprocessor;
  mutable std::mutex mutex;
  std::unordered_map<uint32_t, StreamCallback> callbacks;
  std::vector<std::thread> audioThreads;
  std::atomic<bool> running{false};
};

}  // namespace tt::services
