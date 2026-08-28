// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
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
#include "metrics/tts_conditioning_stage.hpp"
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
  /** Terminal path for a task the worker finished. `message` carries the
   *  worker-side conditioning timings to observe alongside the request's own
   *  duration. */
  void finishRequest(uint32_t taskId,
                     const tt::ipc::tts::TtsAudioChunkMessage& message);

  /**
   * One in-flight request: its stream sink, the stamp that makes conditioning
   * readable as a share of the whole request, and the main-process conditioning
   * measurement held until the request terminates.
   *
   * The measurement is held rather than observed immediately so that
   * conditioning and request duration are observed together, over exactly the
   * same population of requests. Observing conditioning up front would count
   * requests whose duration never lands (client cancellations), which would
   * inflate conditioning's apparent share of engine time.
   */
  struct InFlightRequest {
    StreamCallback callback;
    std::chrono::steady_clock::time_point submittedAt;
    tt::metrics::TtsConditioningStage conditioningStage =
        tt::metrics::TtsConditioningStage::TextConditioning;
    double conditioningSeconds = 0.0;
  };

  config::TtsConfig ttsConfig;
  std::unique_ptr<tt::worker::WorkerManager> workerManager;
  std::unique_ptr<tt::ipc::tts::TtsQueueSet> queueManager;
  TtsRequestPreprocessor requestPreprocessor;
  mutable std::mutex mutex;
  std::unordered_map<uint32_t, InFlightRequest> inFlight;
  std::vector<std::thread> audioThreads;
  std::atomic<bool> running{false};
};

}  // namespace tt::services
