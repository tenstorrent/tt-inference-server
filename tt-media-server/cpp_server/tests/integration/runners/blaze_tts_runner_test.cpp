// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_tts_runner.hpp"

#include <gtest/gtest.h>
#include <stdlib.h>

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "config/settings.hpp"
#include "ipc/in_memory/in_memory_cancel_queue.hpp"
#include "runtime/worker/single_process_worker_metrics.hpp"
#include "runtime/worker/tts_metrics_layout.hpp"
#include "runtime/worker/worker_metrics_shm.hpp"

namespace tt::runners::blaze {
namespace {

namespace sched = tts_scheduler;

class RecordingTtsScheduler final : public sched::ITtsScheduler {
 public:
  void start() override {}
  void stop() override {}
  bool pushRequest(const sched::SchedulerRequest&) override { return true; }
  bool submit(const sched::TtsSubmit&) override { return true; }
  bool tryPopResponse(sched::SchedulerResponse&) override { return false; }
  bool tryPopToken(sched::TokenOutput&) override { return false; }
  bool tryPopAudio(sched::AudioOutput&) override { return false; }

  bool enqueueVoiceEncode(sched::VoiceEncodeRequest request) override {
    ++voiceEncodeCalls;
    pendingResult.requestId = request.requestId;
    pendingResult.speechIds = {12, 34, 56};
    pendingResult.status = sched::VoiceEncodeStatus::Completed;
    hasPendingResult = true;
    return true;
  }

  bool tryPopVoiceEncodeResult(sched::VoiceEncodeResult& result) override {
    if (!hasPendingResult) {
      return false;
    }
    result = std::move(pendingResult);
    hasPendingResult = false;
    return true;
  }

  std::atomic<uint32_t> voiceEncodeCalls = 0;

 private:
  sched::VoiceEncodeResult pendingResult;
  bool hasPendingResult = false;
};

/**
 * Scheduler that acks ALLOCATE and then emits a fixed number of codec tokens
 * followed by a terminal token and one final audio chunk — the shape the real
 * TTS engine produces, minus the audio.
 */
class TokenEmittingTtsScheduler final : public sched::ITtsScheduler {
 public:
  static constexpr uint32_t kCodecTokens = 5;
  /** Tokens the runner counts per request: the codec tokens plus the terminal
   *  one, which the engine stamps on a real token rather than a sentinel. */
  static constexpr uint32_t kCountedTokens = kCodecTokens + 1;

  void start() override {}
  void stop() override {}

  bool pushRequest(const sched::SchedulerRequest& request) override {
    sched::SchedulerResponse response;
    response.type = request.type;
    response.requestId = request.requestId;
    response.taskId = request.taskId;
    response.slotId =
        request.type == sched::RequestType::ALLOCATE ? 0u : request.slotId;
    responses.push_back(response);
    return true;
  }

  bool submit(const sched::TtsSubmit& request) override {
    sched::SchedulerResponse response;
    response.type = sched::RequestType::SUBMIT;
    response.requestId = request.requestId;
    response.taskId = request.taskId;
    response.slotId = request.slotId;
    responses.push_back(response);

    for (uint32_t i = 0; i < kCodecTokens; ++i) {
      tokens.push_back({.requestId = request.requestId,
                        .taskId = request.taskId,
                        .slotId = request.slotId,
                        .tokenId = i,
                        .final = false});
    }
    tokens.push_back({.requestId = request.requestId,
                      .taskId = request.taskId,
                      .slotId = request.slotId,
                      .tokenId = kCodecTokens,
                      .final = true});

    sched::AudioOutput audioOut;
    audioOut.requestId = request.requestId;
    audioOut.taskId = request.taskId;
    audioOut.slotId = request.slotId;
    audioOut.samplesBf16 = {0, 0, 0, 0};
    audioOut.last = true;
    audio.push_back(std::move(audioOut));
    return true;
  }

  bool tryPopResponse(sched::SchedulerResponse& response) override {
    return popFront(responses, response);
  }
  bool tryPopToken(sched::TokenOutput& output) override {
    return popFront(tokens, output);
  }
  bool tryPopAudio(sched::AudioOutput& output) override {
    return popFront(audio, output);
  }
  bool enqueueVoiceEncode(sched::VoiceEncodeRequest) override { return true; }
  bool tryPopVoiceEncodeResult(sched::VoiceEncodeResult&) override {
    return false;
  }

 private:
  template <typename T>
  static bool popFront(std::deque<T>& queue, T& out) {
    if (queue.empty()) return false;
    out = std::move(queue.front());
    queue.pop_front();
    return true;
  }

  std::deque<sched::SchedulerResponse> responses;
  std::deque<sched::TokenOutput> tokens;
  std::deque<sched::AudioOutput> audio;
};

std::string uniqueQueueName(const char* prefix) {
  static std::atomic<uint32_t> sequence = 0;
  return std::string(prefix) + "_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()) +
         "_" + std::to_string(++sequence);
}

bool waitForFinish(ipc::tts::TtsAudioChunkQueue& audioQueue, uint32_t taskId) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    ipc::tts::TtsAudioChunkMessage message;
    if (audioQueue.tryPop(message) && message.task_id == taskId &&
        message.isFinal()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

TEST(BlazeTtsRunnerIntegrationTest, ReusesSpeechIdsForMatchingVoiceSample) {
  config::TtsConfig config;
  config.maxUsers = 1;
  config.taskQueueCapacity = 1;
  config.audioQueueCapacity = 4;
  config.tokenizerPath.clear();

  ipc::tts::TtsTaskQueue taskQueue(uniqueQueueName("tts_task"), 4);
  ipc::tts::TtsAudioChunkQueue audioQueue(uniqueQueueName("tts_audio"), 4);
  ipc::in_memory::CancelQueue cancelQueue;
  auto scheduler = std::make_unique<RecordingTtsScheduler>();
  auto* schedulerPtr = scheduler.get();
  BlazeTtsRunner runner(config, std::move(scheduler), &taskQueue, &audioQueue,
                        &cancelQueue);
  std::thread runnerThread([&runner] { runner.start(); });

  const std::vector<int16_t> samples = {1, 2, 3, 4};
  taskQueue.push({.task_id = 1, .text = "first", .voiceWavPcm = samples});
  EXPECT_TRUE(waitForFinish(audioQueue, 1));
  EXPECT_EQ(schedulerPtr->voiceEncodeCalls.load(), 1u);

  taskQueue.push({.task_id = 2, .text = "second", .voiceWavPcm = samples});
  EXPECT_TRUE(waitForFinish(audioQueue, 2));
  EXPECT_EQ(schedulerPtr->voiceEncodeCalls.load(), 1u);

  taskQueue.shutdown();
  runnerThread.join();
  taskQueue.remove();
  audioQueue.remove();
}

TEST(BlazeTtsRunnerIntegrationTest, PublishesCodecTokensPerVoiceSource) {
  namespace tts_layout = tt::worker::tts;

  // Own the segment name so the test never collides with a server running on
  // the same host, and create it main-side before the writer attaches.
  const std::string shmName = uniqueQueueName("tts_metrics_shm");
  ASSERT_EQ(::setenv("TT_WORKER_METRICS_SHM", shmName.c_str(), 1), 0);
  auto shm = tt::worker::WorkerMetricsShm::create(
      tt::config::workerMetricsShmName(), 1);
  ASSERT_NE(shm, nullptr);
  tt::worker::SingleProcessWorkerMetrics::instance().initialize(
      0, tt::worker::MetricsLayout::TTS_RUNNER);

  config::TtsConfig config;
  config.maxUsers = 1;
  config.taskQueueCapacity = 1;
  config.audioQueueCapacity = 4;
  config.tokenizerPath.clear();

  ipc::tts::TtsTaskQueue taskQueue(uniqueQueueName("tts_task"), 4);
  ipc::tts::TtsAudioChunkQueue audioQueue(uniqueQueueName("tts_audio"), 8);
  ipc::in_memory::CancelQueue cancelQueue;
  BlazeTtsRunner runner(config, std::make_unique<TokenEmittingTtsScheduler>(),
                        &taskQueue, &audioQueue, &cancelQueue);
  std::thread runnerThread([&runner] { runner.start(); });

  // promptTokens are supplied directly so the runner skips prompt compilation
  // (which needs a tokenizer on disk) and goes straight to ALLOCATE.
  const std::vector<uint32_t> promptTokens = {1, 2, 3};
  taskQueue.push(
      {.task_id = 1, .text = "default voice", .promptTokens = promptTokens});
  EXPECT_TRUE(waitForFinish(audioQueue, 1));

  taskQueue.push({.task_id = 2,
                  .text = "described voice",
                  .description = "a calm narrator",
                  .promptTokens = promptTokens});
  EXPECT_TRUE(waitForFinish(audioQueue, 2));

  taskQueue.shutdown();
  runnerThread.join();

  const uint32_t expected = TokenEmittingTtsScheduler::kCountedTokens;
  EXPECT_EQ(shm->loadScratch(0, tts_layout::codecTokensIdx(
                                    tts_layout::VoiceSource::Default)),
            expected);
  EXPECT_EQ(shm->loadScratch(0, tts_layout::codecTokensIdx(
                                    tts_layout::VoiceSource::Description)),
            expected);
  EXPECT_EQ(shm->loadScratch(0, tts_layout::codecTokensIdx(
                                    tts_layout::VoiceSource::VoiceSample)),
            0u);
  EXPECT_NE(shm->loadScratch(0, tts_layout::SCRATCH_LAST_OUTPUT_EPOCH_MS), 0u);

  taskQueue.remove();
  audioQueue.remove();
}

}  // namespace
}  // namespace tt::runners::blaze
