// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_tts_runner.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "ipc/in_memory/in_memory_cancel_queue.hpp"

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

}  // namespace
}  // namespace tt::runners::blaze
