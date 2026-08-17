// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_tts_runner.hpp"

#include <gtest/gtest.h>
#include <stdlib.h>

#include <algorithm>
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

/**
 * Scheduler that withholds every stream's audio until both requests have been
 * submitted, then releases all of it at once.
 *
 * That is what makes the batch attribution testable: the runner buckets a
 * drainAudioOutputs() sweep by how many distinct slots it covered, so a
 * scheduler that released audio per-submit would produce a bucket that depends
 * on step-thread timing. Holding both back forces exactly one sweep over two
 * streams.
 */
class BatchedAudioTtsScheduler final : public sched::ITtsScheduler {
 public:
  static constexpr size_t kStreams = 2;
  static constexpr size_t kChunksPerStream = 2;
  static constexpr size_t kFramesPerChunk = 240;
  // Deliberately stereo: samplesBf16 is interleaved, so the runner has to
  // divide by the channel count to arrive at frames.
  static constexpr uint16_t kChannels = 2;

  static constexpr uint64_t kExpectedChunks = kStreams * kChunksPerStream;
  static constexpr uint64_t kExpectedFrames = kExpectedChunks * kFramesPerChunk;

  void start() override {}
  void stop() override {}

  bool pushRequest(const sched::SchedulerRequest& request) override {
    sched::SchedulerResponse response;
    response.type = request.type;
    response.requestId = request.requestId;
    response.taskId = request.taskId;
    response.slotId = request.type == sched::RequestType::ALLOCATE
                          ? nextSlot++
                          : request.slotId;
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

    stagedTokens.push_back({.requestId = request.requestId,
                            .taskId = request.taskId,
                            .slotId = request.slotId,
                            .tokenId = 0,
                            .final = true});

    for (size_t chunk = 0; chunk < kChunksPerStream; ++chunk) {
      sched::AudioOutput audioOut;
      audioOut.requestId = request.requestId;
      audioOut.taskId = request.taskId;
      audioOut.slotId = request.slotId;
      audioOut.chunkIndex = static_cast<uint32_t>(chunk);
      audioOut.channels = kChannels;
      audioOut.samplesBf16.assign(kFramesPerChunk * kChannels, 0);
      audioOut.last = chunk == kChunksPerStream - 1;
      stagedAudio.push_back(std::move(audioOut));
    }

    if (++submits == kStreams) {
      for (auto& token : stagedTokens) tokens.push_back(std::move(token));
      for (auto& chunk : stagedAudio) audio.push_back(std::move(chunk));
      stagedTokens.clear();
      stagedAudio.clear();
    }
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

  uint32_t nextSlot = 0;
  size_t submits = 0;
  std::deque<sched::SchedulerResponse> responses;
  std::deque<sched::TokenOutput> tokens;
  std::deque<sched::AudioOutput> audio;
  std::deque<sched::TokenOutput> stagedTokens;
  std::deque<sched::AudioOutput> stagedAudio;
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

TEST(BlazeTtsRunnerIntegrationTest, PublishesVocodedAudioPerBatchBucket) {
  namespace tts_layout = tt::worker::tts;
  using Scheduler = BatchedAudioTtsScheduler;

  const std::string shmName = uniqueQueueName("tts_metrics_shm");
  ASSERT_EQ(::setenv("TT_WORKER_METRICS_SHM", shmName.c_str(), 1), 0);
  auto shm = tt::worker::WorkerMetricsShm::create(
      tt::config::workerMetricsShmName(), 1);
  ASSERT_NE(shm, nullptr);
  tt::worker::SingleProcessWorkerMetrics::instance().initialize(
      0, tt::worker::MetricsLayout::TTS_RUNNER);

  config::TtsConfig config;
  config.maxUsers = Scheduler::kStreams;
  config.taskQueueCapacity = 4;
  config.audioQueueCapacity = 16;
  config.audioSampleRateHz = 24000;
  config.tokenizerPath.clear();

  ipc::tts::TtsTaskQueue taskQueue(uniqueQueueName("tts_task"), 4);
  ipc::tts::TtsAudioChunkQueue audioQueue(uniqueQueueName("tts_audio"), 16);
  ipc::in_memory::CancelQueue cancelQueue;
  BlazeTtsRunner runner(config, std::make_unique<Scheduler>(), &taskQueue,
                        &audioQueue, &cancelQueue);
  std::thread runnerThread([&runner] { runner.start(); });

  const std::vector<uint32_t> promptTokens = {1, 2, 3};
  taskQueue.push({.task_id = 1, .text = "first", .promptTokens = promptTokens});
  taskQueue.push(
      {.task_id = 2, .text = "second", .promptTokens = promptTokens});
  EXPECT_TRUE(waitForFinish(audioQueue, 1));
  EXPECT_TRUE(waitForFinish(audioQueue, 2));

  taskQueue.shutdown();
  runnerThread.join();

  // Both streams' chunks are released together, so the whole sweep is
  // attributed to the two-stream bucket and no other bucket is touched.
  EXPECT_EQ(shm->loadScratch(
                0, tts_layout::audioFramesIdx(tts_layout::BatchBucket::B2)),
            Scheduler::kExpectedFrames);
  EXPECT_EQ(shm->loadScratch(
                0, tts_layout::vocoderChunksIdx(tts_layout::BatchBucket::B2)),
            Scheduler::kExpectedChunks);
  EXPECT_EQ(shm->loadScratch(
                0, tts_layout::audioFramesIdx(tts_layout::BatchBucket::B1)),
            0u);
  EXPECT_EQ(shm->loadScratch(
                0, tts_layout::vocoderChunksIdx(tts_layout::BatchBucket::B1)),
            0u);

  // Sample rate is what converts the frame counter into audio seconds, and the
  // vocode clock is separate from the codec-token one.
  EXPECT_EQ(shm->loadScratch(0, tts_layout::SCRATCH_AUDIO_SAMPLE_RATE_HZ),
            config.audioSampleRateHz);
  EXPECT_NE(shm->loadScratch(0, tts_layout::SCRATCH_LAST_VOCODE_EPOCH_MS), 0u);

  taskQueue.remove();
  audioQueue.remove();
}

TEST(TtsMetricsLayoutTest, BucketsBatchSizesAndKeepsCellsDisjoint) {
  namespace tts_layout = tt::worker::tts;
  using tts_layout::BatchBucket;

  EXPECT_EQ(tts_layout::batchBucketOf(0), BatchBucket::B1);
  EXPECT_EQ(tts_layout::batchBucketOf(1), BatchBucket::B1);
  EXPECT_EQ(tts_layout::batchBucketOf(2), BatchBucket::B2);
  EXPECT_EQ(tts_layout::batchBucketOf(3), BatchBucket::B3_4);
  EXPECT_EQ(tts_layout::batchBucketOf(4), BatchBucket::B3_4);
  EXPECT_EQ(tts_layout::batchBucketOf(8), BatchBucket::B5_8);
  EXPECT_EQ(tts_layout::batchBucketOf(16), BatchBucket::B9_16);
  EXPECT_EQ(tts_layout::batchBucketOf(17), BatchBucket::B17Plus);
  // PM_MAX_USERS, the largest batch the runner could ever observe.
  EXPECT_EQ(tts_layout::batchBucketOf(128), BatchBucket::B17Plus);

  // Every cell the TTS layout writes must be distinct: a collision would make
  // two unrelated series alias each other in shm, which no test downstream of
  // the layout would catch.
  std::vector<size_t> indices{tts_layout::SCRATCH_STEP_EPOCH_MS,
                              tts_layout::SCRATCH_LAST_OUTPUT_EPOCH_MS,
                              tts_layout::SCRATCH_AUDIO_SAMPLE_RATE_HZ,
                              tts_layout::SCRATCH_LAST_VOCODE_EPOCH_MS};
  for (size_t i = 0; i < tts_layout::VOICE_SOURCE_COUNT; ++i) {
    indices.push_back(
        tts_layout::codecTokensIdx(static_cast<tts_layout::VoiceSource>(i)));
  }
  for (size_t i = 0; i < tts_layout::BATCH_BUCKET_COUNT; ++i) {
    const auto bucket = static_cast<BatchBucket>(i);
    indices.push_back(tts_layout::audioFramesIdx(bucket));
    indices.push_back(tts_layout::vocoderChunksIdx(bucket));
  }

  const size_t total = indices.size();
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  EXPECT_EQ(indices.size(), total) << "TTS scratch indices collide";
  EXPECT_LT(indices.back(), tts_layout::SCRATCH_RESERVED_END);
}

}  // namespace
}  // namespace tt::runners::blaze
