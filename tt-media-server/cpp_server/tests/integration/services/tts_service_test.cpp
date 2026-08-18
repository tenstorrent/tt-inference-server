// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// Main-process half of the TTS conditioning metrics. TtsService::finishRequest
// is the single point that decides which conditioning stages a finished request
// contributes and observes the request-duration denominator, so these tests
// drive a real TtsService — real preprocessor, real IPC queues, a stand-in
// worker subprocess — and assert on the exposition text ServerMetrics renders,
// which is what the dashboard reads.
//
// Every request here carries a voice sample. That is the preprocessor branch
// which normalizes PCM rather than compiling a prompt, and so the only one
// reachable without a TTS tokenizer on disk.

#include "services/tts_service.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "../../support/test_worker_main.hpp"
#include "config/runner_config.hpp"
#include "domain/tts/tts_types.hpp"
#include "ipc/tts_ipc.hpp"
#include "metrics/metrics.hpp"
#include "runtime/worker/worker_manager.hpp"

namespace tt::services {
namespace {

namespace tts_domain = tt::domain::tts;

constexpr size_t K_WORKER_COUNT = 1;

// Worker-side conditioning timings planted on the terminal message. Exact and
// distinct so the microsecond-to-second conversion is pinned by value, not just
// asserted to have happened.
constexpr uint32_t K_VOICE_ENCODE_US = 250000;
constexpr double K_VOICE_ENCODE_SECONDS = 0.25;
constexpr uint32_t K_PROMPT_COMPILE_US = 1500;
constexpr double K_PROMPT_COMPILE_SECONDS = 0.0015;

// Exposition series the TTS dashboard queries. Spelled out rather than derived
// from the production label helpers so renaming a stage fails here.
constexpr const char* K_CONDITIONING_COUNT =
    "tt_tts_conditioning_seconds_count";
constexpr const char* K_CONDITIONING_SUM = "tt_tts_conditioning_seconds_sum";
constexpr const char* K_REQUEST_COUNT = "tt_tts_request_duration_seconds_count";
constexpr const char* K_REQUEST_SUM = "tt_tts_request_duration_seconds_sum";

/**
 * Env the TTS stack reads once and caches. MODEL_SERVICE decides which IPC
 * queues WorkerManager opens for its worker; the two per-process names keep
 * this binary from reaching into the warmup queue or metrics segment of a
 * server running on the same host. The forked worker inherits all of it.
 */
void configureEnvForTest() {
  const std::string suffix = std::to_string(::getpid());
  ::setenv("MODEL_SERVICE", "tts", 1);
  ::setenv("DEVICE_IDS", "(0)", 1);
  ::setenv("TT_WARMUP_SIGNALS_QUEUE",
           ("tts_service_test_warmup_" + suffix).c_str(), 1);
  ::setenv("TT_WORKER_METRICS_SHM",
           ("tts_service_test_metrics_" + suffix).c_str(), 1);
}

/**
 * Value of the `series` sample whose label set contains `labelMatch`, read out
 * of Prometheus exposition text. A series with no matching sample reads as 0,
 * which gives a stage that has never been observed a baseline to delta from.
 */
double metricValue(const std::string& exposition, const std::string& series,
                   const std::string& labelMatch) {
  const std::string prefix = series + "{";
  std::istringstream lines(exposition);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.rfind(prefix, 0) != 0) continue;
    if (line.find(labelMatch) == std::string::npos) continue;
    const size_t valuePos = line.rfind(' ');
    if (valuePos == std::string::npos) continue;
    return std::stod(line.substr(valuePos + 1));
  }
  return 0.0;
}

/**
 * The counters these tests assert on, sampled together. Assertions compare a
 * before/after pair rather than absolute values: ServerMetrics is a
 * process-wide singleton, so a test cannot assume it starts from zero.
 */
struct MetricsSnapshot {
  double textConditioning = 0.0;
  double voiceNormalization = 0.0;
  double voiceEncode = 0.0;
  double promptCompile = 0.0;
  double voiceEncodeSeconds = 0.0;
  double promptCompileSeconds = 0.0;
  double requests = 0.0;
  double requestSeconds = 0.0;

  static MetricsSnapshot read() {
    const std::string text =
        tt::metrics::ServerMetrics::instance().renderText();
    MetricsSnapshot snapshot;
    snapshot.textConditioning =
        metricValue(text, K_CONDITIONING_COUNT, "stage=\"text_conditioning\"");
    snapshot.voiceNormalization = metricValue(text, K_CONDITIONING_COUNT,
                                              "stage=\"voice_normalization\"");
    snapshot.voiceEncode =
        metricValue(text, K_CONDITIONING_COUNT, "stage=\"voice_encode\"");
    snapshot.promptCompile =
        metricValue(text, K_CONDITIONING_COUNT, "stage=\"prompt_compile\"");
    snapshot.voiceEncodeSeconds =
        metricValue(text, K_CONDITIONING_SUM, "stage=\"voice_encode\"");
    snapshot.promptCompileSeconds =
        metricValue(text, K_CONDITIONING_SUM, "stage=\"prompt_compile\"");
    snapshot.requests = metricValue(text, K_REQUEST_COUNT, "model_name=");
    snapshot.requestSeconds = metricValue(text, K_REQUEST_SUM, "model_name=");
    return snapshot;
  }
};

/**
 * Terminal-event sink for one request. finishRequest runs on the audio-drain
 * thread, so tests wait on this instead of sleeping; a satisfied signal also
 * serves as a drain barrier for everything queued ahead of it.
 *
 * Held by shared_ptr because the service owns the callback for as long as the
 * request is in flight, which can outlive the test body that submitted it.
 */
class FinishSignal {
 public:
  void onEvent(const tts_domain::TtsEvent& event) {
    const auto* reason = std::get_if<tts_domain::TtsFinishReason>(&event);
    if (reason == nullptr) return;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (finishReason.has_value()) return;
      finishReason = *reason;
    }
    signalled.notify_all();
  }

  std::optional<tts_domain::TtsFinishReason> wait() {
    std::unique_lock<std::mutex> lock(mutex);
    signalled.wait_for(lock, K_TIMEOUT,
                       [this] { return finishReason.has_value(); });
    return finishReason;
  }

 private:
  static constexpr std::chrono::seconds K_TIMEOUT{5};

  std::mutex mutex;
  std::condition_variable signalled;
  std::optional<tts_domain::TtsFinishReason> finishReason;
};

tts_domain::TtsRequest voiceSampleRequest(uint32_t taskId) {
  tts_domain::TtsRequest request(taskId);
  request.text = "the quick brown fox";
  tts_domain::VoiceSample sample;
  // Long enough, and at a rate that forces a resample, that the main-process
  // conditioning this measures is real work rather than a no-op copy.
  sample.wavPcm.assign(4000, 128);
  sample.sampleRateHz = 16000;
  sample.channels = 1;
  request.voiceSample = std::move(sample);
  return request;
}

class TtsServiceMetricsTest : public ::testing::Test {
 protected:
  static constexpr std::chrono::seconds K_READY_TIMEOUT{10};

  void SetUp() override {
    config::TtsConfig config;
    config.maxUsers = 8;
    config.taskQueueCapacity = 8;
    config.audioQueueCapacity = 8;
    config.cancelQueueCapacity = 8;
    config.tokenizerPath.clear();

    auto queues = std::make_unique<tt::ipc::tts::TtsQueueSet>(
        static_cast<int>(K_WORKER_COUNT), config);
    // The test plays the worker on this queue, pushing the terminal messages a
    // real BlazeTtsRunner would send with its conditioning timings attached.
    audioQueue = queues->audioQueues.front().get();
    service = std::make_unique<TtsService>(
        config, std::make_unique<tt::worker::WorkerManager>(K_WORKER_COUNT),
        std::move(queues));
    service->start();

    // generate() is gated on isModelReady(), which flips once the stand-in
    // worker subprocess (see main below) has signalled warmup.
    const auto deadline = std::chrono::steady_clock::now() + K_READY_TIMEOUT;
    while (!service->isModelReady() &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(service->isModelReady())
        << "stand-in worker subprocess never signalled warmup";
  }

  void TearDown() override { service.reset(); }

  std::shared_ptr<FinishSignal> submitVoiceRequest(uint32_t taskId) {
    auto signal = std::make_shared<FinishSignal>();
    const bool accepted =
        service->generate(voiceSampleRequest(taskId),
                          [signal](const tts_domain::TtsEvent& event) {
                            signal->onEvent(event);
                          });
    EXPECT_TRUE(accepted);
    return signal;
  }

  void pushTerminalMessage(uint32_t taskId, uint32_t voiceEncodeUs,
                           uint32_t promptCompileUs) {
    auto message = tt::ipc::tts::TtsAudioChunkMessage::finish(
        taskId, tts_domain::TtsFinishReason::Completed);
    message.voiceEncodeUs = voiceEncodeUs;
    message.promptCompileUs = promptCompileUs;
    ASSERT_TRUE(audioQueue->push(message));
  }

  /**
   * Runs one fresh request through to its terminal event. A single audio-drain
   * thread pops the queue in order, so a completed barrier proves every message
   * pushed before it has already been handled — which is how a test observes
   * that a message was deliberately ignored.
   */
  void runBarrierRequest(uint32_t taskId) {
    auto signal = submitVoiceRequest(taskId);
    pushTerminalMessage(taskId, /*voiceEncodeUs=*/0, /*promptCompileUs=*/0);
    ASSERT_EQ(signal->wait(), tts_domain::TtsFinishReason::Completed);
  }

  tt::ipc::tts::TtsAudioChunkQueue* audioQueue = nullptr;
  std::unique_ptr<TtsService> service;
};

TEST_F(TtsServiceMetricsTest, ObservesMainAndWorkerConditioningStages) {
  const auto before = MetricsSnapshot::read();

  auto signal = submitVoiceRequest(1);
  pushTerminalMessage(1, K_VOICE_ENCODE_US, K_PROMPT_COMPILE_US);
  ASSERT_EQ(signal->wait(), tts_domain::TtsFinishReason::Completed);

  const auto after = MetricsSnapshot::read();

  // A voice-sample request normalizes PCM in the main process, so it belongs to
  // voice_normalization and must leave the text-only stage untouched.
  EXPECT_EQ(after.voiceNormalization - before.voiceNormalization, 1.0);
  EXPECT_EQ(after.textConditioning - before.textConditioning, 0.0);

  // Both worker stages ran, and their microsecond fields arrive as seconds.
  EXPECT_EQ(after.voiceEncode - before.voiceEncode, 1.0);
  EXPECT_EQ(after.promptCompile - before.promptCompile, 1.0);
  EXPECT_NEAR(after.voiceEncodeSeconds - before.voiceEncodeSeconds,
              K_VOICE_ENCODE_SECONDS, 1e-9);
  EXPECT_NEAR(after.promptCompileSeconds - before.promptCompileSeconds,
              K_PROMPT_COMPILE_SECONDS, 1e-9);

  // The denominator that makes conditioning readable as a share of engine time.
  EXPECT_EQ(after.requests - before.requests, 1.0);
  EXPECT_GT(after.requestSeconds - before.requestSeconds, 0.0);
}

TEST_F(TtsServiceMetricsTest, SkipsWorkerStagesThatDidNotRun) {
  const auto before = MetricsSnapshot::read();

  auto signal = submitVoiceRequest(1);
  // Zero microseconds is how the worker reports a stage it never ran — here a
  // voice-sample cache hit, which skips the encode entirely.
  pushTerminalMessage(1, /*voiceEncodeUs=*/0, /*promptCompileUs=*/0);
  ASSERT_EQ(signal->wait(), tts_domain::TtsFinishReason::Completed);

  const auto after = MetricsSnapshot::read();

  // Not observed at all, rather than observed as zero: a zero-length sample
  // would read as a real instantaneous encode and drag the stage's quantiles
  // toward zero.
  EXPECT_EQ(after.voiceEncode - before.voiceEncode, 0.0);
  EXPECT_EQ(after.promptCompile - before.promptCompile, 0.0);
  EXPECT_EQ(after.voiceEncodeSeconds - before.voiceEncodeSeconds, 0.0);

  // The main-process stage and the denominator still land, so the request is
  // present in the share even though two of its stages are silent.
  EXPECT_EQ(after.voiceNormalization - before.voiceNormalization, 1.0);
  EXPECT_EQ(after.requests - before.requests, 1.0);
}

TEST_F(TtsServiceMetricsTest, IgnoresTerminalMessageForCancelledRequest) {
  const auto before = MetricsSnapshot::read();

  auto cancelled = submitVoiceRequest(1);
  service->cancel(1);
  ASSERT_EQ(cancelled->wait(), tts_domain::TtsFinishReason::Cancelled);

  // The worker does not learn about a cancel it has already raced past, so its
  // terminal message still arrives, timings and all. A client abort ends the
  // request at an arbitrary point, so none of it describes engine time.
  pushTerminalMessage(1, K_VOICE_ENCODE_US, K_PROMPT_COMPILE_US);
  runBarrierRequest(2);

  const auto after = MetricsSnapshot::read();

  // Exactly the barrier request: the cancelled one contributes to neither the
  // conditioning numerator nor the duration denominator, so it cannot skew the
  // share in either direction.
  EXPECT_EQ(after.requests - before.requests, 1.0);
  EXPECT_EQ(after.voiceNormalization - before.voiceNormalization, 1.0);
  EXPECT_EQ(after.voiceEncode - before.voiceEncode, 0.0);
  EXPECT_EQ(after.promptCompile - before.promptCompile, 0.0);
}

TEST_F(TtsServiceMetricsTest, ObservesAFinishedRequestOnlyOnce) {
  const auto before = MetricsSnapshot::read();

  auto signal = submitVoiceRequest(1);
  pushTerminalMessage(1, K_VOICE_ENCODE_US, K_PROMPT_COMPILE_US);
  ASSERT_EQ(signal->wait(), tts_domain::TtsFinishReason::Completed);

  // A second terminal message for the same task — a worker retry, or a replay
  // out of the audio queue — finds no in-flight entry and must not observe the
  // request again; double counting here would inflate every conditioning share.
  pushTerminalMessage(1, K_VOICE_ENCODE_US, K_PROMPT_COMPILE_US);
  runBarrierRequest(2);

  const auto after = MetricsSnapshot::read();

  // One observation for the finished request, one for the barrier.
  EXPECT_EQ(after.requests - before.requests, 2.0);
  EXPECT_EQ(after.voiceNormalization - before.voiceNormalization, 2.0);
  // The barrier ran neither worker stage, so these see only the first request.
  EXPECT_EQ(after.voiceEncode - before.voiceEncode, 1.0);
  EXPECT_NEAR(after.voiceEncodeSeconds - before.voiceEncodeSeconds,
              K_VOICE_ENCODE_SECONDS, 1e-9);
}

}  // namespace
}  // namespace tt::services

int main(int argc, char** argv) {
  // WorkerManager spawns its worker by re-execing this binary. The stand-in
  // only has to signal warmup and idle; the test process plays the worker on
  // the TTS queues itself.
  if (argc >= 3 && std::string(argv[1]) == "--worker") {
    return tt::test::runWorkerSubprocess(std::atoi(argv[2]));
  }
  tt::services::configureEnvForTest();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
