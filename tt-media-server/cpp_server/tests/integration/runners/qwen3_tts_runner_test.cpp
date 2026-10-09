// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Qwen3TtsRunner against a fake tt-metal: the runner's embedded interpreter
// imports a stand-in `ttnn`, `torch` and `models.demos.audio.qwen3_tts`
// written to a temp directory, so the C++ side of the runner (task intake,
// cancels, PCM16 chunking, the terminal message, metrics) runs without a
// device. The counterpart of blaze_tts_runner_test.cpp, whose schedulers play
// the same role for the Blaze runner.

#include "runtime/runners/qwen3_tts/qwen3_tts_runner.hpp"

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <stdlib.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "config/defaults.hpp"
#include "config/settings.hpp"
#include "ipc/in_memory/in_memory_cancel_queue.hpp"
#include "runtime/worker/single_process_worker_metrics.hpp"
#include "runtime/worker/tts_metrics_layout.hpp"
#include "runtime/worker/worker_metrics_shm.hpp"

namespace tt::runners::qwen3_tts {
namespace {

namespace py = pybind11;
namespace fs = std::filesystem;
namespace tts_layout = tt::worker::tts;
using domain::tts::TtsFinishReason;

constexpr const char* PIPELINE_MODULE =
    "models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline";
constexpr const char* WEIGHTS_MODULE = "models.demos.audio.qwen3_tts.weights";

// What the fake pipeline produces per utterance; mirrored in FAKE_PIPELINE.
constexpr uint32_t FAKE_FRAMES = 3;
constexpr uint32_t FAKE_CODEBOOKS = 16;
constexpr size_t FAKE_SAMPLES = 100000;
constexpr uint64_t CODEC_TOKENS_PER_UTTERANCE = FAKE_FRAMES * FAKE_CODEBOOKS;
// 0.5 in float, scaled the way the runner converts to PCM16.
constexpr int16_t FAKE_SAMPLE_VALUE = 16383;

constexpr const char* FAKE_TTNN = R"py(
def open_device(device_id=0, l1_small_size=0, trace_region_size=0):
    return object()


def close_device(device):
    pass
)py";

// Just enough of torch for the runner's tensor <-> PCM16 conversions.
constexpr const char* FAKE_TORCH = R"py(
import numpy as np

int16 = np.int16


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def reshape(self, *shape):
        return Tensor(self.array.reshape(*shape))

    def float(self):
        return Tensor(self.array.astype(np.float32))

    def clamp(self, low, high):
        return Tensor(np.clip(self.array, low, high))

    def mul(self, factor):
        return Tensor(self.array * factor)

    def to(self, dtype):
        return Tensor(self.array.astype(dtype))

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.array))

    def numpy(self):
        return self.array


def from_numpy(array):
    return Tensor(array)
)py";

// The release is a module global so a test can make the checkpoint disagree
// with the deployment (see setCheckpointRelease()).
constexpr const char* FAKE_WEIGHTS = R"py(
KIND = "custom_voice"
SIZE = "1b7"


def model_kind():
    return KIND


def model_size():
    return SIZE


def checkpoint_dir():
    return "/fake/qwen3-tts"
)py";

// Text drives behaviour: "fail" raises, "silent" ends before the first frame,
// "slow" generates long enough to be cancelled, anything else speaks
// FRAMES frames. Every call is recorded in `calls` for the test to inspect.
constexpr const char* FAKE_PIPELINE = R"py(
import time

import numpy as np
import torch

FRAMES = 3
CODEBOOKS = 16
SAMPLES = 100000
calls = []


def reset():
    calls.clear()


def build_clone_reference(device, wav, ref_text, sample_rate=24000,
                          x_vector_only=False):
    time.sleep(0.002)
    calls.append(("build_clone_reference",
                  {"ref_text": ref_text, "sample_rate": sample_rate,
                   "x_vector_only": x_vector_only,
                   "samples": int(wav.numpy().size)}))
    return object()


class Qwen3TTSPipeline:
    def __init__(self, device, max_frames=400, seed=None):
        self.max_frames = max_frames

    def reseed(self, seed):
        calls.append(("reseed", {"seed": seed}))

    def release(self):
        pass

    def _speak(self, method, text, on_frame, **kwargs):
        calls.append((method, dict(text=text, **kwargs)))
        if text == "fail":
            raise RuntimeError("boom\nsecond line")
        if text == "silent":
            raise RuntimeError("talker hit end-of-speech before any frame")
        frames = 2000 if text == "slow" else FRAMES
        for step in range(frames):
            if on_frame is not None:
                on_frame(step, list(range(CODEBOOKS)))
            if text == "slow":
                time.sleep(0.002)
        return torch.from_numpy(np.full((1, SAMPLES), 0.5,
                                        dtype=np.float32)), 24000

    def generate(self, text, speaker=None, language=None, max_frames=None,
                 on_frame=None, instruct=None):
        return self._speak("generate", text, on_frame, speaker=speaker,
                           language=language, instruct=instruct)

    def generate_clone(self, text, reference, language=None, max_frames=None,
                       on_frame=None, x_vector_only=False, instruct=None):
        return self._speak("generate_clone", text, on_frame,
                           language=language, x_vector_only=x_vector_only,
                           instruct=instruct)

    def generate_design(self, text, instruct, language=None, max_frames=None,
                        on_frame=None):
        return self._speak("generate_design", text, on_frame,
                           instruct=instruct, language=language)
)py";

void writeFile(const fs::path& path, const std::string& content) {
  fs::create_directories(path.parent_path());
  std::ofstream(path) << content;
}

std::string uniqueName(const char* prefix) {
  static std::atomic<uint32_t> sequence = 0;
  return std::string(prefix) + "_" + std::to_string(::getpid()) + "_" +
         std::to_string(++sequence);
}

config::TtsConfig makeConfig(config::Qwen3TtsRelease release) {
  config::TtsConfig config;
  config.runner_type = config::ModelRunnerType::TT_QWEN3_TTS;
  config.maxUsers = 1;
  config.taskQueueCapacity = 4;
  config.audioQueueCapacity = 16;
  config.voiceSampleRateHz = config::defaults::TTS_QWEN3_SAMPLE_RATE_HZ;
  config.audioSampleRateHz = config::defaults::TTS_QWEN3_SAMPLE_RATE_HZ;
  config.tokenizerPath.clear();
  config.qwen3Release = release;
  config.qwen3MaxFrames = 8;
  return config;
}

ipc::tts::TtsIpcTask makeTask(uint32_t taskId, std::string text) {
  ipc::tts::TtsIpcTask task;
  task.task_id = taskId;
  task.text = std::move(text);
  return task;
}

/** A runner on its own thread with its own queues, as one worker runs it. */
class RunnerHarness {
 public:
  explicit RunnerHarness(const config::TtsConfig& config)
      : taskQueue(uniqueName("qwen3_tts_task"), 4),
        audioQueue(uniqueName("qwen3_tts_audio"), 16),
        runner(std::make_unique<Qwen3TtsRunner>(config, &taskQueue, &audioQueue,
                                                &cancelQueue)) {
    thread = std::thread([this] {
      try {
        runner->start();
      } catch (const std::exception& e) {
        ADD_FAILURE() << "runner exited: " << e.what();
      }
      exited.store(true);
    });
  }

  ~RunnerHarness() {
    taskQueue.shutdown();
    thread.join();
    runner.reset();
    taskQueue.remove();
    audioQueue.remove();
  }

  /** Every message for `taskId` up to and including its terminal one; empty
   *  when the terminal message does not arrive in time. */
  std::vector<ipc::tts::TtsAudioChunkMessage> collect(uint32_t taskId) {
    std::vector<ipc::tts::TtsAudioChunkMessage> messages;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (std::chrono::steady_clock::now() < deadline) {
      ipc::tts::TtsAudioChunkMessage message;
      if (!audioQueue.tryPop(message)) {
        if (exited.load()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      if (message.task_id != taskId) continue;
      messages.push_back(std::move(message));
      if (messages.back().isFinal()) return messages;
    }
    ADD_FAILURE() << "no terminal message for taskId=" << taskId;
    return {};
  }

  ipc::tts::TtsTaskQueue taskQueue;
  ipc::tts::TtsAudioChunkQueue audioQueue;
  ipc::in_memory::CancelQueue cancelQueue;

 private:
  std::unique_ptr<Qwen3TtsRunner> runner;
  std::atomic<bool> exited{false};
  std::thread thread;
};

/** One recorded call of the fake pipeline: str() of each argument, "None"
 *  for None. Plain strings, so it outlives the GIL it was read under. */
using FakeCall = std::map<std::string, std::string>;

class Qwen3TtsRunnerIntegrationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (pythonChecked) return;
    pythonChecked = true;

    fakeRoot = fs::temp_directory_path() / uniqueName("qwen3_tts_fake_metal");
    writeFile(fakeRoot / "ttnn" / "__init__.py", FAKE_TTNN);
    writeFile(fakeRoot / "torch" / "__init__.py", FAKE_TORCH);
    // Regular packages, so they win over tt-metal's namespace `models` even
    // if a real checkout is also on sys.path.
    for (const auto* dir :
         {"models", "models/demos", "models/demos/audio",
          "models/demos/audio/qwen3_tts", "models/demos/audio/qwen3_tts/tt"}) {
      writeFile(fakeRoot / dir / "__init__.py", "");
    }
    writeFile(fakeRoot / "models/demos/audio/qwen3_tts/weights.py",
              FAKE_WEIGHTS);
    writeFile(
        fakeRoot / "models/demos/audio/qwen3_tts/tt/ttnn_qwen3_pipeline.py",
        FAKE_PIPELINE);
    // The runner puts both first on sys.path.
    ::setenv("TT_METAL_HOME", fakeRoot.c_str(), 1);
    ::setenv("TT_PYTHON_PATH", fakeRoot.c_str(), 1);

    // Started here rather than by the runner so a missing numpy (which the
    // runner's pybind11 arrays need) skips instead of failing warmup. The
    // runner supports an interpreter someone else started.
    py::initialize_interpreter();
    try {
      // First on sys.path before anything imports `models` or `ttnn`, so the
      // fakes are what the runner (and this test) get from sys.modules.
      py::module_::import("sys").attr("path").attr("insert")(0,
                                                             fakeRoot.string());
      py::module_::import("numpy");
      pythonReady = true;
    } catch (const py::error_already_set& e) {
      skipReason = std::string("numpy is not importable: ") + e.what();
    }
    PyEval_SaveThread();
  }

  static void TearDownTestSuite() {
    // Imported modules stay in sys.modules; the files are no longer needed.
    std::error_code ignored;
    fs::remove_all(fakeRoot, ignored);
  }

  void SetUp() override {
    if (!pythonReady) GTEST_SKIP() << skipReason;
    setCheckpointRelease("custom_voice");

    // Own the segment name so the test never touches a running server's, and
    // create it main-side before the runner attaches, as WorkerManager does.
    ASSERT_EQ(::setenv("TT_WORKER_METRICS_SHM",
                       uniqueName("qwen3_tts_metrics").c_str(), 1),
              0);
    shm = tt::worker::WorkerMetricsShm::create(
        tt::config::workerMetricsShmName(), 1);
    ASSERT_NE(shm, nullptr);
    tt::worker::SingleProcessWorkerMetrics::instance().initialize(
        0, tt::worker::MetricsLayout::TTS_RUNNER);

    py::gil_scoped_acquire gil;
    py::module_::import(PIPELINE_MODULE).attr("reset")();
  }

  /** What the fake checkpoint reports as its release
   *  (weights.model_kind()). */
  static void setCheckpointRelease(const char* kind) {
    py::gil_scoped_acquire gil;
    py::module_::import(WEIGHTS_MODULE).attr("KIND") = kind;
  }

  uint64_t codecTokens(tts_layout::VoiceSource source) const {
    return shm->loadScratch(0, tts_layout::codecTokensIdx(source));
  }

  /** The fake pipeline's calls of `method`, oldest first. */
  static std::vector<FakeCall> calls(const std::string& method) {
    py::gil_scoped_acquire gil;
    std::vector<FakeCall> out;
    for (const auto& entry :
         py::module_::import(PIPELINE_MODULE).attr("calls")) {
      const auto call = entry.cast<py::tuple>();
      if (call[0].cast<std::string>() != method) continue;
      FakeCall args;
      for (const auto& [name, value] : call[1].cast<py::dict>()) {
        args[name.cast<std::string>()] = py::str(value).cast<std::string>();
      }
      out.push_back(std::move(args));
    }
    return out;
  }

  /** One argument of the last `method` call. */
  static std::string lastArg(const std::string& method, const char* name) {
    const auto all = calls(method);
    if (all.empty()) return "<no call>";
    const auto it = all.back().find(name);
    return it == all.back().end() ? "<no argument>" : it->second;
  }

  static size_t countCalls(const std::string& method, const std::string& text) {
    size_t count = 0;
    for (const auto& call : calls(method)) {
      if (call.contains("text") && call.at("text") == text) ++count;
    }
    return count;
  }

  std::unique_ptr<tt::worker::WorkerMetricsShm> shm;

  static inline bool pythonChecked = false;
  static inline bool pythonReady = false;
  static inline std::string skipReason;
  static inline fs::path fakeRoot;
};

TEST_F(Qwen3TtsRunnerIntegrationTest, StreamsPcm16ChunksThenOneTerminal) {
  RunnerHarness harness(makeConfig(config::Qwen3TtsRelease::CUSTOM_VOICE));
  auto task = makeTask(1, "hello");
  task.speaker = "ryan";
  harness.taskQueue.push(task);

  const auto messages = harness.collect(1);
  ASSERT_EQ(messages.size(), 3u);  // two chunks, then the terminal message

  // The whole utterance, in TTS_QWEN3_CHUNK_SAMPLES pieces, as PCM16 only.
  size_t samples = 0;
  for (size_t i = 0; i + 1 < messages.size(); ++i) {
    const auto& chunk = messages[i];
    EXPECT_FALSE(chunk.isFinal());
    EXPECT_EQ(chunk.chunkIndex, i);
    EXPECT_EQ(chunk.sampleRateHz, config::defaults::TTS_QWEN3_SAMPLE_RATE_HZ);
    EXPECT_EQ(chunk.channels, 1u);
    EXPECT_TRUE(chunk.samplesBf16.empty());
    ASSERT_FALSE(chunk.samplesPcm16.empty());
    EXPECT_EQ(chunk.samplesPcm16.front(), FAKE_SAMPLE_VALUE);
    samples += chunk.samplesPcm16.size();
  }
  EXPECT_EQ(messages[0].samplesPcm16.size(),
            config::defaults::TTS_QWEN3_CHUNK_SAMPLES);
  EXPECT_EQ(samples, FAKE_SAMPLES);

  const auto& terminal = messages.back();
  EXPECT_TRUE(terminal.isFinal());
  EXPECT_EQ(terminal.finishReason(), TtsFinishReason::Completed);
  EXPECT_TRUE(terminal.error.empty());
  // No voice clip and no worker-side prompt compile: both stages report "did
  // not run", which the parent skips.
  EXPECT_EQ(terminal.voiceEncodeUs, 0u);
  EXPECT_EQ(terminal.promptCompileUs, 0u);

  // A named speaker speaks English unless the request says otherwise.
  EXPECT_EQ(lastArg("generate", "speaker"), "ryan");
  EXPECT_EQ(lastArg("generate", "language"), "English");
  EXPECT_EQ(lastArg("generate", "instruct"), "None");
}

TEST_F(Qwen3TtsRunnerIntegrationTest, PublishesCodecTokensAndVocodedAudio) {
  const auto config = makeConfig(config::Qwen3TtsRelease::CUSTOM_VOICE);
  {
    RunnerHarness harness(config);
    auto plain = makeTask(1, "hello");
    plain.speaker = "ryan";
    harness.taskQueue.push(plain);
    ASSERT_FALSE(harness.collect(1).empty());

    auto instructed = makeTask(2, "hello");
    instructed.speaker = "ryan";
    instructed.description = "Speak slowly.";
    harness.taskQueue.push(instructed);
    ASSERT_FALSE(harness.collect(2).empty());
  }

  // One codec token per codebook per frame, under the same voice_source
  // labels BlazeTtsRunner uses; warmup is not counted.
  EXPECT_EQ(codecTokens(tts_layout::VoiceSource::Default),
            CODEC_TOKENS_PER_UTTERANCE);
  EXPECT_EQ(codecTokens(tts_layout::VoiceSource::Description),
            CODEC_TOKENS_PER_UTTERANCE);
  EXPECT_EQ(codecTokens(tts_layout::VoiceSource::VoiceSample), 0u);

  // Batch 1: every utterance lands in the "1" bucket.
  EXPECT_EQ(shm->loadScratch(
                0, tts_layout::audioFramesIdx(tts_layout::BatchBucket::B1)),
            2 * FAKE_SAMPLES);
  EXPECT_EQ(shm->loadScratch(
                0, tts_layout::vocoderChunksIdx(tts_layout::BatchBucket::B1)),
            4u);
  EXPECT_EQ(shm->loadScratch(
                0, tts_layout::audioFramesIdx(tts_layout::BatchBucket::B2)),
            0u);
  EXPECT_EQ(shm->loadScratch(0, tts_layout::SCRATCH_AUDIO_SAMPLE_RATE_HZ),
            config.audioSampleRateHz);
  EXPECT_NE(shm->loadScratch(0, tts_layout::SCRATCH_LAST_OUTPUT_EPOCH_MS), 0u);
  EXPECT_NE(shm->loadScratch(0, tts_layout::SCRATCH_LAST_VOCODE_EPOCH_MS), 0u);
}

TEST_F(Qwen3TtsRunnerIntegrationTest, CloneReportsVoiceEncodeTime) {
  setCheckpointRelease("base");
  RunnerHarness harness(makeConfig(config::Qwen3TtsRelease::BASE));
  const std::vector<int16_t> clip(2400, 1000);

  // Voice-only clone: no transcript.
  auto voiceOnly = makeTask(1, "hello");
  voiceOnly.voiceWavPcm = clip;
  harness.taskQueue.push(voiceOnly);
  auto messages = harness.collect(1);
  ASSERT_FALSE(messages.empty());
  EXPECT_EQ(messages.back().finishReason(), TtsFinishReason::Completed);
  // Building the clone reference is the worker's voice_encode stage, carried
  // on the terminal message as BlazeTtsRunner carries its encoder time.
  EXPECT_GT(messages.back().voiceEncodeUs, 0u);
  EXPECT_EQ(messages.back().promptCompileUs, 0u);
  EXPECT_EQ(lastArg("build_clone_reference", "x_vector_only"), "True");
  EXPECT_EQ(lastArg("build_clone_reference", "ref_text"), "None");
  EXPECT_EQ(lastArg("build_clone_reference", "samples"), "2400");
  EXPECT_EQ(lastArg("generate_clone", "language"), "Auto");

  // In-context clone: the transcript travels with the clip.
  auto inContext = makeTask(2, "hello");
  inContext.voiceWavPcm = clip;
  inContext.referenceText = "what the clip says";
  inContext.language = "German";
  harness.taskQueue.push(inContext);
  messages = harness.collect(2);
  ASSERT_FALSE(messages.empty());
  EXPECT_EQ(messages.back().finishReason(), TtsFinishReason::Completed);
  EXPECT_EQ(lastArg("build_clone_reference", "x_vector_only"), "False");
  EXPECT_EQ(lastArg("build_clone_reference", "ref_text"), "what the clip says");
  EXPECT_EQ(lastArg("generate_clone", "language"), "German");

  EXPECT_EQ(codecTokens(tts_layout::VoiceSource::VoiceSample),
            2 * CODEC_TOKENS_PER_UTTERANCE);
}

TEST_F(Qwen3TtsRunnerIntegrationTest, CancelBeforeDequeueSkipsGeneration) {
  RunnerHarness harness(makeConfig(config::Qwen3TtsRelease::CUSTOM_VOICE));

  // The parent broadcasts a cancel to every worker, possibly before the task
  // reaches one; the runner must still honour it once the task arrives.
  harness.cancelQueue.push(7);
  auto cancelled = makeTask(7, "never spoken");
  cancelled.speaker = "ryan";
  harness.taskQueue.push(cancelled);
  const auto messages = harness.collect(7);
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages.back().finishReason(), TtsFinishReason::Cancelled);
  EXPECT_EQ(countCalls("generate", "never spoken"), 0u);

  // The worker keeps serving.
  auto next = makeTask(8, "hello");
  next.speaker = "ryan";
  harness.taskQueue.push(next);
  const auto nextMessages = harness.collect(8);
  ASSERT_FALSE(nextMessages.empty());
  EXPECT_EQ(nextMessages.back().finishReason(), TtsFinishReason::Completed);
}

TEST_F(Qwen3TtsRunnerIntegrationTest, CancelDuringGenerationStopsTheUtterance) {
  RunnerHarness harness(makeConfig(config::Qwen3TtsRelease::CUSTOM_VOICE));
  auto task = makeTask(9, "slow");
  task.speaker = "ryan";
  harness.taskQueue.push(task);

  // Cancel once frames are coming out, i.e. mid-utterance.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(20);
  while (codecTokens(tts_layout::VoiceSource::Default) == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_GT(codecTokens(tts_layout::VoiceSource::Default), 0u);
  harness.cancelQueue.push(9);

  // The per-frame callback sees the cancel: no audio, one Cancelled terminal.
  const auto messages = harness.collect(9);
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages.back().finishReason(), TtsFinishReason::Cancelled);
  EXPECT_LT(codecTokens(tts_layout::VoiceSource::Default),
            2000u * FAKE_CODEBOOKS);
}

TEST_F(Qwen3TtsRunnerIntegrationTest, ModelErrorsEndTheTaskWithAnError) {
  RunnerHarness harness(makeConfig(config::Qwen3TtsRelease::CUSTOM_VOICE));

  auto failing = makeTask(1, "fail");
  failing.speaker = "ryan";
  harness.taskQueue.push(failing);
  auto messages = harness.collect(1);
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages.back().finishReason(), TtsFinishReason::Error);
  // The Python exception's first line only, without its traceback.
  EXPECT_EQ(messages.back().error, "RuntimeError: boom");

  auto silent = makeTask(2, "silent");
  silent.speaker = "ryan";
  harness.taskQueue.push(silent);
  messages = harness.collect(2);
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages.back().finishReason(), TtsFinishReason::Error);
  EXPECT_EQ(messages.back().error,
            "TTS model produced no audio for this text; try again");

  auto next = makeTask(3, "hello");
  next.speaker = "ryan";
  harness.taskQueue.push(next);
  messages = harness.collect(3);
  ASSERT_FALSE(messages.empty());
  EXPECT_EQ(messages.back().finishReason(), TtsFinishReason::Completed);
}

TEST_F(Qwen3TtsRunnerIntegrationTest, WarmupRefusesAnotherRelease) {
  // The deployment was configured for CustomVoice; the checkpoint is Base.
  setCheckpointRelease("base");
  ipc::tts::TtsTaskQueue taskQueue(uniqueName("qwen3_tts_task"), 4);
  ipc::tts::TtsAudioChunkQueue audioQueue(uniqueName("qwen3_tts_audio"), 4);
  ipc::in_memory::CancelQueue cancelQueue;
  {
    Qwen3TtsRunner runner(makeConfig(config::Qwen3TtsRelease::CUSTOM_VOICE),
                          &taskQueue, &audioQueue, &cancelQueue);
    EXPECT_FALSE(runner.warmup());
  }
  EXPECT_TRUE(calls("generate").empty());
  taskQueue.remove();
  audioQueue.remove();
}

TEST(Qwen3TtsRunnerTest, RejectsAnOutputRateOtherThanTheCodecs) {
  ipc::tts::TtsTaskQueue taskQueue(uniqueName("qwen3_tts_task"), 4);
  ipc::tts::TtsAudioChunkQueue audioQueue(uniqueName("qwen3_tts_audio"), 4);
  ipc::in_memory::CancelQueue cancelQueue;
  auto config = makeConfig(config::Qwen3TtsRelease::CUSTOM_VOICE);
  config.audioSampleRateHz = 48000;
  EXPECT_THROW(Qwen3TtsRunner(config, &taskQueue, &audioQueue, &cancelQueue),
               std::invalid_argument);
  taskQueue.remove();
  audioQueue.remove();
}

}  // namespace
}  // namespace tt::runners::qwen3_tts
