// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/qwen3_tts/qwen3_tts_runner.hpp"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <thread>
#include <utility>

#include "config/defaults.hpp"
#include "runtime/worker/single_process_worker_metrics.hpp"
#include "runtime/worker/tts_metrics_layout.hpp"
#include "utils/logger.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace tt::runners::qwen3_tts {

namespace {

constexpr const char* PIPELINE_MODULE =
    "models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline";
constexpr const char* WEIGHTS_MODULE = "models.demos.audio.qwen3_tts.weights";

// Device knobs from models/demos/audio/qwen3_tts/demo/demo.py:open_device():
// the two decode traces need the trace region, the codec's convolutions the
// L1_SMALL scratch (65536 measured best; smaller starves the codec, larger
// steals L1 from it).
constexpr int L1_SMALL_SIZE = 65536;
constexpr int64_t TRACE_REGION_SIZE = 90'000'000;

// Short warmup utterance: enough to compile prefill, both decode traces and
// one codec length bucket without holding up readiness for long.
constexpr const char* WARMUP_TEXT = "Hello, this is a warm-up.";
constexpr uint32_t WARMUP_MAX_FRAMES = 32;
constexpr const char* WARMUP_VOICE_DESCRIPTION =
    "A clear, neutral adult voice at a moderate pace.";
constexpr const char* WARMUP_SPEAKER = "ryan";
constexpr double WARMUP_TONE_SECONDS = 3.0;
constexpr double WARMUP_TONE_HZ = 220.0;

// Cancels for tasks this worker may never see (the parent broadcasts them to
// every worker); remembered only this long.
constexpr size_t CANCELLED_TASK_MEMORY = 4096;

constexpr auto IDLE_POLL = std::chrono::milliseconds(2);
constexpr auto PUSH_RETRY = std::chrono::milliseconds(1);

// What the language defaults to when a request leaves it out, as
// models/demos/audio/qwen3_tts/demo/demo_server.py does: a named speaker speaks
// English unless told otherwise; a clone or a designed voice lets the model
// infer the language.
constexpr const char* DEFAULT_SPEAKER_LANGUAGE = "English";
constexpr const char* AUTO_LANGUAGE = "Auto";

// Thrown by the per-frame callback; the flag beside it is what tells a cancel
// apart from a model error once the exception has crossed Python.
constexpr const char* CANCELLED_MESSAGE = "qwen3-tts request cancelled";

/** No GIL needed: thrown from C++ code between Python calls. */
struct TaskCancelled {};

std::string firstLine(const std::string& s) {
  const auto pos = s.find('\n');
  return pos == std::string::npos ? s : s.substr(0, pos);
}

uint32_t elapsedUsSince(std::chrono::steady_clock::time_point start) {
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  if (us <= 0) return 0;
  return static_cast<uint32_t>(
      std::min<int64_t>(us, std::numeric_limits<uint32_t>::max()));
}

/** Per-worker process environment; must run before the interpreter starts
 *  and before ttnn reads it. Mirrors embedding_worker_main.cpp: a private
 *  TT_METAL_CACHE (concurrent JIT builds in one directory race) and the
 *  tt-metal checkout as working directory (kernels use relative includes). */
void exportWorkerEnvironment(const config::TtsConfig& cfg) {
  if (!cfg.visibleDevices.empty()) {
    setenv("TT_VISIBLE_DEVICES", cfg.visibleDevices.c_str(), 1);
  }
  // Host work is small (sampling, embeddings, the mel front-end); keep a
  // worker per chip from oversubscribing the CPU. Overridable.
  setenv("OMP_NUM_THREADS", "2", 0);
  setenv("MKL_NUM_THREADS", "2", 0);

  const char* metalHome = std::getenv("TT_METAL_HOME");
  if (!metalHome || !*metalHome) {
    TT_LOG_WARN(
        "[Qwen3TtsRunner] TT_METAL_HOME is not set; the worker cannot import "
        "the model or compile kernels");
    return;
  }
  std::string deviceSuffix = cfg.visibleDevices.empty()
                                 ? "worker" + std::to_string(cfg.workerId)
                                 : cfg.visibleDevices;
  std::replace(deviceSuffix.begin(), deviceSuffix.end(), ',', '_');
  const std::string metalCache =
      std::string(metalHome) + "/built/" + deviceSuffix;
  setenv("TT_METAL_CACHE", metalCache.c_str(), 1);
  if (chdir(metalHome) != 0) {
    TT_LOG_ERROR(
        "[Qwen3TtsRunner] chdir to TT_METAL_HOME '{}' failed; kernels with "
        "tt-metal-relative include paths will not compile",
        metalHome);
  }
  TT_LOG_INFO(
      "[Qwen3TtsRunner] Worker {} environment: TT_VISIBLE_DEVICES={} "
      "TT_METAL_CACHE={} cwd={}",
      cfg.workerId, cfg.visibleDevices.empty() ? "(all)" : cfg.visibleDevices,
      metalCache, metalHome);
}

/** Put TT_METAL_HOME (models.demos lives there) and TT_PYTHON_PATH on
 *  sys.path, as the SDXL and embedding runners do. */
void ensureSysPath() {
  py::list sysPath = py::module_::import("sys").attr("path");
  auto prepend = [&sysPath](const char* envName) {
    const char* value = std::getenv(envName);
    if (!value || !*value) return;
    for (const auto& entry : sysPath) {
      if (py::str(entry).cast<std::string>() == value) return;
    }
    sysPath.attr("insert")(0, py::str(value));
    TT_LOG_INFO("[Qwen3TtsRunner] Prepended {} to sys.path: {}", envName,
                value);
  };
  prepend("TT_METAL_HOME");
  prepend("TT_PYTHON_PATH");
}

py::object optionalStr(const std::optional<std::string>& value) {
  return value ? py::object(py::str(*value)) : py::object(py::none());
}

tt::worker::tts::VoiceSource voiceSourceOf(const ipc::tts::TtsIpcTask& task) {
  if (!task.voiceWavPcm.empty()) {
    return tt::worker::tts::VoiceSource::VoiceSample;
  }
  if (task.description.has_value()) {
    return tt::worker::tts::VoiceSource::Description;
  }
  return tt::worker::tts::VoiceSource::Default;
}

}  // namespace

namespace detail {

struct __attribute__((visibility("hidden"))) PythonState {
  bool ownsInterpreter = false;
  py::object torch;
  py::object ttnn;
  py::object pipelineModule;
  py::object device;
  py::object pipeline;

  /** int16 PCM -> float32 tensor in [-1, 1), as soundfile reads a PCM16 WAV. */
  py::object pcmToTensor(const std::vector<int16_t>& pcm) const {
    py::array_t<float> samples(static_cast<py::ssize_t>(pcm.size()));
    float* out = samples.mutable_data();
    for (size_t i = 0; i < pcm.size(); ++i) {
      out[i] = static_cast<float>(pcm[i]) / 32768.0F;
    }
    return torch.attr("from_numpy")(samples);
  }

  /** Waveform tensor [1, N] (float, 24 kHz) -> PCM16, clamped and truncated
   *  the way the TTS-2 bf16 path converts. */
  std::vector<int16_t> tensorToPcm(const py::object& waveform) const {
    py::object pcm = waveform.attr("reshape")(-1)
                         .attr("float")()
                         .attr("clamp")(-1.0, 1.0)
                         .attr("mul")(32767.0)
                         .attr("to")(torch.attr("int16"))
                         .attr("contiguous")()
                         .attr("numpy")();
    auto samples = py::array_t < int16_t,
         py::array::c_style | py::array::forcecast > (pcm);
    return std::vector<int16_t>(samples.data(),
                                samples.data() + samples.size());
  }

  void closeDevice() {
    pipeline = py::object();
    if (device && !device.is_none() && ttnn) {
      try {
        ttnn.attr("close_device")(device);
      } catch (const py::error_already_set& e) {
        TT_LOG_WARN("[Qwen3TtsRunner] close_device failed: {}",
                    firstLine(e.what()));
      }
    }
    device = py::object();
  }
};

}  // namespace detail

Qwen3TtsRunner::Qwen3TtsRunner(config::TtsConfig config,
                               ipc::tts::TtsTaskQueue* taskQueue,
                               ipc::tts::TtsAudioChunkQueue* audioQueue,
                               ipc::ICancelQueue* cancelQueue)
    : config(std::move(config)),
      taskQueue(taskQueue),
      audioQueue(audioQueue),
      cancelQueue(cancelQueue),
      cancelled(CANCELLED_TASK_MEMORY),
      python(std::make_unique<detail::PythonState>()) {
  if (!this->taskQueue) {
    throw std::invalid_argument("Qwen3TtsRunner: taskQueue must not be null");
  }
  if (!this->audioQueue) {
    throw std::invalid_argument("Qwen3TtsRunner: audioQueue must not be null");
  }
  if (!this->cancelQueue) {
    throw std::invalid_argument("Qwen3TtsRunner: cancelQueue must not be null");
  }
  if (this->config.audioSampleRateHz !=
      config::defaults::TTS_QWEN3_SAMPLE_RATE_HZ) {
    throw std::invalid_argument(
        "Qwen3TtsRunner: the codec produces 24 kHz audio, but the configured "
        "output rate is " +
        std::to_string(this->config.audioSampleRateHz) + " Hz");
  }
  exportWorkerEnvironment(this->config);
  // Fixed for the runner's lifetime; turns the frame counter into seconds.
  tt::worker::SingleProcessWorkerMetrics::instance().publishAudioSampleRate(
      this->config.audioSampleRateHz);
}

Qwen3TtsRunner::~Qwen3TtsRunner() {
  stop();
  if (!python || !Py_IsInitialized()) return;
  try {
    py::gil_scoped_acquire gil;
    if (python->pipeline) {
      try {
        python->pipeline.attr("release")();
      } catch (const py::error_already_set&) {
        // Closing the device drops the traces anyway.
      }
    }
    python->closeDevice();
    python->pipelineModule = py::object();
    python->ttnn = py::object();
    python->torch = py::object();
  } catch (...) {
    // Destructors must not throw; the process is going away regardless.
  }
}

void Qwen3TtsRunner::stop() { stopped.store(true, std::memory_order_release); }

bool Qwen3TtsRunner::warmup() {
  python->ownsInterpreter = !Py_IsInitialized();
  if (python->ownsInterpreter) {
    py::initialize_interpreter();
    TT_LOG_INFO("[Qwen3TtsRunner] Python interpreter initialized");
  }

  bool ok = false;
  {
    py::gil_scoped_acquire gil;
    try {
      ensureSysPath();
      python->torch = py::module_::import("torch");

      // The deployment chose a release up front, and the parent has been
      // validating requests against it; refuse to serve a checkpoint that
      // turns out to be a different one.
      py::module_ weights = py::module_::import(WEIGHTS_MODULE);
      const auto kind = weights.attr("model_kind")().cast<std::string>();
      const auto size = weights.attr("model_size")().cast<std::string>();
      const std::string wantKind = config::toString(config.qwen3Release);
      const std::string wantSize = config::toString(config.qwen3ModelSize);
      if (kind != wantKind || size != wantSize) {
        throw std::runtime_error(
            "the checkpoint is a " + size + " " + kind +
            " release, but this deployment is configured for " + wantSize +
            " " + wantKind +
            "; fix HF_MODEL / QWEN3_TTS_CKPT or TTS_QWEN3_RELEASE / "
            "TTS_QWEN3_MODEL_SIZE");
      }
      TT_LOG_INFO("[Qwen3TtsRunner] Checkpoint {} {} at {}", size, kind,
                  weights.attr("checkpoint_dir")().cast<std::string>());

      python->ttnn = py::module_::import("ttnn");
      python->device = python->ttnn.attr("open_device")(
          "device_id"_a = 0, "l1_small_size"_a = L1_SMALL_SIZE,
          "trace_region_size"_a = TRACE_REGION_SIZE);
      TT_LOG_INFO(
          "[Qwen3TtsRunner] Device opened (TT_VISIBLE_DEVICES={})",
          config.visibleDevices.empty() ? "(all)" : config.visibleDevices);

      python->pipelineModule = py::module_::import(PIPELINE_MODULE);
      py::object seed = config.qwen3Seed
                            ? py::object(py::int_(*config.qwen3Seed))
                            : py::object(py::none());
      python->pipeline = python->pipelineModule.attr("Qwen3TTSPipeline")(
          python->device, "max_frames"_a = config.qwen3MaxFrames,
          "seed"_a = seed);
      TT_LOG_INFO("[Qwen3TtsRunner] Pipeline built (max_frames={})",
                  config.qwen3MaxFrames);

      const auto started = std::chrono::steady_clock::now();
      try {
        const uint32_t frames =
            std::min(config.qwen3MaxFrames, WARMUP_MAX_FRAMES);
        switch (config.qwen3Release) {
          case config::Qwen3TtsRelease::BASE: {
            // A synthetic tone, voice-only: compiles the speaker encoder and
            // the decode path without needing a real clip or transcript.
            const auto count = static_cast<size_t>(WARMUP_TONE_SECONDS *
                                                   config.voiceSampleRateHz);
            std::vector<int16_t> tone(count);
            for (size_t i = 0; i < count; ++i) {
              tone[i] = static_cast<int16_t>(
                  9830.0 * std::sin(2.0 * std::numbers::pi * WARMUP_TONE_HZ *
                                    i / config.voiceSampleRateHz));
            }
            py::object reference =
                python->pipelineModule.attr("build_clone_reference")(
                    python->device, python->pcmToTensor(tone), py::none(),
                    "sample_rate"_a = config.voiceSampleRateHz,
                    "x_vector_only"_a = true);
            python->pipeline.attr("generate_clone")(
                WARMUP_TEXT, reference, "language"_a = AUTO_LANGUAGE,
                "max_frames"_a = frames, "x_vector_only"_a = true);
            break;
          }
          case config::Qwen3TtsRelease::CUSTOM_VOICE: {
            std::string speaker = WARMUP_SPEAKER;
            if (!config.qwen3Speakers.empty() &&
                std::find(config.qwen3Speakers.begin(),
                          config.qwen3Speakers.end(),
                          speaker) == config.qwen3Speakers.end()) {
              speaker = config.qwen3Speakers.front();
            }
            python->pipeline.attr("generate")(
                WARMUP_TEXT, "speaker"_a = speaker,
                "language"_a = DEFAULT_SPEAKER_LANGUAGE,
                "max_frames"_a = frames);
            break;
          }
          case config::Qwen3TtsRelease::VOICE_DESIGN:
            python->pipeline.attr("generate_design")(
                WARMUP_TEXT, WARMUP_VOICE_DESCRIPTION,
                "language"_a = AUTO_LANGUAGE, "max_frames"_a = frames);
            break;
        }
      } catch (const py::error_already_set& e) {
        // A sampled first frame can be end-of-speech; everything up to it has
        // compiled, which is what warmup is for.
        if (std::string(e.what()).find("end-of-speech before any frame") ==
            std::string::npos) {
          throw;
        }
        TT_LOG_WARN(
            "[Qwen3TtsRunner] Warmup utterance ended before its first frame; "
            "continuing");
      }
      TT_LOG_INFO("[Qwen3TtsRunner] Warmup utterance done in {:.1f} s",
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - started)
                      .count());
      ok = true;
    } catch (const py::error_already_set& e) {
      TT_LOG_ERROR("[Qwen3TtsRunner] Warmup failed:\n{}", e.what());
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[Qwen3TtsRunner] Warmup failed: {}", e.what());
    }
    if (!ok) {
      python->closeDevice();
    }
  }

  if (python->ownsInterpreter) {
    // initialize_interpreter() left this thread holding the GIL; hand it back
    // so every later section takes it explicitly.
    PyEval_SaveThread();
  }
  return ok;
}

void Qwen3TtsRunner::run() {
  TT_LOG_INFO("[Qwen3TtsRunner] Serving {} {} (worker {})",
              config::toString(config.qwen3ModelSize),
              config::toString(config.qwen3Release), config.workerId);
  auto& metrics = tt::worker::SingleProcessWorkerMetrics::instance();
  ipc::tts::TtsIpcTask task;
  while (!stopped.load(std::memory_order_acquire)) {
    metrics.updateStepHeartbeat();
    drainCancels();
    if (!taskQueue->tryPop(task)) {
      std::this_thread::sleep_for(IDLE_POLL);
      continue;
    }
    if (task.isDone()) {
      stopped.store(true, std::memory_order_release);
      break;
    }
    handleTask(task);
  }
  TT_LOG_INFO("[Qwen3TtsRunner] Stopped");
}

void Qwen3TtsRunner::drainCancels() {
  cancelScratch.clear();
  cancelQueue->tryPopAll(cancelScratch);
  for (uint32_t id : cancelScratch) {
    cancelled.add(id);
  }
}

bool Qwen3TtsRunner::pollCancelled(uint32_t taskId) {
  drainCancels();
  return cancelled.contains(taskId);
}

void Qwen3TtsRunner::handleTask(const ipc::tts::TtsIpcTask& task) {
  const uint32_t taskId = task.task_id;
  if (pollCancelled(taskId)) {
    cancelled.take(taskId);
    sendFinish(taskId, domain::tts::TtsFinishReason::Cancelled);
    return;
  }

  auto reason = domain::tts::TtsFinishReason::Completed;
  std::string error;
  uint32_t voiceEncodeUs = 0;
  std::vector<int16_t> pcm;
  bool cancelRequested = false;
  const auto voiceSource = voiceSourceOf(task);
  const auto started = std::chrono::steady_clock::now();

  try {
    py::gil_scoped_acquire gil;
    try {
      auto& pipeline = python->pipeline;
      if (config.qwen3Seed) {
        // Reproducible per request, not per position in the worker's history.
        pipeline.attr("reseed")(*config.qwen3Seed);
      }

      // on_frame(step, frame): once per generated 12.5 Hz frame.
      py::cpp_function onFrame([this, taskId, voiceSource, &cancelRequested](
                                   const py::object& /*step*/,
                                   const py::object& frame) {
        auto& metrics = tt::worker::SingleProcessWorkerMetrics::instance();
        // One codec token per codebook: the talker's plus the predictor's.
        const size_t codes = py::len(frame);
        for (size_t i = 0; i < codes; ++i) {
          metrics.onCodecToken(voiceSource);
        }
        metrics.updateStepHeartbeat();
        if (stopped.load(std::memory_order_acquire) || pollCancelled(taskId)) {
          cancelRequested = true;
          throw std::runtime_error(CANCELLED_MESSAGE);
        }
      });

      const auto language = [&task](const char* fallback) {
        return task.language ? *task.language : std::string(fallback);
      };

      py::object result;
      switch (config.qwen3Release) {
        case config::Qwen3TtsRelease::BASE: {
          // Voice-only (x-vector) without a transcript, in-context with one.
          const bool xVectorOnly = !task.referenceText.has_value();
          const auto encodeStart = std::chrono::steady_clock::now();
          // Eager device work beside a live trace hangs the card, and the
          // previous request left its traces captured.
          pipeline.attr("release")();
          py::object reference =
              python->pipelineModule.attr("build_clone_reference")(
                  python->device, python->pcmToTensor(task.voiceWavPcm),
                  optionalStr(task.referenceText),
                  "sample_rate"_a = config.voiceSampleRateHz,
                  "x_vector_only"_a = xVectorOnly);
          voiceEncodeUs = elapsedUsSince(encodeStart);
          if (pollCancelled(taskId)) throw TaskCancelled{};
          result = pipeline.attr("generate_clone")(
              task.text, reference, "language"_a = language(AUTO_LANGUAGE),
              "on_frame"_a = onFrame, "x_vector_only"_a = xVectorOnly,
              "instruct"_a = xVectorOnly ? optionalStr(task.description)
                                         : py::object(py::none()));
          break;
        }
        case config::Qwen3TtsRelease::CUSTOM_VOICE:
          result = pipeline.attr("generate")(
              task.text, "speaker"_a = optionalStr(task.speaker),
              "language"_a = language(DEFAULT_SPEAKER_LANGUAGE),
              "on_frame"_a = onFrame,
              "instruct"_a = optionalStr(task.description));
          break;
        case config::Qwen3TtsRelease::VOICE_DESIGN:
          result = pipeline.attr("generate_design")(
              task.text, optionalStr(task.description),
              "language"_a = language(AUTO_LANGUAGE), "on_frame"_a = onFrame);
          break;
      }
      pcm = python->tensorToPcm(result.cast<py::tuple>()[0]);
    } catch (const py::error_already_set& e) {
      // Destroyed under the GIL, inside this scope.
      if (cancelRequested) {
        reason = domain::tts::TtsFinishReason::Cancelled;
      } else {
        reason = domain::tts::TtsFinishReason::Error;
        const std::string what = e.what();
        error = what.find("end-of-speech before any frame") != std::string::npos
                    ? "the model produced no audio for this text; try again"
                    : firstLine(what);
        TT_LOG_ERROR("[Qwen3TtsRunner] Task {} failed:\n{}", taskId, what);
      }
    }
  } catch (const TaskCancelled&) {
    reason = domain::tts::TtsFinishReason::Cancelled;
  } catch (const std::exception& e) {
    reason = domain::tts::TtsFinishReason::Error;
    error = e.what();
    TT_LOG_ERROR("[Qwen3TtsRunner] Task {} failed: {}", taskId, e.what());
  } catch (...) {
    reason = domain::tts::TtsFinishReason::Error;
    error = "unknown error in the Qwen3-TTS runner";
    TT_LOG_ERROR("[Qwen3TtsRunner] Task {} failed with an unknown exception",
                 taskId);
  }

  if (reason == domain::tts::TtsFinishReason::Completed) {
    const auto chunks =
        splitIntoChunks(taskId, pcm, config::defaults::TTS_QWEN3_CHUNK_SAMPLES,
                        config.audioSampleRateHz);
    for (const auto& chunk : chunks) {
      if (pollCancelled(taskId) || !pushBlocking(chunk)) {
        reason = domain::tts::TtsFinishReason::Cancelled;
        break;
      }
    }
    if (!chunks.empty()) {
      tt::worker::SingleProcessWorkerMetrics::instance().onVocodedAudio(
          tt::worker::tts::batchBucketOf(1), pcm.size(), chunks.size());
    }
    TT_LOG_INFO(
        "[Qwen3TtsRunner] Task {} done: {:.2f} s of audio in {:.2f} s ({} "
        "chunks)",
        taskId, static_cast<double>(pcm.size()) / config.audioSampleRateHz,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      started)
            .count(),
        chunks.size());
  }

  cancelled.take(taskId);
  sendFinish(taskId, reason, std::move(error), voiceEncodeUs);
}

bool Qwen3TtsRunner::pushBlocking(
    const ipc::tts::TtsAudioChunkMessage& message) {
  while (!audioQueue->push(message)) {
    if (stopped.load(std::memory_order_acquire)) {
      return false;
    }
    std::this_thread::sleep_for(PUSH_RETRY);
  }
  return true;
}

void Qwen3TtsRunner::sendFinish(uint32_t taskId,
                                domain::tts::TtsFinishReason reason,
                                std::string error, uint32_t voiceEncodeUs) {
  auto message =
      ipc::tts::TtsAudioChunkMessage::finish(taskId, reason, std::move(error));
  // Carried on the terminal message, the one the parent sees exactly once.
  message.voiceEncodeUs = voiceEncodeUs;
  if (!pushBlocking(message)) {
    TT_LOG_WARN(
        "[Qwen3TtsRunner] Shutting down before the terminal message for task "
        "{} was delivered",
        taskId);
  }
}

}  // namespace tt::runners::qwen3_tts
