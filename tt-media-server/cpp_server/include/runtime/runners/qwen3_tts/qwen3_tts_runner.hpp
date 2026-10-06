// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/tts/tts_types.hpp"
#include "ipc/interface/cancel_queue.hpp"
#include "ipc/tts_ipc.hpp"
#include "runtime/runners/ipc_runner.hpp"
#include "runtime/runners/qwen3_tts/qwen3_tts_runner_support.hpp"

namespace tt::runners::qwen3_tts {

namespace detail {
/** The embedded-Python half (interpreter, ttnn device, pipeline). Kept behind
 *  this declaration so pybind11 types never leak into headers. */
struct PythonState;
}  // namespace detail

/**
 * Qwen3-TTS worker runner: drives tt-metal's
 * `models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline` through an embedded
 * interpreter, one utterance at a time (batch 1).
 *
 * Consumes the shared TTS task queue and answers on this worker's audio queue
 * with PCM16 chunks followed by exactly one terminal message (completed, error
 * or cancelled) per task. The codec decodes a whole utterance at once, so all
 * of an utterance's audio arrives together after generation; the cancel queue
 * is polled once per generated frame (12.5 Hz) to abort early.
 */
class Qwen3TtsRunner : public IRunner {
 public:
  Qwen3TtsRunner(config::TtsConfig config, ipc::tts::TtsTaskQueue* taskQueue,
                 ipc::tts::TtsAudioChunkQueue* audioQueue,
                 ipc::ICancelQueue* cancelQueue);
  ~Qwen3TtsRunner() override;

  Qwen3TtsRunner(const Qwen3TtsRunner&) = delete;
  Qwen3TtsRunner& operator=(const Qwen3TtsRunner&) = delete;

  /** Open the device, load the release's weights, check they are the release
   *  this deployment was configured for, and speak one short utterance so
   *  kernels compile before the worker reports ready. */
  bool warmup() override;
  void stop() override;
  const char* runnerType() const override { return "Qwen3TtsRunner"; }

 private:
  void run() override;
  void handleTask(const ipc::tts::TtsIpcTask& task);
  /** Move everything on the cancel queue into `cancelled`. */
  void drainCancels();
  /** drainCancels(), then whether `taskId` has been cancelled. */
  bool pollCancelled(uint32_t taskId);
  /** Push, retrying while the parent drains the queue. False on shutdown. */
  bool pushBlocking(const ipc::tts::TtsAudioChunkMessage& message);
  void sendFinish(uint32_t taskId, domain::tts::TtsFinishReason reason,
                  std::string error = {}, uint32_t voiceEncodeUs = 0);

  config::TtsConfig config;
  ipc::tts::TtsTaskQueue* taskQueue;
  ipc::tts::TtsAudioChunkQueue* audioQueue;
  ipc::ICancelQueue* cancelQueue;
  CancelledTaskSet cancelled;
  std::vector<uint32_t> cancelScratch;
  std::unique_ptr<detail::PythonState> python;
  std::atomic<bool> stopped{false};
};

}  // namespace tt::runners::qwen3_tts
