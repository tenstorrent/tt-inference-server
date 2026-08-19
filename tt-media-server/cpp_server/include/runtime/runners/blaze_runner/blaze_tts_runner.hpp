// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "config/runner_config.hpp"
#include "ipc/interface/cancel_queue.hpp"
#include "ipc/tts_ipc.hpp"
#include "runtime/runners/blaze_runner/tts_scheduler_interface.hpp"
#include "runtime/runners/ipc_runner.hpp"
#include "utils/voice_sample_cache.hpp"

namespace tt::runners::blaze {

class BlazeTtsRunner : public IRunner {
 public:
  BlazeTtsRunner(config::TtsConfig config,
                 std::unique_ptr<tts_scheduler::ITtsScheduler> ttsScheduler,
                 ipc::tts::TtsTaskQueue* taskQueue,
                 ipc::tts::TtsAudioChunkQueue* audioQueue,
                 ipc::ICancelQueue* cancelQueue);
  ~BlazeTtsRunner() override;

  bool warmup() override;
  void stop() override;
  const char* runnerType() const override { return "BlazeTtsRunner"; }

 private:
  enum class SlotState {
    FREE,
    ALLOCATING,
    RUNNING,
    AWAITING_TERMINAL_DELIVERY,
    AWAITING_STOP_ACK,
    AWAITING_EVICT_ACK,
  };

  struct SlotContext {
    SlotState state = SlotState::FREE;
    uint32_t slotId = tts_scheduler::INVALID_SLOT;
    uint32_t task_id = 0;
    bool completionPending = false;
    bool audioLastReceived = false;
  };

  struct PendingTerminalMessage {
    ipc::tts::TtsAudioChunkMessage message;
    std::optional<uint32_t> slotIdToEvict;
  };

  void run() override;
  void step();
  void shutdownScheduler();

  void drainPendingTerminalMessages();
  void drainSchedulerResponses();
  void drainTokenOutputs();
  void drainAudioOutputs();
  void drainVoiceEncodeResults();
  void drainControlMessages();
  void drainDeferredStops();
  void drainDeferredEvicts();
  void drainTasks();

  void handleTask(ipc::tts::TtsIpcTask task);
  void handleControl(uint32_t taskId);
  void handleVoiceEncodeResult(const tts_scheduler::VoiceEncodeResult& result);
  void handleSchedulerResponse(
      const tts_scheduler::SchedulerResponse& response);
  void handleTokenOutput(const tts_scheduler::TokenOutput& output);
  void handleAudioOutput(const tts_scheduler::AudioOutput& output);

  void handleAllocateAck(const tts_scheduler::SchedulerResponse& response);
  void handleSubmitAck(const tts_scheduler::SchedulerResponse& response);
  void handleStopAck(const tts_scheduler::SchedulerResponse& response);
  void handleEvictAck(const tts_scheduler::SchedulerResponse& response);

  void compilePromptTokens(ipc::tts::TtsIpcTask& task,
                           const std::vector<uint32_t>& speechIds);
  void allocateTask(ipc::tts::TtsIpcTask task);
  bool sendFinish(uint32_t taskId, domain::tts::TtsFinishReason reason,
                  std::string error = {},
                  std::optional<uint32_t> slotIdToEvict = std::nullopt);
  void handleTerminalDelivered(const PendingTerminalMessage& terminal);
  void maybeFinalizeCompletedSlot(uint32_t slotId);
  void requestStop(uint32_t slotId);
  void requestEvict(uint32_t slotId);
  void cleanupSlot(uint32_t slotId);
  SlotContext* findSlot(uint32_t slotId);
  SlotContext* findSlotByTask(uint32_t taskId);
  bool shouldDropOutput(uint32_t slotId, const char* outputType) const;

  config::TtsConfig config;
  std::unique_ptr<tts_scheduler::ITtsScheduler> scheduler;
  ipc::tts::TtsTaskQueue* taskQueue;
  ipc::tts::TtsAudioChunkQueue* audioQueue;
  ipc::ICancelQueue* cancelQueue;
  std::vector<SlotContext> slots;
  std::unordered_map<uint32_t, ipc::tts::TtsIpcTask> pendingVoiceEncodes;
  utils::VoiceSampleCache voiceSampleCache;
  std::unordered_map<uint32_t, ipc::tts::TtsIpcTask> pendingAllocations;
  std::deque<PendingTerminalMessage> pendingTerminalMessages;
  std::vector<uint32_t> deferredStopSlots;
  std::vector<uint32_t> deferredEvictSlots;
  std::atomic<bool> stopped{false};
  std::chrono::steady_clock::time_point lastOutputTime;
  std::chrono::milliseconds outputHangTimeout;
};

}  // namespace tt::runners::blaze
