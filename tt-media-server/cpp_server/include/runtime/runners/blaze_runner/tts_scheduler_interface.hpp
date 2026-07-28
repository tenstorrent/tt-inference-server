// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "domain/tts/tts_types.hpp"

namespace tt::runners::blaze::tts_scheduler {

constexpr uint32_t INVALID_SLOT = UINT32_MAX;

enum class RequestType {
  ALLOCATE,
  SUBMIT,
  CONTINUE,
  STOP,
  EVICT,
};

struct SchedulerRequest {
  RequestType type = RequestType::ALLOCATE;
  uint32_t requestId = 0;
  uint32_t taskId = 0;
  uint32_t slotId = INVALID_SLOT;
};

struct GenerationParams {
  uint32_t maxNewTokens = 0;
  bool ignoreEos = false;
  std::vector<uint32_t> stopTokens;
};

struct TtsSubmit {
  uint32_t requestId = 0;
  uint32_t taskId = 0;
  uint32_t slotId = INVALID_SLOT;
  std::vector<uint32_t> promptTokens;
  GenerationParams generation;
};

struct SchedulerResponse {
  RequestType type = RequestType::ALLOCATE;
  uint32_t requestId = 0;
  uint32_t taskId = 0;
  uint32_t slotId = INVALID_SLOT;
  int32_t errorCode = 0;
  std::string error;
};

struct TokenOutput {
  uint32_t requestId = 0;
  uint32_t taskId = 0;
  uint32_t slotId = INVALID_SLOT;
  uint32_t tokenId = 0;
  bool final = false;
};

struct AudioOutput {
  uint32_t requestId = 0;
  uint32_t taskId = 0;
  uint32_t slotId = INVALID_SLOT;
  uint32_t chunkIndex = 0;
  std::vector<uint16_t> samplesBf16;
  uint32_t sampleRateHz = 0;
  uint16_t channels = 0;
  bool final = false;
  domain::tts::TtsFinishReason finishReason =
      domain::tts::TtsFinishReason::Completed;
  std::string error;
};

struct VoiceEncodeRequest {
  uint32_t requestId = 0;
  std::vector<int16_t> wavPcm;
};

enum class VoiceEncodeStatus : uint8_t {
  Completed,
  Cancelled,
  Error,
};

struct VoiceEncodeResult {
  uint32_t requestId = 0;
  std::vector<uint32_t> speechIds;
  VoiceEncodeStatus status = VoiceEncodeStatus::Error;
};

class ITtsScheduler {
 public:
  virtual ~ITtsScheduler() = default;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual bool pushRequest(const SchedulerRequest& request) = 0;
  virtual bool submit(const TtsSubmit& request) = 0;
  virtual bool tryPopResponse(SchedulerResponse& response) = 0;
  virtual bool tryPopToken(TokenOutput& output) = 0;
  virtual bool tryPopAudio(AudioOutput& output) = 0;
  virtual bool enqueueVoiceEncode(VoiceEncodeRequest request) = 0;
  virtual bool tryPopVoiceEncodeResult(VoiceEncodeResult& result) = 0;
  virtual bool isComplete(uint32_t slotId) const = 0;
  virtual uint32_t getInFlightCount(uint32_t slotId) const = 0;
};

}  // namespace tt::runners::blaze::tts_scheduler
