// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "domain/base_request.hpp"
#include "domain/json_field.hpp"

namespace tt::domain::tts {

/** Voice reference audio normalized by the API/service layer before scheduling.
 */
struct VoiceSample {
  std::vector<int16_t> wavPcm;
  uint32_t sampleRateHz = 0;
  uint16_t channels = 0;
};

/** Client-facing TTS request. This type intentionally carries client request
 * identity only; scheduler slot IDs are internal to the runner/scheduler layer.
 */
struct TtsRequest : tt::domain::BaseRequest {
  using tt::domain::BaseRequest::BaseRequest;

  std::string text;
  std::optional<std::string> description;
  std::optional<VoiceSample> voiceSample;

  static TtsRequest fromJson(const Json::Value& json, uint32_t taskId) {
    TtsRequest request(taskId);
    if (!json.isMember("text") || json["text"].isNull()) {
      throw std::invalid_argument("Missing required field: text");
    }
    request.text = json_field::getString(json["text"], "text");
    if (json.isMember("description") && !json["description"].isNull()) {
      request.description =
          json_field::getString(json["description"], "description");
    }
    return request;
  }
};

/** Generation knobs that are safe to pass across the service/worker boundary
 * before translating to the TTS scheduler's model-specific params.
 */
struct TtsGenerationParams {
  bool ignoreEos = false;
  std::vector<uint32_t> stopTokenIds;
};

/** Worker-boundary task. Text has already been templated/tokenized and voice
 * audio has already been validated/normalized by the service layer.
 */
struct TtsTask {
  uint32_t task_id = 0;
  std::string text;
  std::optional<std::string> description;
  std::vector<uint32_t> promptTokens;
  std::vector<int16_t> voiceWavPcm;
  TtsGenerationParams generation;
};

/** Audio produced by the decoder for one streamed chunk. */
struct TtsAudioChunk {
  uint32_t task_id = 0;
  uint32_t chunkIndex = 0;
  std::vector<uint16_t> samplesBf16;
  uint32_t sampleRateHz = 0;
  uint16_t channels = 0;
};

enum class TtsFinishReason {
  Completed,
  Cancelled,
  Error,
};

using TtsEvent = std::variant<TtsAudioChunk, TtsFinishReason>;

}  // namespace tt::domain::tts
