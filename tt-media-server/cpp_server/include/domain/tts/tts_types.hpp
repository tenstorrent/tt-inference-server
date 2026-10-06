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
  // Optional voice controls. Which ones a deployment accepts depends on the
  // model it serves (see qwen3_tts_request_validator.hpp); TTS-2 ignores them.
  // `speaker` names a built-in voice, `language` the language to speak, and
  // `referenceText` is the transcript of the voice sample.
  std::optional<std::string> speaker;
  std::optional<std::string> language;
  std::optional<std::string> referenceText;

  static TtsRequest fromJson(const Json::Value& json, uint32_t taskId) {
    TtsRequest request(taskId);
    if (!json.isMember("text") || json["text"].isNull()) {
      throw std::invalid_argument("Missing required field: text");
    }
    request.text = json_field::getString(json["text"], "text");
    request.description = optionalString(json, "description");
    request.speaker = optionalString(json, "speaker");
    request.language = optionalString(json, "language");
    request.referenceText = optionalString(json, "reference_text");
    return request;
  }

  /** The text fields of a multipart/form-data request (the voice sample is a
   *  file part, decoded by the controller). `ParamMap` is any string->string
   *  map with find()/end(), e.g. drogon's MultiPartParser::getParameters(). */
  template <typename ParamMap>
  static TtsRequest fromFormFields(const ParamMap& params, uint32_t taskId) {
    auto field = [&params](const char* name) -> std::optional<std::string> {
      auto it = params.find(name);
      if (it == params.end()) {
        return std::nullopt;
      }
      return it->second;
    };
    auto text = field("text");
    if (!text.has_value()) {
      throw std::invalid_argument("Missing required field: text");
    }
    TtsRequest request(taskId);
    request.text = *text;
    request.description = field("description");
    request.speaker = field("speaker");
    request.language = field("language");
    request.referenceText = field("reference_text");
    return request;
  }

 private:
  static std::optional<std::string> optionalString(const Json::Value& json,
                                                   const char* field) {
    if (!json.isMember(field) || json[field].isNull()) {
      return std::nullopt;
    }
    return json_field::getString(json[field], field);
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
  // Raw voice controls for runners that template their own prompt (Qwen3-TTS);
  // empty for TTS-2, which gets promptTokens instead.
  std::optional<std::string> speaker;
  std::optional<std::string> language;
  std::optional<std::string> referenceText;
};

/** Audio produced by the decoder for one streamed chunk. Exactly one of the
 *  two sample vectors is filled: BF16 bit patterns (TTS-2) or PCM16 samples
 *  (Qwen3-TTS), which go to the client as they are. */
struct TtsAudioChunk {
  uint32_t task_id = 0;
  uint32_t chunkIndex = 0;
  std::vector<uint16_t> samplesBf16;
  uint32_t sampleRateHz = 0;
  uint16_t channels = 0;
  std::vector<int16_t> samplesPcm16;
};

enum class TtsFinishReason {
  Completed,
  Cancelled,
  Error,
};

using TtsEvent = std::variant<TtsAudioChunk, TtsFinishReason>;

}  // namespace tt::domain::tts
