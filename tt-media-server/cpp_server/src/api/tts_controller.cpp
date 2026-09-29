// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/tts_controller.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "api/error_response.hpp"
#include "api/response_writer/streaming_wav_response_writer.hpp"
#include "config/settings.hpp"
#include "services/service_container.hpp"
#include "utils/audio_codec.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

namespace tt::api {

namespace {

using tt::domain::tts::TtsEvent;
using tt::domain::tts::TtsRequest;

template <typename ParamMap>
std::optional<std::string> findParam(const ParamMap& params,
                                     const std::string& name) {
  auto it = params.find(name);
  if (it == params.end()) {
    return std::nullopt;
  }
  return it->second;
}

// speech_ids arrives in a multipart text part either as a JSON array
// ("[12, 34]") or as a comma-separated list ("12,34"). The leading bracket
// decides: jsoncpp is lenient and would otherwise accept "12,34" as the number
// 12.
std::vector<uint32_t> parseSpeechIdsParam(const std::string& raw) {
  constexpr const char* kSpace = " \t\r\n";
  const size_t first = raw.find_first_not_of(kSpace);
  if (first != std::string::npos && raw[first] == '[') {
    Json::Value parsed;
    Json::CharReaderBuilder builder;
    builder["failIfExtra"] = true;
    std::string errors;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(raw.data(), raw.data() + raw.size(), &parsed, &errors)) {
      throw std::invalid_argument("speech_ids is not a valid JSON array: " +
                                  errors);
    }
    return TtsRequest::parseSpeechIds(parsed);
  }

  Json::Value parsed(Json::arrayValue);
  std::string_view rest = raw;
  while (true) {
    const size_t comma = rest.find(',');
    std::string item(rest.substr(0, comma));
    const size_t begin = item.find_first_not_of(kSpace);
    item = begin == std::string::npos
               ? std::string{}
               : item.substr(begin, item.find_last_not_of(kSpace) - begin + 1);
    if (item.empty() ||
        item.find_first_not_of("0123456789") != std::string::npos) {
      throw std::invalid_argument(
          "speech_ids must be a JSON array or comma-separated list of "
          "non-negative integers");
    }
    parsed.append(Json::Value(static_cast<Json::UInt64>(std::stoull(item))));
    if (comma == std::string_view::npos) {
      break;
    }
    rest = rest.substr(comma + 1);
  }
  return TtsRequest::parseSpeechIds(parsed);
}

TtsRequest parseTtsRequest(const drogon::HttpRequestPtr& req, uint32_t taskId) {
  if (auto json = req->getJsonObject()) {
    return TtsRequest::fromJson(*json, taskId);
  }

  drogon::MultiPartParser parser;
  if (parser.parse(req) != 0) {
    throw std::invalid_argument("Request must be JSON or multipart/form-data");
  }

  const auto& params = parser.getParameters();
  auto text = findParam(params, "text");
  if (!text.has_value()) {
    throw std::invalid_argument("Missing required field: text");
  }

  TtsRequest request(taskId);
  request.text = *text;
  request.description = findParam(params, "description");

  if (auto speechIds = findParam(params, "speech_ids")) {
    request.promptSpeechIds = parseSpeechIdsParam(*speechIds);
  }

  const auto& files = parser.getFiles();
  if (!files.empty()) {
    const auto& file = files.front();
    const auto& content = file.fileContent();
    request.voiceSample = tt::utils::audio_codec::decodePcm16Wav(
        std::string_view(content.data(), content.size()));
  }
  return request;
}

void handleTtsStreaming(
    const std::shared_ptr<tt::services::TtsService>& service,
    TtsRequest request,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
  const uint32_t taskId = request.task_id;
  auto servicePtr = service;

  auto writer = StreamingWavResponseWriter::create(
      trantor::EventLoop::getEventLoopOfCurrentThread(),
      {.task_id = taskId,
       .sampleRateHz = servicePtr->outputSampleRateHz(),
       .channels = servicePtr->outputChannels(),
       .onCancelRequest = [servicePtr](uint32_t id) {
         servicePtr->cancel(id);
       }});

  auto onEvent = [writer](const TtsEvent& event) {
    writer->handleEvent(event);
  };

  bool accepted = false;
  try {
    accepted = servicePtr->generate(std::move(request), std::move(onEvent));
  } catch (const tt::services::QueueFullException& e) {
    callback(errorResponse(drogon::k429TooManyRequests, e.what(),
                           "rate_limit_exceeded"));
    return;
  } catch (const std::exception& e) {
    callback(errorResponse(drogon::k400BadRequest, e.what(),
                           "invalid_request_error"));
    return;
  }

  if (!accepted) {
    callback(errorResponse(drogon::k503ServiceUnavailable,
                           "TTS generation backend is not available yet",
                           "service_unavailable"));
    return;
  }

  callback(writer->buildResponse());
}

}  // namespace

TtsController::TtsController() {
  if (!tt::config::isTtsService()) {
    return;
  }

  service = std::dynamic_pointer_cast<tt::services::TtsService>(
      tt::services::ServiceContainer::instance().getService(
          tt::config::ModelService::TTS));
  if (!service) {
    throw std::runtime_error(
        "[TtsController] TTS service not found in container. "
        "Ensure initializeServices() is called before Drogon starts.");
  }
  TT_LOG_INFO("[TtsController] Initialized");
}

void TtsController::speech(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
  if (!service) {
    callback(errorResponse(drogon::k503ServiceUnavailable,
                           "TTS service is not configured",
                           "service_unavailable"));
    return;
  }

  try {
    const auto taskId =
        static_cast<uint32_t>(tt::utils::TaskIDGenerator::generate());
    auto request = parseTtsRequest(req, taskId);
    handleTtsStreaming(service, std::move(request), std::move(callback));
  } catch (const std::exception& e) {
    callback(errorResponse(drogon::k400BadRequest, e.what(),
                           "invalid_request_error"));
  }
}

}  // namespace tt::api
