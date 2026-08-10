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
