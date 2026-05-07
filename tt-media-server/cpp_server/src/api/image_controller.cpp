// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/image_controller.hpp"

#include <chrono>
#include <optional>

#include "api/error_response.hpp"
#include "config/defaults.hpp"
#include "config/settings.hpp"
#include "services/service_container.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"
#include "utils/thread_pool.hpp"

namespace tt::api {

namespace {

tt::utils::ThreadPool& callbackPool() {
  static tt::utils::ThreadPool pool(
      tt::config::defaults::CALLBACK_POOL_THREADS);
  return pool;
}

// Endpoint kinds drive request validation. The runtime keeps using a single
// ImageGenerateRequest carrier struct for all three modes (mirroring the
// Python class hierarchy that overloads ImageToImageRequest from
// ImageGenerateRequest), but each kind has its own required-field set.
enum class EndpointKind { GENERATE, IMAGE_TO_IMAGE, EDIT };

void handle(const drogon::HttpRequestPtr& req,
            std::function<void(const drogon::HttpResponsePtr&)> callback,
            std::shared_ptr<tt::services::ImageService> service,
            std::atomic<uint64_t>& counter, const char* endpointTag,
            EndpointKind kind) {
  auto startTime = std::chrono::steady_clock::now();

  auto json = req->getJsonObject();
  if (!json) {
    callback(errorResponse(drogon::k400BadRequest, "Invalid JSON body",
                           "invalid_request_error"));
    return;
  }
  if (!json->isMember("prompt")) {
    callback(errorResponse(drogon::k400BadRequest,
                           "Missing required field: prompt",
                           "invalid_request_error"));
    return;
  }
  if ((kind == EndpointKind::IMAGE_TO_IMAGE || kind == EndpointKind::EDIT) &&
      !json->isMember("image")) {
    callback(errorResponse(drogon::k400BadRequest,
                           "Missing required field: image",
                           "invalid_request_error"));
    return;
  }
  if (kind == EndpointKind::EDIT && !json->isMember("mask")) {
    callback(errorResponse(drogon::k400BadRequest,
                           "Missing required field: mask",
                           "invalid_request_error"));
    return;
  }

  if (!service->isModelReady()) {
    callback(errorResponse(
        drogon::k503ServiceUnavailable,
        "Model is still warming up. Poll /tt-liveness for model_ready=true.",
        "service_unavailable"));
    return;
  }

  uint32_t taskId = tt::utils::TaskIDGenerator::generate();
  std::optional<tt::domain::ImageGenerateRequest> requestOpt;
  try {
    requestOpt = tt::domain::ImageGenerateRequest::fromJson(*json, taskId);
  } catch (const std::exception& e) {
    // fromJson throws std::invalid_argument from json_field helpers, but
    // any std::exception (including Json::Exception from a malformed
    // value JsonCpp couldn't reject earlier) is a client-side payload bug
    // and should map to 400, not 500.
    callback(errorResponse(drogon::k400BadRequest, e.what(),
                           "invalid_request_error"));
    return;
  }
  auto request = std::move(*requestOpt);
  uint64_t reqNum = counter.fetch_add(1);
  TT_LOG_INFO("[ImageController] {} req={} started", endpointTag, reqNum);

  callbackPool().submit([service, request = std::move(request),
                         callback = std::move(callback), reqNum, startTime,
                         endpointTag]() {
    try {
      auto response = service->submitRequest(std::move(request));
      double totalMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - startTime)
                           .count();
      if (!response.error.empty()) {
        TT_LOG_ERROR("[ImageController] {} req={} failed in {}ms: {}",
                     endpointTag, reqNum, totalMs, response.error);
        callback(errorResponse(drogon::k500InternalServerError, response.error,
                               "server_error"));
        return;
      }
      Json::Value jsonResponse = response.toOpenaiJson();
      auto resp = drogon::HttpResponse::newHttpJsonResponse(jsonResponse);
      TT_LOG_INFO("[ImageController] {} req={} done in {}ms", endpointTag,
                  reqNum, totalMs);
      callback(resp);
    } catch (const tt::services::QueueFullException& e) {
      callback(errorResponse(drogon::k429TooManyRequests, e.what(),
                             "rate_limit_exceeded"));
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[ImageController] {} req={} threw: {}", endpointTag, reqNum,
                   e.what());
      callback(errorResponse(drogon::k500InternalServerError,
                             std::string("Internal error: ") + e.what(),
                             "server_error"));
    }
  });
}

}  // namespace

ImageController::ImageController() {
  if (!tt::config::isImageService()) {
    return;
  }
  service_ = std::dynamic_pointer_cast<tt::services::ImageService>(
      tt::services::ServiceContainer::instance().getService(
          tt::config::ModelService::IMAGE));
  if (!service_) {
    throw std::runtime_error(
        "[ImageController] Image service not found in container. "
        "Ensure initializeServices() is called before Drogon starts.");
  }
  TT_LOG_INFO("[ImageController] Initialized");
}

ImageController::~ImageController() = default;

void ImageController::generate(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
  handle(req, std::move(callback), service_, request_counter_, "generations",
         EndpointKind::GENERATE);
}

void ImageController::imageToImage(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
  handle(req, std::move(callback), service_, request_counter_, "image-to-image",
         EndpointKind::IMAGE_TO_IMAGE);
}

void ImageController::edit(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
  handle(req, std::move(callback), service_, request_counter_, "edits",
         EndpointKind::EDIT);
}

}  // namespace tt::api
