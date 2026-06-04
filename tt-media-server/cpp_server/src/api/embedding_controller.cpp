// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "api/embedding_controller.hpp"

#include <chrono>
#include <optional>
#include <string>

#include "api/error_response.hpp"
#include "config/settings.hpp"
#include "services/base_service.hpp"
#include "services/service_container.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"
#include "utils/thread_pool.hpp"

namespace tt::api {

EmbeddingController::EmbeddingController() {
  if (!tt::config::isEmbeddingService()) {
    return;
  }

  service_ = std::dynamic_pointer_cast<tt::services::EmbeddingService>(
      tt::services::ServiceContainer::instance().getService(
          tt::config::ModelService::EMBEDDING));
  if (!service_) {
    throw std::runtime_error(
        "[EmbeddingController] Embedding service not found in container. "
        "Ensure initializeServices() is called before Drogon starts.");
  }
  TT_LOG_INFO("[EmbeddingController] Initialized (service already started)");
}

EmbeddingController::~EmbeddingController() = default;

void EmbeddingController::createEmbedding(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
  auto startTime = std::chrono::steady_clock::now();

  // Resolved up front so every response on this request - including early
  // validation failures - echoes the same `X-Request-Id` back to the client.
  const std::string traceId =
      tt::utils::TraceIdGenerator::resolveOrGenerate(
          req->getHeader("x-request-id"));

  auto json = req->getJsonObject();
  if (!json) {
    callback(withRequestId(errorResponse(drogon::k400BadRequest,
                                         "Invalid JSON body",
                                         "invalid_request_error"),
                           traceId));
    return;
  }

  if (!json->isMember("input")) {
    callback(withRequestId(errorResponse(drogon::k400BadRequest,
                                         "Missing required field: input",
                                         "invalid_request_error"),
                           traceId));
    return;
  }

  uint32_t taskId = tt::utils::TaskIDGenerator::generate();
  std::optional<domain::EmbeddingRequest> requestOpt;
  try {
    requestOpt = domain::EmbeddingRequest::fromJson(*json, std::move(taskId));
  } catch (const std::invalid_argument& e) {
    callback(withRequestId(errorResponse(drogon::k400BadRequest, e.what(),
                                         "invalid_request_error"),
                           traceId));
    return;
  }
  auto request = std::move(*requestOpt);
  request.trace_id = traceId;

  if (request.model.empty()) {
    request.model = "BAAI/bge-large-en-v1.5";
  }

  uint64_t reqNum = request_counter_.fetch_add(1);

  auto submitTime = std::chrono::steady_clock::now();

  tt::utils::controllerCallbackPool().submit([service = service_,
                                              request = std::move(request),
                                              callback = std::move(callback),
                                              reqNum, startTime, submitTime]() {
    try {
      auto response = service->submitRequest(std::move(request));
      auto gotResponseTime = std::chrono::steady_clock::now();

      if (!response.error.empty()) {
        callback(withRequestId(
            errorResponse(drogon::k500InternalServerError, response.error,
                          "server_error"),
            request.trace_id));
        return;
      }

      Json::Value jsonResponse = response.toOpenaiJson();
      auto builtJsonTime = std::chrono::steady_clock::now();

      auto resp = drogon::HttpResponse::newHttpJsonResponse(jsonResponse);
      if (!request.trace_id.empty()) {
        resp->addHeader("X-Request-Id", request.trace_id);
      }

      if (reqNum % 100 == 0) {
        double parseMs =
            std::chrono::duration<double, std::milli>(submitTime - startTime)
                .count();
        double waitMs = std::chrono::duration<double, std::milli>(
                            gotResponseTime - submitTime)
                            .count();
        double buildMs = std::chrono::duration<double, std::milli>(
                             builtJsonTime - gotResponseTime)
                             .count();
        double totalMs =
            std::chrono::duration<double, std::milli>(builtJsonTime - startTime)
                .count();
        TT_LOG_DEBUG(
            "[EmbeddingController] req={} parse={}ms wait={}ms build={}ms "
            "total={}ms",
            reqNum, parseMs, waitMs, buildMs, totalMs);
      }

      callback(resp);

    } catch (const services::QueueFullException& e) {
      callback(withRequestId(errorResponse(drogon::k429TooManyRequests,
                                           e.what(), "rate_limit_exceeded"),
                             request.trace_id));
    } catch (const std::exception& e) {
      callback(withRequestId(
          errorResponse(drogon::k500InternalServerError,
                        std::string("Internal error: ") + e.what(),
                        "server_error"),
          request.trace_id));
    }
  });
}

}  // namespace tt::api
