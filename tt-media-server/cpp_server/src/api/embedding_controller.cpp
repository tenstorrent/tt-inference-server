// SPDX-License-Identifier: Apache-2.0
#include "utils/id_generator.hpp"
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include <chrono>
#include <optional>

#include "api/embedding_controller.hpp"
#include "api/error_response.hpp"
#include "config/defaults.hpp"
#include "config/settings.hpp"
#include "services/base_service.hpp"
#include "utils/logger.hpp"
#include "utils/service_container.hpp"
#include "utils/thread_pool.hpp"

namespace tt::api {

namespace {
tt::utils::ThreadPool& getCallbackPool() {
  static tt::utils::ThreadPool pool(
      tt::config::defaults::CALLBACK_POOL_THREADS);
  return pool;
}
}  // namespace

EmbeddingController::EmbeddingController() {
  if (!tt::config::isEmbeddingService()) {
    return;
  }

  service_ = tt::utils::ServiceContainer::instance().embedding();
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

  // Parse request body
  auto json = req->getJsonObject();
  if (!json) {
    callback(errorResponse(drogon::k400BadRequest, "Invalid JSON body",
                           "invalid_request_error"));
    return;
  }

  if (!json->isMember("input")) {
    callback(errorResponse(drogon::k400BadRequest,
                           "Missing required field: input",
                           "invalid_request_error"));
    return;
  }

  uint32_t taskId = tt::utils::TaskIDGenerator::generate();
  std::optional<domain::EmbeddingRequest> requestOpt;
  try {
    requestOpt = domain::EmbeddingRequest::fromJson(*json, std::move(taskId));
  } catch (const std::invalid_argument& e) {
    callback(errorResponse(drogon::k400BadRequest, e.what(),
                           "invalid_request_error"));
    return;
  }
  auto request = std::move(*requestOpt);

  // Default model if not specified
  if (request.model.empty()) {
    request.model = "BAAI/bge-large-en-v1.5";
  }

  uint64_t reqNum = request_counter_.fetch_add(1);

  auto submitTime = std::chrono::steady_clock::now();

  getCallbackPool().submit([service = service_, request = std::move(request),
                            callback = std::move(callback), reqNum, startTime,
                            submitTime]() {
    try {
      auto response = service->submitRequest(std::move(request));
      auto gotResponseTime = std::chrono::steady_clock::now();

      if (!response.error.empty()) {
        callback(errorResponse(drogon::k500InternalServerError, response.error,
                               "server_error"));
        return;
      }

      Json::Value jsonResponse = response.toOpenaiJson();
      auto builtJsonTime = std::chrono::steady_clock::now();

      auto resp = drogon::HttpResponse::newHttpJsonResponse(jsonResponse);

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
      callback(errorResponse(drogon::k429TooManyRequests, e.what(),
                             "rate_limit_exceeded"));
    } catch (const std::exception& e) {
      callback(errorResponse(drogon::k500InternalServerError,
                             std::string("Internal error: ") + e.what(),
                             "server_error"));
    }
  });
}

}  // namespace tt::api
