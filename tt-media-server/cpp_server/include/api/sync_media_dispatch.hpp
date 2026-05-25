// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>
#include <json/json.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "api/error_response.hpp"
#include "services/service.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"
#include "utils/thread_pool.hpp"

namespace tt::api {

/**
 * Validate the JSON body, parse the request, and dispatch
 * `service->submitRequest` on the shared callback pool. Maps parse errors to
 * 400, queue full to 429, runner exceptions to 500, model-not-ready to 503.
 * `Request` must expose `static Request fromJson(const Json::Value&, uint32_t)`
 * and `Response` must expose `error` plus `toOpenaiJson()`.
 */
template <typename Service, typename Request>
void dispatchJsonRequest(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)> callback,
    std::shared_ptr<Service> service, std::atomic<uint64_t>& counter,
    const char* logTag, const char* endpointTag,
    std::initializer_list<const char*> requiredFields,
    std::function<void(Request&)> postParse = {}) {
  const auto startTime = std::chrono::steady_clock::now();

  auto json = req->getJsonObject();
  if (!json) {
    callback(errorResponse(drogon::k400BadRequest, "Invalid JSON body",
                           "invalid_request_error"));
    return;
  }
  for (const char* field : requiredFields) {
    if (!json->isMember(field) || (*json)[field].isNull()) {
      callback(errorResponse(drogon::k400BadRequest,
                             std::string("Missing required field: ") + field,
                             "invalid_request_error"));
      return;
    }
  }

  if (!service->isModelReady()) {
    callback(errorResponse(
        drogon::k503ServiceUnavailable,
        "Model is still warming up. Poll /tt-liveness for model_ready=true.",
        "service_unavailable"));
    return;
  }

  const uint32_t taskId = tt::utils::TaskIDGenerator::generate();
  std::optional<Request> requestOpt;
  try {
    requestOpt = Request::fromJson(*json, taskId);
  } catch (const std::exception& e) {
    callback(errorResponse(drogon::k400BadRequest, e.what(),
                           "invalid_request_error"));
    return;
  }
  Request request = std::move(*requestOpt);
  if (postParse) {
    postParse(request);
  }

  const uint64_t reqNum = counter.fetch_add(1);
  TT_LOG_INFO("[{}] {} req={} started", logTag, endpointTag, reqNum);

  tt::utils::controllerCallbackPool().submit(
      [service, request = std::move(request), callback = std::move(callback),
       reqNum, startTime, logTag, endpointTag]() mutable {
        try {
          auto response = service->submitRequest(std::move(request));
          const double totalMs =
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - startTime)
                  .count();
          if (!response.error.empty()) {
            TT_LOG_ERROR("[{}] {} req={} failed in {}ms: {}", logTag,
                         endpointTag, reqNum, totalMs, response.error);
            callback(errorResponse(drogon::k500InternalServerError,
                                   response.error, "server_error"));
            return;
          }
          auto resp = drogon::HttpResponse::newHttpJsonResponse(
              response.toOpenaiJson());
          TT_LOG_INFO("[{}] {} req={} done in {}ms", logTag, endpointTag,
                      reqNum, totalMs);
          callback(resp);
        } catch (const tt::services::QueueFullException& e) {
          callback(errorResponse(drogon::k429TooManyRequests, e.what(),
                                 "rate_limit_exceeded"));
        } catch (const std::exception& e) {
          TT_LOG_ERROR("[{}] {} req={} threw: {}", logTag, endpointTag, reqNum,
                       e.what());
          callback(errorResponse(drogon::k500InternalServerError,
                                 std::string("Internal error: ") + e.what(),
                                 "server_error"));
        }
      });
}

}  // namespace tt::api
