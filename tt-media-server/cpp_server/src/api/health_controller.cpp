// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/health_controller.hpp"

#include <chrono>

#include "config/settings.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/logger.hpp"
#include "utils/service_container.hpp"

namespace tt::api {

HealthController::HealthController() {
  auto& container = tt::utils::ServiceContainer::instance();
  service_ = container.configuredService();
  socket_ = container.socket();
  TT_LOG_INFO("[HealthController] Initialized (service={}, socket={})",
              (service_ ? "yes" : "no"), (socket_ ? "yes" : "no"));
}

void HealthController::health(
    const drogon::HttpRequestPtr&,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  Json::Value response;
  response["timestamp"] = static_cast<Json::Int64>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());

  bool hasAliveWorkers = false;
  bool hasReadyWorkers = false;
  if (service_) {
    try {
      auto status = service_->getSystemStatus();
      for (const auto& w : status.workerInfo) {
        if (w.is_alive) {
          hasAliveWorkers = true;
        }
        if (w.is_ready) {
          hasReadyWorkers = true;
        }
      }
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[HealthController] Failed to get worker status: {}",
                   e.what());
    }
  }

  bool socketHealthy = true;
  if (socket_) {
    response["socket_status"] = socket_->getStatus();
    socketHealthy = socket_->isConnected();
  }

  if (hasReadyWorkers && hasAliveWorkers && socketHealthy) {
    response["status"] = "healthy";
    callback(drogon::HttpResponse::newHttpJsonResponse(response));
  } else if (!hasAliveWorkers) {
    response["status"] = "unhealthy";
    if (!hasAliveWorkers) {
      response["error"] = "no workers are alive";
    } else {
      response["error"] = "socket not connected";
    }
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    resp->setStatusCode(drogon::k503ServiceUnavailable);
    callback(resp);
  } else if (!hasReadyWorkers) {
    response["status"] = "unhealthy";
    response["error"] = "no workers are ready";
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    resp->setStatusCode(drogon::k503ServiceUnavailable);
    callback(resp);
  } else {
    response["status"] = "unhealthy";
    response["error"] = "socket not connected";
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    resp->setStatusCode(drogon::k503ServiceUnavailable);
    callback(resp);
  }
}

void HealthController::ready(
    const drogon::HttpRequestPtr&,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  if (!service_) {
    Json::Value response;
    response["status"] = "alive";
    response["model_ready"] = false;
    response["error"] = "no service configured";
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    resp->setStatusCode(drogon::k500InternalServerError);
    callback(resp);
    return;
  }

  try {
    auto status = service_->getSystemStatus();

    Json::Value response;
    response["status"] = "alive";
    response["model_ready"] = status.modelReady;
    response["queue_size"] = static_cast<Json::UInt64>(status.queueSize);
    response["max_queue_size"] = static_cast<Json::UInt64>(status.maxQueueSize);

    if (socket_) {
      response["socket_status"] = socket_->getStatus();
    }

    Json::Value workers(Json::arrayValue);
    for (const auto& w : status.workerInfo) {
      Json::Value wj;
      wj["worker_id"] = w.worker_id;
      wj["is_ready"] = w.is_ready;
      wj["is_alive"] = w.is_alive;
      wj["pid"] = static_cast<Json::Int64>(w.pid);
      workers.append(wj);
    }
    response["workers"] = workers;

    callback(drogon::HttpResponse::newHttpJsonResponse(response));
  } catch (const std::exception& e) {
    Json::Value response;
    response["status"] = "alive";
    response["model_ready"] = false;
    response["error"] = std::string("Liveness check failed: ") + e.what();
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    resp->setStatusCode(drogon::k500InternalServerError);
    callback(resp);
  }
}

void HealthController::getMaxSessionCount(
    const drogon::HttpRequestPtr&,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  Json::Value response;
  response["max_session_count"] =
      static_cast<Json::UInt64>(tt::config::maxSessionsCount());
  callback(drogon::HttpResponse::newHttpJsonResponse(response));
}

void HealthController::setMaxSessionCount(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
  try {
    auto json = req->getJsonObject();
    if (!json) {
      Json::Value response;
      response["error"] = "Request body must be JSON";
      auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
      resp->setStatusCode(drogon::k400BadRequest);
      callback(resp);
      return;
    }

    if (!json->isMember("max_session_count")) {
      Json::Value response;
      response["error"] = "Missing required field: max_session_count";
      auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
      resp->setStatusCode(drogon::k400BadRequest);
      callback(resp);
      return;
    }

    const auto& countValue = (*json)["max_session_count"];
    if (!countValue.isUInt64() && !countValue.isInt64()) {
      Json::Value response;
      response["error"] = "max_session_count must be a non-negative integer";
      auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
      resp->setStatusCode(drogon::k400BadRequest);
      callback(resp);
      return;
    }

    size_t newCount = static_cast<size_t>(countValue.asUInt64());

    TT_LOG_INFO("[HealthController] Setting max session count to {}", newCount);
    tt::config::setMaxSessionsCount(newCount);

    Json::Value response;
    response["max_session_count"] = static_cast<Json::UInt64>(newCount);
    response["status"] = "success";
    callback(drogon::HttpResponse::newHttpJsonResponse(response));
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[HealthController] Failed to set max session count: {}",
                 e.what());
    Json::Value response;
    response["error"] =
        std::string("Failed to set max session count: ") + e.what();
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    resp->setStatusCode(drogon::k500InternalServerError);
    callback(resp);
  }
}

}  // namespace tt::api
