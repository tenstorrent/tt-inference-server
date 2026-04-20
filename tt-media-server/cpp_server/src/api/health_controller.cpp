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
      for (const auto& w : status.worker_info) {
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
    response["model_ready"] = status.model_ready;
    response["queue_size"] = static_cast<Json::UInt64>(status.queue_size);
    response["max_queue_size"] =
        static_cast<Json::UInt64>(status.max_queue_size);

    if (socket_) {
      response["socket_status"] = socket_->getStatus();
    }

    Json::Value workers(Json::arrayValue);
    for (const auto& w : status.worker_info) {
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

}  // namespace tt::api
