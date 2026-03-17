// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "api/health_controller.hpp"
#include "config/settings.hpp"
#include "utils/service_factory.hpp"
#include "utils/logger.hpp"

#include <chrono>
#include "services/llm_service.hpp"
#include "services/embedding_service.hpp"

namespace tt::api {

HealthController::HealthController() {
    service_ = tt::utils::service_factory::get_configured_service();
    TT_LOG_INFO("[HealthController] Initialized (service={})", (service_ ? "yes" : "no"));
}

void HealthController::health(
    const drogon::HttpRequestPtr&,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {

    Json::Value response;
    response["status"] = "healthy";
    response["timestamp"] = static_cast<Json::Int64>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );

    callback(drogon::HttpResponse::newHttpJsonResponse(response));
}

void HealthController::ready(
    const drogon::HttpRequestPtr&,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {

    if (!service_) {
        Json::Value response;
        response["model_ready"] = false;
        response["error"] = "no service configured";
        auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
        resp->setStatusCode(drogon::k503ServiceUnavailable);
        callback(resp);
        return;
    }

    auto status = service_->get_system_status();

    Json::Value response;
    response["model_ready"] = status.model_ready;
    response["queue_size"] = static_cast<Json::UInt64>(status.queue_size);
    response["max_queue_size"] = static_cast<Json::UInt64>(status.max_queue_size);

    Json::Value workers(Json::arrayValue);
    for (const auto& w : status.worker_info) {
        Json::Value wj;
        wj["worker_id"] = w.worker_id;
        wj["is_ready"] = w.is_ready;
        wj["processed_requests"] = static_cast<Json::UInt64>(w.processed_requests);
        workers.append(wj);
    }
    response["workers"] = workers;

    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    if (!status.model_ready) {
        resp->setStatusCode(drogon::k503ServiceUnavailable);
    }
    callback(resp);
}

} // namespace tt::api
