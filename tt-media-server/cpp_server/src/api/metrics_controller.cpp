// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "api/metrics_controller.hpp"

#include "metrics/metrics.hpp"

namespace tt::api {

void MetricsController::metrics(
    const drogon::HttpRequestPtr& /*req*/,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  std::string body = tt::metrics::ServerMetrics::instance().renderText();

  auto resp = drogon::HttpResponse::newHttpResponse();
  // Prometheus text format content-type (version 0.0.4)
  resp->setContentTypeString("text/plain; version=0.0.4; charset=utf-8");
  resp->setBody(std::move(body));
  resp->setStatusCode(drogon::k200OK);
  callback(resp);
}

}  // namespace tt::api
