// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>

namespace tt::api {

/**
 * GET /metrics
 *
 * Exposes the Prometheus text-format metrics scrape endpoint.
 * No authentication required (listed in the security filter bypass list).
 */
class MetricsController : public drogon::HttpController<MetricsController> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(MetricsController::metrics, "/metrics", drogon::Get);
  METHOD_LIST_END

  void metrics(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;
};

}  // namespace tt::api
