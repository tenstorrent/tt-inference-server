// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/gateway_metrics_server.hpp"

#include "gateway/gateway_metrics.hpp"

namespace tt::gateway {
namespace {

bool isMetricsRequest(std::string_view request) {
  return request.starts_with("GET /metrics ") ||
         request.starts_with("GET /metrics?");
}

}  // namespace

GatewayMetricsServer::GatewayMetricsServer(GatewayMetrics& metrics)
    : gatewayMetrics(metrics),
      httpServer(
          "GatewayMetricsServer", "Serving metrics endpoint",
          [this](std::string_view request) { return handleRequest(request); }) {
}

GatewayMetricsServer::~GatewayMetricsServer() = default;

bool GatewayMetricsServer::start(uint16_t port) {
  return httpServer.start(port);
}

void GatewayMetricsServer::stop() { httpServer.stop(); }

uint16_t GatewayMetricsServer::port() const { return httpServer.port(); }

std::optional<GatewayHttpResponse> GatewayMetricsServer::handleRequest(
    std::string_view request) const {
  if (!isMetricsRequest(request)) {
    return std::nullopt;
  }

  return GatewayHttpResponse{200, "OK", "text/plain; version=0.0.4",
                             gatewayMetrics.renderText()};
}

}  // namespace tt::gateway
