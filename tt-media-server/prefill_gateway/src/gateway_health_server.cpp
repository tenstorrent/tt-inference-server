// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/gateway_health_server.hpp"

#include <utility>

namespace tt::gateway {
namespace {

bool isLivenessRequest(std::string_view request) {
  return request.starts_with("GET /tt-liveness ") ||
         request.starts_with("GET /tt-liveness?");
}

bool isHealthRequest(std::string_view request) {
  return request.starts_with("GET /health ") ||
         request.starts_with("GET /health?");
}

}  // namespace

GatewayHealthServer::GatewayHealthServer()
    : httpServer(
          "GatewayHealthServer", "Serving health endpoints",
          [this](std::string_view request) { return handleRequest(request); }) {
}

bool GatewayHealthServer::start(uint16_t port) {
  return httpServer.start(port);
}

void GatewayHealthServer::stop() { httpServer.stop(); }

uint16_t GatewayHealthServer::port() const { return httpServer.port(); }

void GatewayHealthServer::setHealthProvider(
    std::function<std::string()> provider) {
  healthProvider = std::move(provider);
}

std::optional<GatewayHttpResponse> GatewayHealthServer::handleRequest(
    std::string_view request) {
  if (!isLivenessRequest(request) && !isHealthRequest(request)) {
    return std::nullopt;
  }

  return GatewayHttpResponse{
      200, "OK", "application/json",
      healthProvider ? healthProvider() : R"({"status":"alive"})"};
}

}  // namespace tt::gateway
