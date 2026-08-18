// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "gateway/gateway_health.hpp"
#include "gateway/gateway_http_server.hpp"

namespace tt::gateway {

class GatewayHealthServer {
 public:
  GatewayHealthServer();
  GatewayHealthServer(const GatewayHealthServer&) = delete;
  GatewayHealthServer& operator=(const GatewayHealthServer&) = delete;
  ~GatewayHealthServer() = default;

  bool start(uint16_t port);
  void stop();
  uint16_t port() const;
  void setHealthProvider(std::function<GatewayHealthStatus()> provider);

 private:
  std::optional<GatewayHttpResponse> handleRequest(std::string_view request);

  std::function<GatewayHealthStatus()> healthProvider;
  GatewayHttpServer httpServer;
};

}  // namespace tt::gateway
