// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "gateway/gateway_http_server.hpp"

namespace tt::gateway {

class GatewayMetrics;

class GatewayMetricsServer {
 public:
  explicit GatewayMetricsServer(GatewayMetrics& metrics);
  GatewayMetricsServer(const GatewayMetricsServer&) = delete;
  GatewayMetricsServer& operator=(const GatewayMetricsServer&) = delete;
  ~GatewayMetricsServer();

  bool start(uint16_t port);
  void stop();
  uint16_t port() const;

 private:
  std::optional<GatewayHttpResponse> handleRequest(
      std::string_view request) const;

  GatewayMetrics& gatewayMetrics;
  GatewayHttpServer httpServer;
};

}  // namespace tt::gateway
