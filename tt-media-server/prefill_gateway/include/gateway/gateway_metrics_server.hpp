// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <string>
#include <thread>

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
  void setHealthProvider(std::function<std::string()> provider);

 private:
  void serve(std::stop_token stopToken, uint16_t port,
             std::promise<bool> initialized);
  void serveClient(int clientFd);
  static void closeFd(int fd);

  GatewayMetrics& gatewayMetrics;
  std::function<std::string()> healthProvider;
  std::atomic<bool> running{false};
  std::atomic<uint16_t> listeningPort{0};
  std::atomic<int> serverFd{-1};
  std::jthread serverThread;
};

}  // namespace tt::gateway
