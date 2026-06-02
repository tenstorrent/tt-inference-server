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

  GatewayMetrics& metrics_;
  std::function<std::string()> health_provider_;
  std::atomic<bool> running_{false};
  std::atomic<uint16_t> port_{0};
  std::atomic<int> server_fd_{-1};
  std::jthread server_thread_;
};

}  // namespace tt::gateway
