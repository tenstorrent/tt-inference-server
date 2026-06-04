// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace tt::gateway {

struct GatewayHttpResponse {
  int status;
  std::string statusText;
  std::string contentType;
  std::string body;
};

class GatewayHttpServer {
 public:
  using RequestHandler =
      std::function<std::optional<GatewayHttpResponse>(std::string_view)>;

  GatewayHttpServer(std::string logName, std::string readyMessage,
                    RequestHandler requestHandler);
  GatewayHttpServer(const GatewayHttpServer&) = delete;
  GatewayHttpServer& operator=(const GatewayHttpServer&) = delete;
  ~GatewayHttpServer();

  bool start(uint16_t port);
  void stop();
  uint16_t port() const;

 private:
  void serve(std::stop_token stopToken, uint16_t port,
             std::promise<bool> initialized);
  void serveClient(int clientFd);
  static std::string formatHttpResponse(const GatewayHttpResponse& response);
  static void closeFd(int fd);

  std::string logName;
  std::string readyMessage;
  RequestHandler requestHandler;
  std::atomic<bool> running{false};
  std::atomic<uint16_t> listeningPort{0};
  std::atomic<int> serverFd{-1};
  std::jthread serverThread;
};

}  // namespace tt::gateway
