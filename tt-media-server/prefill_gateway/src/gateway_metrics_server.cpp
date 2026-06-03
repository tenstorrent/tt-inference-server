// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/gateway_metrics_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <utility>

#include "gateway/gateway_metrics.hpp"
#include "utils/logger.hpp"

namespace tt::gateway {
namespace {

constexpr int SOCKET_BACKLOG = 16;
constexpr size_t REQUEST_BUFFER_SIZE = 2048;

std::string httpResponse(int status, std::string_view statusText,
                         std::string_view contentType, std::string_view body) {
  std::ostringstream out;
  out << "HTTP/1.1 " << status << " " << statusText << "\r\n"
      << "Content-Type: " << contentType << "\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n\r\n"
      << body;
  return out.str();
}

bool isMetricsRequest(std::string_view request) {
  return request.starts_with("GET /metrics ") ||
         request.starts_with("GET /metrics?");
}

bool isLivenessRequest(std::string_view request) {
  return request.starts_with("GET /tt-liveness ") ||
         request.starts_with("GET /tt-liveness?");
}

bool isHealthRequest(std::string_view request) {
  return request.starts_with("GET /health ") ||
         request.starts_with("GET /health?");
}

}  // namespace

GatewayMetricsServer::GatewayMetricsServer(GatewayMetrics& metrics)
    : gatewayMetrics(metrics) {}

GatewayMetricsServer::~GatewayMetricsServer() { stop(); }

bool GatewayMetricsServer::start(uint16_t port) {
  if (port == 0) {
    return true;
  }
  if (running) {
    return false;
  }

  running = true;
  listeningPort = port;
  std::promise<bool> initialized;
  auto initializedFuture = initialized.get_future();
  serverThread =
      std::jthread([this, port, initialized = std::move(initialized)](
                       std::stop_token stopToken) mutable {
        serve(stopToken, port, std::move(initialized));
      });
  const bool initializedOk = initializedFuture.get();
  if (!initializedOk) {
    stop();
  }
  return initializedOk;
}

void GatewayMetricsServer::stop() {
  if (!running) {
    return;
  }

  running = false;
  const int fd = serverFd.exchange(-1);
  if (fd >= 0) {
    shutdown(fd, SHUT_RDWR);
    closeFd(fd);
  }
  if (serverThread.joinable()) {
    serverThread.request_stop();
    serverThread.join();
  }
  listeningPort = 0;
}

uint16_t GatewayMetricsServer::port() const { return listeningPort; }

void GatewayMetricsServer::setHealthProvider(
    std::function<std::string()> provider) {
  healthProvider = std::move(provider);
}

void GatewayMetricsServer::serve(std::stop_token stopToken, uint16_t port,
                                 std::promise<bool> initialized) {
  const int serverFd = socket(AF_INET, SOCK_STREAM, 0);
  this->serverFd = serverFd;
  if (serverFd < 0) {
    TT_LOG_ERROR("[GatewayMetricsServer] socket() failed: {}", strerror(errno));
    running = false;
    initialized.set_value(false);
    return;
  }

  int reuse = 1;
  setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(port);

  if (bind(serverFd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) <
      0) {
    TT_LOG_ERROR("[GatewayMetricsServer] bind(:{}) failed: {}", port,
                 strerror(errno));
    closeFd(this->serverFd.exchange(-1));
    running = false;
    initialized.set_value(false);
    return;
  }

  if (listen(serverFd, SOCKET_BACKLOG) < 0) {
    TT_LOG_ERROR("[GatewayMetricsServer] listen(:{}) failed: {}", port,
                 strerror(errno));
    closeFd(this->serverFd.exchange(-1));
    running = false;
    initialized.set_value(false);
    return;
  }

  TT_LOG_INFO("[GatewayMetricsServer] Serving HTTP endpoints on port {}", port);
  initialized.set_value(true);

  while (running && !stopToken.stop_requested()) {
    sockaddr_in clientAddress{};
    socklen_t clientLength = sizeof(clientAddress);
    const int activeServerFd = this->serverFd.load();
    if (activeServerFd < 0) {
      break;
    }
    const int clientFd =
        accept(activeServerFd, reinterpret_cast<sockaddr*>(&clientAddress),
               &clientLength);
    if (clientFd < 0) {
      if (running && errno != EINTR && errno != EBADF && errno != EINVAL) {
        TT_LOG_WARN("[GatewayMetricsServer] accept() failed: {}",
                    strerror(errno));
      }
      continue;
    }
    serveClient(clientFd);
  }
}

void GatewayMetricsServer::serveClient(int clientFd) {
  std::array<char, REQUEST_BUFFER_SIZE> buffer{};
  const ssize_t bytesRead = recv(clientFd, buffer.data(), buffer.size() - 1, 0);
  if (bytesRead <= 0) {
    closeFd(clientFd);
    return;
  }

  const std::string_view request(buffer.data(), static_cast<size_t>(bytesRead));
  std::string response;
  if (isMetricsRequest(request)) {
    response = httpResponse(200, "OK", "text/plain; version=0.0.4",
                            gatewayMetrics.renderText());
  } else if (isLivenessRequest(request) || isHealthRequest(request)) {
    const std::string body =
        healthProvider ? healthProvider() : R"({"status":"alive"})";
    response = httpResponse(200, "OK", "application/json", body);
  } else {
    response = httpResponse(404, "Not Found", "text/plain", "not found\n");
  }

  ssize_t offset = 0;
  while (offset < static_cast<ssize_t>(response.size())) {
    const ssize_t sent =
        send(clientFd, response.data() + offset, response.size() - offset, 0);
    if (sent <= 0) {
      break;
    }
    offset += sent;
  }
  closeFd(clientFd);
}

void GatewayMetricsServer::closeFd(int fd) {
  if (fd >= 0) {
    close(fd);
  }
}

}  // namespace tt::gateway
