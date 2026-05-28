// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/gateway_metrics_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <sstream>

#include "gateway/gateway_metrics.hpp"
#include "utils/logger.hpp"

namespace tt::gateway {
namespace {

constexpr int SOCKET_BACKLOG = 16;
constexpr size_t REQUEST_BUFFER_SIZE = 2048;

std::string httpResponse(int status, const std::string& statusText,
                         const std::string& contentType,
                         const std::string& body) {
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

}  // namespace

GatewayMetricsServer::GatewayMetricsServer(GatewayMetrics& metrics)
    : metrics_(metrics) {}

GatewayMetricsServer::~GatewayMetricsServer() { stop(); }

bool GatewayMetricsServer::start(uint16_t port) {
  if (port == 0) {
    return true;
  }
  if (running_) {
    return false;
  }

  running_ = true;
  port_ = port;
  std::promise<bool> initialized;
  auto initializedFuture = initialized.get_future();
  server_thread_ =
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
  if (!running_) {
    return;
  }

  running_ = false;
  if (server_fd_ >= 0) {
    shutdown(server_fd_, SHUT_RDWR);
    closeFd(server_fd_);
    server_fd_ = -1;
  }
  if (server_thread_.joinable()) {
    server_thread_.request_stop();
    server_thread_.join();
  }
  port_ = 0;
}

uint16_t GatewayMetricsServer::port() const { return port_; }

void GatewayMetricsServer::serve(std::stop_token stopToken, uint16_t port,
                                 std::promise<bool> initialized) {
  server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    TT_LOG_ERROR("[GatewayMetricsServer] socket() failed: {}", strerror(errno));
    running_ = false;
    initialized.set_value(false);
    return;
  }

  int reuse = 1;
  setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(port);

  if (bind(server_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) <
      0) {
    TT_LOG_ERROR("[GatewayMetricsServer] bind(:{}) failed: {}", port,
                 strerror(errno));
    closeFd(server_fd_);
    server_fd_ = -1;
    running_ = false;
    initialized.set_value(false);
    return;
  }

  if (listen(server_fd_, SOCKET_BACKLOG) < 0) {
    TT_LOG_ERROR("[GatewayMetricsServer] listen(:{}) failed: {}", port,
                 strerror(errno));
    closeFd(server_fd_);
    server_fd_ = -1;
    running_ = false;
    initialized.set_value(false);
    return;
  }

  TT_LOG_INFO("[GatewayMetricsServer] Serving /metrics on port {}", port);
  initialized.set_value(true);

  while (running_ && !stopToken.stop_requested()) {
    sockaddr_in clientAddress{};
    socklen_t clientLength = sizeof(clientAddress);
    const int clientFd = accept(
        server_fd_, reinterpret_cast<sockaddr*>(&clientAddress), &clientLength);
    if (clientFd < 0) {
      if (running_ && errno != EINTR && errno != EBADF && errno != EINVAL) {
        TT_LOG_WARN("[GatewayMetricsServer] accept() failed: {}",
                    strerror(errno));
      }
      continue;
    }
    serveClient(clientFd);
  }
}

void GatewayMetricsServer::serveClient(int clientFd) {
  char buffer[REQUEST_BUFFER_SIZE] = {};
  const ssize_t bytesRead = recv(clientFd, buffer, sizeof(buffer) - 1, 0);
  if (bytesRead <= 0) {
    closeFd(clientFd);
    return;
  }

  const std::string_view request(buffer, static_cast<size_t>(bytesRead));
  const std::string response =
      isMetricsRequest(request)
          ? httpResponse(200, "OK", "text/plain; version=0.0.4",
                         metrics_.renderText())
          : httpResponse(404, "Not Found", "text/plain", "not found\n");

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
