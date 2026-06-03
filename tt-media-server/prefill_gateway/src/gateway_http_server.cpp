// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/gateway_http_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <utility>

#include "utils/logger.hpp"

namespace tt::gateway {
namespace {

constexpr int SOCKET_BACKLOG = 16;
constexpr size_t REQUEST_BUFFER_SIZE = 2048;

}  // namespace

GatewayHttpServer::GatewayHttpServer(std::string logName,
                                     std::string readyMessage,
                                     RequestHandler requestHandler)
    : logName(std::move(logName)),
      readyMessage(std::move(readyMessage)),
      requestHandler(std::move(requestHandler)) {}

GatewayHttpServer::~GatewayHttpServer() { stop(); }

bool GatewayHttpServer::start(uint16_t port) {
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

void GatewayHttpServer::stop() {
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

uint16_t GatewayHttpServer::port() const { return listeningPort; }

void GatewayHttpServer::serve(std::stop_token stopToken, uint16_t port,
                              std::promise<bool> initialized) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  serverFd = fd;
  if (fd < 0) {
    TT_LOG_ERROR("[{}] socket() failed: {}", logName, strerror(errno));
    running = false;
    initialized.set_value(false);
    return;
  }

  int reuse = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(port);

  if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
    TT_LOG_ERROR("[{}] bind(:{}) failed: {}", logName, port, strerror(errno));
    closeFd(serverFd.exchange(-1));
    running = false;
    initialized.set_value(false);
    return;
  }

  if (listen(fd, SOCKET_BACKLOG) < 0) {
    TT_LOG_ERROR("[{}] listen(:{}) failed: {}", logName, port, strerror(errno));
    closeFd(serverFd.exchange(-1));
    running = false;
    initialized.set_value(false);
    return;
  }

  TT_LOG_INFO("[{}] {} on port {}", logName, readyMessage, port);
  initialized.set_value(true);

  while (running && !stopToken.stop_requested()) {
    sockaddr_in clientAddress{};
    socklen_t clientLength = sizeof(clientAddress);
    const int activeServerFd = serverFd.load();
    if (activeServerFd < 0) {
      break;
    }
    const int clientFd =
        accept(activeServerFd, reinterpret_cast<sockaddr*>(&clientAddress),
               &clientLength);
    if (clientFd < 0) {
      if (running && errno != EINTR && errno != EBADF && errno != EINVAL) {
        TT_LOG_WARN("[{}] accept() failed: {}", logName, strerror(errno));
      }
      continue;
    }
    serveClient(clientFd);
  }
}

void GatewayHttpServer::serveClient(int clientFd) {
  std::array<char, REQUEST_BUFFER_SIZE> buffer{};
  const ssize_t bytesRead = recv(clientFd, buffer.data(), buffer.size() - 1, 0);
  if (bytesRead <= 0) {
    closeFd(clientFd);
    return;
  }

  const std::string_view request(buffer.data(), static_cast<size_t>(bytesRead));
  const auto response = requestHandler(request).value_or(
      GatewayHttpResponse{404, "Not Found", "text/plain", "not found\n"});
  const std::string serialized = formatHttpResponse(response);

  ssize_t offset = 0;
  while (offset < static_cast<ssize_t>(serialized.size())) {
    const ssize_t sent = send(clientFd, serialized.data() + offset,
                              serialized.size() - offset, 0);
    if (sent <= 0) {
      break;
    }
    offset += sent;
  }
  closeFd(clientFd);
}

std::string GatewayHttpServer::formatHttpResponse(
    const GatewayHttpResponse& response) {
  std::ostringstream out;
  out << "HTTP/1.1 " << response.status << " " << response.statusText
      << "\r\n"
      << "Content-Type: " << response.contentType << "\r\n"
      << "Content-Length: " << response.body.size() << "\r\n"
      << "Connection: close\r\n\r\n"
      << response.body;
  return out.str();
}

void GatewayHttpServer::closeFd(int fd) {
  if (fd >= 0) {
    close(fd);
  }
}

}  // namespace tt::gateway
