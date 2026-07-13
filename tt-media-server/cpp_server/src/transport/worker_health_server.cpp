// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/worker_health_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

#include "transport/worker_health.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

namespace {

// How long accept() waits before rechecking the stop flag; bounds shutdown
// latency without busy-spinning.
constexpr int K_POLL_TIMEOUT_MS = 250;
// Bounds a single client's read/write so a stalled probe can't wedge the loop.
constexpr int K_CLIENT_TIMEOUT_SEC = 2;
constexpr std::size_t K_MAX_REQUEST_BYTES = 8192;
constexpr int K_LISTEN_BACKLOG = 16;

struct Route {
  int code;
  const char* reason;
  std::string contentType;
  std::string body;
};

const char* reasonFor(int code) {
  switch (code) {
    case 200:
      return "OK";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 503:
      return "Service Unavailable";
    default:
      return "OK";
  }
}

// The request line is "METHOD SP PATH SP VERSION"; we only need method + path
// (query string stripped). Returns false if the line is malformed.
bool parseRequestLine(const std::string& raw, std::string& method,
                      std::string& path) {
  const auto lineEnd = raw.find("\r\n");
  const std::string line =
      lineEnd == std::string::npos ? raw : raw.substr(0, lineEnd);
  const auto firstSpace = line.find(' ');
  if (firstSpace == std::string::npos) return false;
  const auto secondSpace = line.find(' ', firstSpace + 1);
  if (secondSpace == std::string::npos) return false;
  method = line.substr(0, firstSpace);
  path = line.substr(firstSpace + 1, secondSpace - firstSpace - 1);
  const auto query = path.find('?');
  if (query != std::string::npos) path.resize(query);
  return true;
}

Route routeFor(WorkerHealth& health, const std::string& method,
               const std::string& path) {
  if (method != "GET") {
    return {405, reasonFor(405), "application/json",
            R"({"error":"method not allowed"})"};
  }
  if (path == "/healthz") {
    const bool live = health.isLive();
    return {live ? 200 : 503, reasonFor(live ? 200 : 503), "application/json",
            health.healthJson()};
  }
  if (path == "/readyz") {
    const bool ready = health.isReady();
    return {ready ? 200 : 503, reasonFor(ready ? 200 : 503), "application/json",
            health.readyJson()};
  }
  if (path == "/metrics") {
    return {200, reasonFor(200), "text/plain; version=0.0.4",
            health.metricsText()};
  }
  return {404, reasonFor(404), "application/json", R"({"error":"not found"})"};
}

std::string buildResponse(const Route& r) {
  std::string out = "HTTP/1.1 ";
  out += std::to_string(r.code);
  out += ' ';
  out += r.reason;
  out += "\r\nContent-Type: ";
  out += r.contentType;
  out += "\r\nContent-Length: ";
  out += std::to_string(r.body.size());
  out += "\r\nConnection: close\r\n\r\n";
  out += r.body;
  return out;
}

// MSG_NOSIGNAL: a probe that disconnects before reading must not raise SIGPIPE
// and kill the whole worker (this process only handles SIGTERM/SIGINT). Match
// tcp_socket_transport.cpp. EINTR is a brief interruption — retry; EPIPE /
// ECONNRESET mean the client is gone — drop this response and keep serving.
bool sendAll(int fd, const std::string& data) {
  std::size_t sent = 0;
  while (sent < data.size()) {
    const ssize_t n = ::send(fd, data.data() + sent, data.size() - sent,
                             MSG_NOSIGNAL);
    if (n > 0) {
      sent += static_cast<std::size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

// Read until the end of the request headers ("\r\n\r\n"), a byte cap, or the
// client stalls/closes. We only parse the first line, but draining the headers
// keeps the client happy before we reply and close.
std::string readRequest(int fd) {
  std::string buf;
  std::array<char, 1024> chunk{};
  while (buf.size() < K_MAX_REQUEST_BYTES) {
    const ssize_t n = ::recv(fd, chunk.data(), chunk.size(), 0);
    if (n <= 0) break;
    buf.append(chunk.data(), static_cast<std::size_t>(n));
    if (buf.find("\r\n\r\n") != std::string::npos) break;
  }
  return buf;
}

void setClientTimeout(int fd) {
  timeval tv{K_CLIENT_TIMEOUT_SEC, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

}  // namespace

WorkerHealthServer::WorkerHealthServer(WorkerHealth& health, std::string host,
                                       uint16_t port)
    : health_(health), host_(std::move(host)), port_(port) {}

WorkerHealthServer::~WorkerHealthServer() { stop(); }

bool WorkerHealthServer::start() {
  listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listenFd_ < 0) {
    TT_LOG_ERROR("[WorkerHealthServer] socket() failed: {}",
                 std::strerror(errno));
    return false;
  }
  const int one = 1;
  ::setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_);
  if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
    TT_LOG_ERROR("[WorkerHealthServer] invalid host '{}'", host_);
    ::close(listenFd_);
    listenFd_ = -1;
    return false;
  }
  if (::bind(listenFd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 ||
      ::listen(listenFd_, K_LISTEN_BACKLOG) < 0) {
    TT_LOG_ERROR("[WorkerHealthServer] bind/listen {}:{} failed: {}", host_,
                 port_, std::strerror(errno));
    ::close(listenFd_);
    listenFd_ = -1;
    return false;
  }

  // Resolve the ephemeral port when 0 was requested (tests read it back).
  socklen_t len = sizeof(addr);
  if (::getsockname(listenFd_, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
    port_ = ntohs(addr.sin_port);
  }

  running_.store(true);
  thread_ = std::thread(&WorkerHealthServer::acceptLoop, this);
  TT_LOG_INFO("[WorkerHealthServer] serving /healthz /readyz /metrics on {}:{}",
              host_, port_);
  return true;
}

void WorkerHealthServer::stop() {
  if (!running_.exchange(false)) return;
  if (thread_.joinable()) thread_.join();
  if (listenFd_ >= 0) {
    ::close(listenFd_);
    listenFd_ = -1;
  }
}

void WorkerHealthServer::acceptLoop() {
  while (running_.load()) {
    pollfd pfd{listenFd_, POLLIN, 0};
    const int rc = ::poll(&pfd, 1, K_POLL_TIMEOUT_MS);
    if (rc <= 0 || (pfd.revents & POLLIN) == 0) continue;
    const int clientFd = ::accept(listenFd_, nullptr, nullptr);
    if (clientFd < 0) continue;
    setClientTimeout(clientFd);
    handleConnection(clientFd);
    ::close(clientFd);
  }
}

void WorkerHealthServer::handleConnection(int clientFd) {
  const std::string request = readRequest(clientFd);
  std::string method;
  std::string path;
  if (!parseRequestLine(request, method, path)) {
    sendAll(clientFd, buildResponse({400, "Bad Request", "application/json",
                                     R"({"error":"bad request"})"}));
    return;
  }
  sendAll(clientFd, buildResponse(routeFor(health_, method, path)));
}

}  // namespace tt::transport
