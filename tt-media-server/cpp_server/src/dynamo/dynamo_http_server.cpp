// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/dynamo_http_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>

#include "utils/logger.hpp"

namespace tt::dynamo {

namespace {

bool writeAll(int fd, const char* data, size_t len) {
  size_t written = 0;
  while (written < len) {
    ssize_t w = ::write(fd, data + written, len - written);
    if (w <= 0) return false;
    written += static_cast<size_t>(w);
  }
  return true;
}

std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

/// Normalize rpc root: no trailing slash, leading slash required.
std::string normalizeRpcRoot(std::string path) {
  if (path.empty()) path = "/v1/rpc";
  if (path.front() != '/') path.insert(path.begin(), '/');
  while (path.size() > 1 && path.back() == '/') {
    path.pop_back();
  }
  return path;
}

}  // namespace

DynamoHttpServer::DynamoHttpServer(HttpServerConfig config,
                                   GenerateHandler handler)
    : config_(std::move(config)), handler_(std::move(handler)) {
  config_.rpc_root_path = normalizeRpcRoot(config_.rpc_root_path);
}

DynamoHttpServer::~DynamoHttpServer() { shutdown(); }

void DynamoHttpServer::shutdown() {
  running_.store(false);
  if (listen_fd_ >= 0) {
    int fd = listen_fd_;
    listen_fd_ = -1;
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
  }
}

bool DynamoHttpServer::read_http_request(int client_fd,
                                         std::vector<uint8_t>& body) {
  std::string headerBlock;
  headerBlock.reserve(4096);
  char buf[4096];
  while (headerBlock.find("\r\n\r\n") == std::string::npos) {
    ssize_t n = ::read(client_fd, buf, sizeof(buf));
    if (n <= 0) return false;
    headerBlock.append(buf, static_cast<size_t>(n));
    if (headerBlock.size() > 1024 * 1024) return false;
  }

  const size_t headerEnd = headerBlock.find("\r\n\r\n");
  const std::string headers = headerBlock.substr(0, headerEnd);
  std::string leftover = headerBlock.substr(headerEnd + 4);

  std::istringstream headerStream(headers);
  std::string requestLine;
  if (!std::getline(headerStream, requestLine)) return false;
  if (!requestLine.empty() && requestLine.back() == '\r') {
    requestLine.pop_back();
  }

  std::istringstream lineStream(requestLine);
  std::string method;
  std::string target;
  std::string version;
  lineStream >> method >> target >> version;
  if (method != "POST") return false;

  const std::string expectedPath =
      config_.rpc_root_path + "/" + config_.endpoint_name;
  if (target != expectedPath && target != expectedPath + "/") {
    TT_LOG_DEBUG("[DynamoHttpServer] Rejecting path '{}' (want '{}')", target,
                 expectedPath);
    return false;
  }

  size_t contentLength = 0;
  std::string line;
  while (std::getline(headerStream, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string name = toLower(line.substr(0, colon));
    std::string value = line.substr(colon + 1);
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
      value.erase(value.begin());
    }
    if (name == "content-length") {
      try {
        contentLength = static_cast<size_t>(std::stoull(value));
      } catch (...) {
        return false;
      }
    }
  }

  body.assign(leftover.begin(), leftover.end());
  if (body.size() > contentLength) {
    body.resize(contentLength);
  }
  while (body.size() < contentLength) {
    char chunk[8192];
    const size_t need = contentLength - body.size();
    const size_t toRead = std::min(need, sizeof(chunk));
    ssize_t n = ::read(client_fd, chunk, toRead);
    if (n <= 0) return false;
    body.insert(body.end(), chunk, chunk + n);
  }
  return true;
}

void DynamoHttpServer::handle_connection(int client_fd) {
  std::vector<uint8_t> body;
  const bool ok = read_http_request(client_fd, body);
  if (ok) {
    TT_LOG_DEBUG("[DynamoHttpServer] Accepted POST payload_bytes={}",
                 body.size());
    static constexpr char kAccepted[] =
        "HTTP/1.1 202 Accepted\r\n"
        "Content-Length: 0\r\n"
        "Connection: close\r\n"
        "\r\n";
    writeAll(client_fd, kAccepted, sizeof(kAccepted) - 1);
    process_two_part_payload(body, handler_);
  } else {
    static constexpr char kNotFound[] =
        "HTTP/1.1 404 Not Found\r\n"
        "Content-Length: 0\r\n"
        "Connection: close\r\n"
        "\r\n";
    writeAll(client_fd, kNotFound, sizeof(kNotFound) - 1);
  }
  ::close(client_fd);
}

void DynamoHttpServer::run() {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    throw std::runtime_error(
        "DynamoHttpServer: failed to create listen socket");
  }

  int opt = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in bind_addr{};
  bind_addr.sin_family = AF_INET;
  bind_addr.sin_port = htons(config_.bind_port);
  ::inet_pton(AF_INET, config_.bind_host.c_str(), &bind_addr.sin_addr);

  if (::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&bind_addr),
             sizeof(bind_addr)) < 0) {
    throw std::runtime_error("DynamoHttpServer: failed to bind");
  }

  struct sockaddr_in actual{};
  socklen_t len = sizeof(actual);
  ::getsockname(listen_fd_, reinterpret_cast<struct sockaddr*>(&actual), &len);
  actual_port_ = ntohs(actual.sin_port);

  if (::listen(listen_fd_, 128) < 0) {
    throw std::runtime_error("DynamoHttpServer: failed to listen");
  }

  running_.store(true);
  TT_LOG_INFO("[DynamoHttpServer] Listening on {}:{} path={}/{}",
              config_.bind_host, actual_port_, config_.rpc_root_path,
              config_.endpoint_name);

  while (running_.load()) {
    int client_fd = ::accept(listen_fd_, nullptr, nullptr);
    if (client_fd < 0) {
      if (!running_.load()) break;
      continue;
    }
    std::thread([this, client_fd]() { handle_connection(client_fd); }).detach();
  }
}

}  // namespace tt::dynamo
