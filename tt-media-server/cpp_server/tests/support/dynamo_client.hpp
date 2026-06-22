// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// HTTP client helpers for sending requests through Dynamo frontend.
//
// The integration tests can use these to route requests through the external
// Dynamo frontend (HTTP → Dynamo → TCP → DynamoEndpoint → LLMPipeline) while
// still inspecting IPC queues for gray-box assertions.

#pragma once

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace tt::test {

// Detect Docker gateway IP from /proc/net/route (for container-to-host access)
inline std::string detectDockerGateway() {
  std::ifstream route("/proc/net/route");
  if (!route) return "127.0.0.1";

  std::string line;
  std::getline(route, line);  // skip header
  while (std::getline(route, line)) {
    std::istringstream iss(line);
    std::string iface, dest, gateway;
    if (iss >> iface >> dest >> gateway && dest == "00000000") {
      unsigned int gw = std::stoul(gateway, nullptr, 16);
      unsigned char* bytes = reinterpret_cast<unsigned char*>(&gw);
      return std::to_string(bytes[0]) + "." + std::to_string(bytes[1]) + "." +
             std::to_string(bytes[2]) + "." + std::to_string(bytes[3]);
    }
  }
  return "127.0.0.1";
}

struct DynamoConfig {
  std::string host = detectDockerGateway();
  uint16_t port = 8080;  // Dynamo frontend default from deploy.sh
  // Default model matches deploy.sh default (HF_MODEL_ID=deepseek-ai/DeepSeek-R1-0528)
  std::string model = "deepseek-ai/DeepSeek-R1-0528";

  static DynamoConfig fromEnv() {
    DynamoConfig cfg;
    if (const char* h = std::getenv("DYNAMO_HOST")) cfg.host = h;
    if (const char* p = std::getenv("DYNAMO_PORT")) cfg.port = std::stoi(p);
    if (const char* m = std::getenv("DYNAMO_MODEL")) cfg.model = m;
    return cfg;
  }
};

// Build HTTP request for Dynamo frontend's /v1/chat/completions endpoint.
inline std::string buildDynamoHttpRequest(const std::string& host,
                                          uint16_t port,
                                          const std::string& body) {
  std::ostringstream oss;
  oss << "POST /v1/chat/completions HTTP/1.1\r\n"
      << "Host: " << host << ":" << port << "\r\n"
      << "Content-Type: application/json\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "\r\n"
      << body;
  return oss.str();
}

// Blocking HTTP POST to Dynamo frontend. Returns the full response bytes
// (status line + headers + body). For SSE streams, reads until "data: [DONE]"
// or timeout. Throws on connect failure.
inline std::string sendDynamoRequest(const DynamoConfig& cfg,
                                     const std::string& body,
                                     int timeoutMs = 30000) {
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    throw std::runtime_error("sendDynamoRequest: failed to create socket");
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(cfg.port);

  if (::inet_pton(AF_INET, cfg.host.c_str(), &addr.sin_addr) <= 0) {
    struct hostent* he = ::gethostbyname(cfg.host.c_str());
    if (!he) {
      ::close(sock);
      throw std::runtime_error("sendDynamoRequest: failed to resolve host: " +
                               cfg.host);
    }
    std::memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
  }

  if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(sock);
    throw std::runtime_error("sendDynamoRequest: failed to connect to " +
                             cfg.host + ":" + std::to_string(cfg.port));
  }

  std::string request = buildDynamoHttpRequest(cfg.host, cfg.port, body);
  if (::send(sock, request.c_str(), request.size(), 0) < 0) {
    ::close(sock);
    throw std::runtime_error("sendDynamoRequest: failed to send request");
  }

  timeval tv{timeoutMs / 1000, (timeoutMs % 1000) * 1000};
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  std::string response;
  char buf[4096];
  ssize_t n;

  // For SSE, read until we see "data: [DONE]" or timeout
  while ((n = ::recv(sock, buf, sizeof(buf), 0)) > 0) {
    response.append(buf, static_cast<size_t>(n));
    if (response.find("data: [DONE]") != std::string::npos) {
      break;
    }
  }

  ::close(sock);
  return response;
}

// Wait for Dynamo frontend to be reachable. Returns true if connected within
// timeout, false otherwise.
inline bool waitForDynamoFrontend(const DynamoConfig& cfg,
                                  int timeoutSec = 60) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);

  while (std::chrono::steady_clock::now() < deadline) {
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(cfg.port);
    ::inet_pton(AF_INET, cfg.host.c_str(), &addr.sin_addr);

    timeval tv{2, 0};
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    bool connected =
        ::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    ::close(sock);

    if (connected) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  return false;
}

// Send a warmup request to ensure Dynamo frontend is fully ready (has
// discovered backends). Returns true if warmup succeeded.
inline bool warmupDynamoFrontend(const DynamoConfig& cfg) {
  const std::string warmupBody =
      R"({"model":")" + cfg.model +
      R"(","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":true})";

  for (int attempt = 0; attempt < 10; ++attempt) {
    try {
      std::string response = sendDynamoRequest(cfg, warmupBody, 30000);
      if (response.find("HTTP/1.1 200") != std::string::npos ||
          response.find("data:") != std::string::npos) {
        return true;
      }
    } catch (...) {
      // Connection failed, retry
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  return false;
}

}  // namespace tt::test
