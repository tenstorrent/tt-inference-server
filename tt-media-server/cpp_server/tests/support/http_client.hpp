// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Bare-bones blocking HTTP client for the integration tests. Avoids pulling
// in another HTTP library and keeps the wire format visible at the call site.

#pragma once

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace tt::test {

// Blocking HTTP POST to /v1/chat/completions. Returns the full response bytes
// (status line + headers + body). Throws on connect failure.
inline std::string sendAndReceive(
    const char* host, uint16_t port, const std::string& body,
    const std::string& apiKey = "your-secret-key") {
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  ::inet_pton(AF_INET, host, &addr.sin_addr);
  if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(sock);
    throw std::runtime_error("sendAndReceive: connect failed");
  }

  const std::string req =
      "POST /v1/chat/completions HTTP/1.1\r\n"
      "Host: " +
      std::string(host) + ":" + std::to_string(port) +
      "\r\n"
      "Content-Type: application/json\r\n"
      "Authorization: Bearer " +
      apiKey +
      "\r\n"
      "Content-Length: " +
      std::to_string(body.size()) +
      "\r\n"
      "\r\n" +
      body;
  ::send(sock, req.c_str(), req.size(), 0);

  // Idle-timeout-based read termination: keep recv'ing until we go quiet
  // for 250ms. Required because we don't send "Connection: close" — Drogon
  // closes the connection prematurely (after just headers) for async SSE
  // streams when the request asks for close — so we leave the connection
  // keep-alive and detect end-of-response by inactivity instead. Total
  // overhead per test: ~250ms after the last byte arrives.
  timeval tv{0, 250000};
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  std::string response;
  char buf[4096];
  ssize_t n;
  while ((n = ::recv(sock, buf, sizeof(buf), 0)) > 0)
    response.append(buf, static_cast<size_t>(n));
  ::close(sock);
  return response;
}

}  // namespace tt::test
