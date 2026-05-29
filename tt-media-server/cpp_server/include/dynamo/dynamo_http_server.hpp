// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

/**
 * Dynamo HTTP request-plane ingress (DYN_REQUEST_PLANE=http).
 *
 * Listens for HTTP/1.1 POST requests whose body is a TwoPartCodec frame
 * (the same bytes the TCP plane places in the request payload). Responds
 * with 202 Accepted immediately, then processes the request asynchronously
 * and streams tokens back to the frontend over the TCP call-home path.
 *
 * NVIDIA Dynamo's frontend HttpRequestClient negotiates HTTP/2 when the
 * stack supports it; HTTP/1.1 is accepted on this listener.
 */

#include <atomic>
#include <cstdint>
#include <string>

#include "dynamo/dynamo_protocol.hpp"

namespace tt::dynamo {

struct HttpServerConfig {
  std::string bind_host = "0.0.0.0";
  /// 0 = OS-assigned (recommended). Set DYN_HTTP_RPC_PORT for a fixed port.
  uint16_t bind_port = 0;
  /// Must match DYN_HTTP_RPC_ROOT_PATH on frontend and worker (default
  /// /v1/rpc).
  std::string rpc_root_path = "/v1/rpc";
  /// Endpoint name registered in discovery (e.g. "generate").
  std::string endpoint_name = "generate";
};

class DynamoHttpServer {
 public:
  DynamoHttpServer(HttpServerConfig config, GenerateHandler handler);
  ~DynamoHttpServer();

  DynamoHttpServer(const DynamoHttpServer&) = delete;
  DynamoHttpServer& operator=(const DynamoHttpServer&) = delete;

  /// Start listening. Blocks the calling thread; intended to run in its own
  /// thread.
  void run();

  void shutdown();

  uint16_t port() const { return actual_port_; }

 private:
  HttpServerConfig config_;
  GenerateHandler handler_;
  int listen_fd_ = -1;
  uint16_t actual_port_ = 0;
  std::atomic<bool> running_{false};

  void handle_connection(int client_fd);
  bool read_http_request(int client_fd, std::vector<uint8_t>& body);
};

}  // namespace tt::dynamo
