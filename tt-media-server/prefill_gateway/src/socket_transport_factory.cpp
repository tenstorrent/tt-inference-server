// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <cstdlib>
#include <string>

#include "sockets/i_socket_transport.hpp"
#include "sockets/tcp_socket_transport.hpp"
#include "sockets/zmq_socket_transport.hpp"
#include "utils/logger.hpp"

namespace tt::sockets {

namespace {
std::string socketTransportFromEnv() {
  const char* v = std::getenv("SOCKET_TRANSPORT");
  return v ? std::string(v) : std::string(transport_names::ZMQ);
}
}  // namespace

std::unique_ptr<ISocketTransport> createSocketTransport() {
  const std::string type = socketTransportFromEnv();
  if (type == transport_names::TCP) {
    TT_LOG_INFO("[Gateway] Using TCP transport");
    return std::make_unique<TcpSocketTransport>();
  }
  if (type != transport_names::ZMQ) {
    TT_LOG_WARN(
        "[Gateway] Unknown SOCKET_TRANSPORT='{}'; expected '{}' or '{}'. "
        "Falling back to ZMQ.",
        type, transport_names::TCP, transport_names::ZMQ);
  }
  TT_LOG_INFO("[Gateway] Using ZMQ transport");
  return std::make_unique<ZmqSocketTransport>();
}

}  // namespace tt::sockets
