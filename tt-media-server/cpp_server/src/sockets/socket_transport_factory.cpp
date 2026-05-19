// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "config/settings.hpp"
#include "sockets/i_socket_transport.hpp"
#include "sockets/tcp_socket_transport.hpp"
#include "sockets/zmq_socket_transport.hpp"
#include "utils/logger.hpp"

namespace tt::sockets {

std::unique_ptr<ISocketTransport> createSocketTransport() {
  const std::string type = tt::config::socketTransport();
  if (type == transport_names::ZMQ) {
    TT_LOG_INFO("[SocketTransport] Using ZMQ transport");
    return std::make_unique<ZmqSocketTransport>();
  }
  if (type != transport_names::TCP) {
    TT_LOG_WARN(
        "[SocketTransport] Unknown SOCKET_TRANSPORT='{}'; expected '{}' or "
        "'{}'. Falling back to TCP.",
        type, transport_names::TCP, transport_names::ZMQ);
  }
  TT_LOG_INFO("[SocketTransport] Using TCP transport");
  return std::make_unique<TcpSocketTransport>();
}

}  // namespace tt::sockets
