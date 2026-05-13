// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/i_socket_transport.hpp"

#include "config/settings.hpp"
#include "sockets/tcp_socket_transport.hpp"
#include "sockets/zmq_socket_transport.hpp"
#include "utils/logger.hpp"

namespace tt::sockets {

std::unique_ptr<ISocketTransport> createSocketTransport() {
  auto type = tt::config::socketTransport();
  if (type == "zmq") {
    TT_LOG_INFO("[SocketTransport] Using ZMQ transport");
    return std::make_unique<ZmqSocketTransport>();
  }
  TT_LOG_INFO("[SocketTransport] Using TCP transport");
  return std::make_unique<TcpSocketTransport>();
}

}  // namespace tt::sockets
