// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <cstdlib>
#include <string>

#include "sockets/i_socket_transport.hpp"
#include "sockets/tcp_socket_transport.hpp"
#include "sockets/zmq_socket_transport.hpp"
#include "utils/logger.hpp"

namespace tt::sockets {

std::unique_ptr<ISocketTransport> createSocketTransport() {
  const char* envVal = std::getenv("SOCKET_TRANSPORT");
  std::string type = envVal ? envVal : "tcp";
  if (type == "zmq") {
    TT_LOG_INFO("[Gateway] Using ZMQ transport");
    return std::make_unique<ZmqSocketTransport>();
  }
  TT_LOG_INFO("[Gateway] Using TCP transport");
  return std::make_unique<TcpSocketTransport>();
}

}  // namespace tt::sockets
