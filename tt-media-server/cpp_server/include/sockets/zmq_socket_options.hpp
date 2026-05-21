// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <zmq.hpp>

namespace tt::sockets::zmq_options {

constexpr int RECEIVE_TIMEOUT_MS = 100;
constexpr int HEARTBEAT_INTERVAL_MS = 1000;
constexpr int HEARTBEAT_TIMEOUT_MS = 3000;
constexpr int HEARTBEAT_TTL_MS = 3000;

inline void applyCommonOptions(zmq::socket_t& socket) {
  socket.set(zmq::sockopt::linger, 0);
  socket.set(zmq::sockopt::rcvtimeo, RECEIVE_TIMEOUT_MS);
  socket.set(zmq::sockopt::heartbeat_ivl, HEARTBEAT_INTERVAL_MS);
  socket.set(zmq::sockopt::heartbeat_timeout, HEARTBEAT_TIMEOUT_MS);
  socket.set(zmq::sockopt::heartbeat_ttl, HEARTBEAT_TTL_MS);
}

inline void applyRouterOptions(zmq::socket_t& socket) {
  applyCommonOptions(socket);
  socket.set(zmq::sockopt::router_mandatory, 1);
}

}  // namespace tt::sockets::zmq_options
