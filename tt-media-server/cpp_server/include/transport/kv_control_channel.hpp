// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <memory>
#include <optional>

#include "sockets/i_socket_transport.hpp"
#include "transport/kv_control_message.hpp"

namespace tt::transport {

/**
 * @brief Control channel for KV migration over a message-framed socket.
 *
 * A thin wrapper that serializes KvControlMessages onto an ISocketTransport
 * (one sendRawData == one logical message) and parses them back. It carries
 * only setup/coordination — table exchange, the migration request, the mirror
 * segment name, done-marker, and ack — never the bulk KV bytes, which move
 * one-sided over Mooncake.
 *
 * The bare receiveRawData() returns an empty buffer for BOTH "connection
 * closed" and "no message ready yet" (a non-blocking transport such as
 * TcpSocketTransport returns empty whenever nothing is buffered yet). Mapping
 * every empty read to "closed" raced with a normal few-millisecond reply and
 * aborted healthy migrations. So receive() uses the transport's tri-state
 * tryReceiveMessage() instead: DATA is delivered, CLOSED fails, and NO_DATA
 * (live connection, peer hasn't replied yet) is retried up to a configurable
 * timeout. The transport reports the true recv() status directly, so the
 * channel never has to guess "not ready vs closed" by polling isConnected().
 *
 * The transport is injected so this is independent of the socket factory and
 * usable over a loopback pair in tests. The receive timeout/poll interval are
 * injectable so tests can keep the not-ready/timeout path fast.
 */
class KvControlChannel {
 public:
  static constexpr std::chrono::milliseconds kDefaultReceiveTimeout{30000};
  static constexpr std::chrono::milliseconds kDefaultPollInterval{2};

  explicit KvControlChannel(
      std::shared_ptr<sockets::ISocketTransport> transport,
      std::chrono::milliseconds receiveTimeout = kDefaultReceiveTimeout,
      std::chrono::milliseconds pollInterval = kDefaultPollInterval);

  /// True if the underlying transport reports a live connection.
  bool isConnected() const;

  /// Serialize and send a message. @return false if the transport rejects it.
  bool send(const KvControlMessage& message);

  /**
   * @brief Wait for the next message.
   *
   * Retries while the transport reports NO_DATA (peer hasn't replied yet).
   * Returns std::nullopt on transport close (CLOSED), a malformed message, or
   * if no message arrives within the receive timeout.
   */
  std::optional<KvControlMessage> receive();

 private:
  std::shared_ptr<sockets::ISocketTransport> transport_;
  std::chrono::milliseconds receive_timeout_;
  std::chrono::milliseconds poll_interval_;
};

}  // namespace tt::transport
