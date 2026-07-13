// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <memory>
#include <mutex>
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
 * timeout. The same timeout is armed as a transport IoBudget around send() and
 * receiveMessage() so a mid-frame stall cannot pin the socket mutex past the
 * advertised deadline (retry counts alone are the wrong unit for 100–350+ MiB
 * TABLE_EXCHANGE payloads). The transport reports the true recv() status
 * directly, so the channel never has to guess "not ready vs closed" by polling
 * isConnected().
 *
 * The transport is injected so this is independent of the socket factory and
 * usable over a loopback pair in tests. The receive timeout/poll interval are
 * injectable so tests can keep the not-ready/timeout path fast.
 *
 * send() / receiveMessage() share an IO mutex so a post-Ready mesh watch can
 * re-run TABLE_EXCHANGE on reconnect without interleaving frames with migrate().
 */
class KvControlChannel {
 public:
  static constexpr std::chrono::milliseconds kDefaultReceiveTimeout{30000};
  static constexpr std::chrono::milliseconds kDefaultPollInterval{2};

  explicit KvControlChannel(
      std::shared_ptr<sockets::ISocketTransport> transport,
      std::chrono::milliseconds receiveTimeout = kDefaultReceiveTimeout,
      std::chrono::milliseconds pollInterval = kDefaultPollInterval);

  /// Outcome of a receive attempt. Distinguishes a *timeout* (connection still
  /// live, no message yet) from a *close* — a long-lived server must keep
  /// waiting on the former but stop on the latter, which the std::optional
  /// receive() cannot express (it collapses both to nullopt).
  enum class ReceiveOutcome { Message, TimedOut, Closed };

  /// True if the underlying transport reports a live connection.
  bool isConnected() const;

  /// Serialize and send a message. @return false if the transport rejects it.
  bool send(const KvControlMessage& message);

  /**
   * @brief Wait for the next message.
   *
   * Retries while the transport reports NO_DATA (peer hasn't replied yet).
   * Returns std::nullopt on transport close (CLOSED), a malformed message, or
   * if no message arrives within the receive timeout. Prefer receiveMessage()
   * when the caller must distinguish an idle timeout from a real close.
   */
  std::optional<KvControlMessage> receive();

  /**
   * @brief Tri-state receive: fills `out` and returns Message, or reports
   *        TimedOut / Closed. A malformed frame is reported as Closed (the
   *        stream is unusable). Unlike receive(), a timeout is not logged as an
   *        error here — it is normal for a server idling between requests.
   */
  ReceiveOutcome receiveMessage(KvControlMessage& out);

 private:
  std::shared_ptr<sockets::ISocketTransport> transport_;
  std::chrono::milliseconds receive_timeout_;
  std::chrono::milliseconds poll_interval_;
  mutable std::mutex ioMutex_;
};

}  // namespace tt::transport
