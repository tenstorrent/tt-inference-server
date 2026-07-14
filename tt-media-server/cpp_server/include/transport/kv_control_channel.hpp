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
 * Multi-message protocols (migrate: Begin→MirrorReady→Done→Ack; TABLE_EXCHANGE:
 * send→recv) must run under Transaction: a recursive mutex held across the
 * whole sequence. Per-call locking alone is not enough — releasing between
 * send and receive lets another thread steal the peer's reply.
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

  /// Outcome of a receive attempt. Distinguishes a *timeout* (connection still
  /// live, no message yet) from a *close* — a long-lived server must keep
  /// waiting on the former but stop on the latter, which the std::optional
  /// receive() cannot express (it collapses both to nullopt).
  enum class ReceiveOutcome { Message, TimedOut, Closed };

  /**
   * @brief Holds the channel transaction mutex for a multi-message protocol.
   *
   * Prefer this around migrate() / TABLE_EXCHANGE. send()/receive() also take
   * the same recursive mutex so single-message decode serve loops stay correct
   * without an explicit Transaction.
   */
  class Transaction {
   public:
    explicit Transaction(KvControlChannel& channel);
    Transaction(KvControlChannel& channel, std::try_to_lock_t);
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    Transaction(Transaction&&) noexcept = default;
    Transaction& operator=(Transaction&&) noexcept = default;

    bool ownsLock() const { return lock_.owns_lock(); }

   private:
    std::unique_lock<std::recursive_mutex> lock_;
  };

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
  friend class Transaction;

  std::shared_ptr<sockets::ISocketTransport> transport_;
  std::chrono::milliseconds receive_timeout_;
  std::chrono::milliseconds poll_interval_;
  // Recursive: Transaction holds across migrate/exchange; send/receive re-lock.
  mutable std::recursive_mutex txnMutex_;
};

}  // namespace tt::transport
