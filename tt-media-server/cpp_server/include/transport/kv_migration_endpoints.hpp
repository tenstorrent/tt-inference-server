// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include "sockets/i_socket_transport.hpp"
#include "transport/kv_control_channel.hpp"
#include "transport/kv_migration_orchestrator.hpp"
#include "transport/mooncake_kv_receiver.hpp"

namespace tt::transport {

/**
 * @brief Resolution → live channels: turns a `host → endpoint` map into the
 *        per-host `KvControlChannel`s that `KvMigrationMultiHostSender` consumes.
 *
 * The multi-host sender separates ROUTING (which hosts a request touches —
 * table-driven) from RESOLUTION (host → control channel — injected). This is the
 * resolution provider for the prefill side: given the decode hosts' endpoints,
 * it opens one client control channel per host and owns their lifetime.
 *
 * Socket creation is injected as a factory so this stays independent of
 * `src/sockets` (and unit-testable over a loopback fake): production passes a
 * `TcpSocketTransport` factory; tests pass a fake. Connectivity is established
 * lazily by the transport and surfaced per-migration (a dead channel makes that
 * host's `migrate()` fail and ack FAILED) — `connect()` only wires the channels.
 *
 * Lifetime: the connector owns the channels (and, via them, the transports), so
 * it must outlive the `KvMigrationMultiHostSender` built from `channels()`.
 */
class KvControlChannelConnector {
 public:
  struct Endpoint {
    std::string host;       ///< IP / DNS name to connect to.
    uint16_t port = 0;
  };

  /// Create a CLIENT transport aimed at `endpoint` (already
  /// initializeAsClient()'d + start()'ed), or nullptr on failure.
  using TransportFactory =
      std::function<std::shared_ptr<sockets::ISocketTransport>(
          const Endpoint& endpoint)>;

  KvControlChannelConnector(
      std::unordered_map<std::string, Endpoint> endpoints,
      TransportFactory factory,
      std::chrono::milliseconds receiveTimeout =
          KvControlChannel::kDefaultReceiveTimeout,
      std::chrono::milliseconds pollInterval =
          KvControlChannel::kDefaultPollInterval);

  /**
   * @brief Open a control channel for every endpoint.
   * @return true iff every endpoint's transport was created. A host whose
   *         factory returns nullptr is skipped (absent from channels()); the
   *         rest are still wired — same comprehensive-report contract as
   *         KvMigrationMultiHostSender (a missing host fails only its own slice).
   */
  bool connect();

  /// host → channel, for KvMigrationMultiHostSender. Contains only the hosts
  /// whose transport was created by connect().
  std::unordered_map<std::string, KvControlChannel*> channels() const;

  std::size_t connectedCount() const { return channels_.size(); }

 private:
  std::unordered_map<std::string, Endpoint> endpoints_;
  TransportFactory factory_;
  std::chrono::milliseconds receive_timeout_;
  std::chrono::milliseconds poll_interval_;
  // The channel keeps its transport alive (it holds a shared_ptr), so we only
  // need to own the channels.
  std::unordered_map<std::string, std::unique_ptr<KvControlChannel>> owned_;
  std::unordered_map<std::string, KvControlChannel*> channels_;
};

/**
 * @brief Decode-side server: listens for a sender, runs the receiver protocol.
 *
 * Owns a server control channel and runs `KvMigrationReceiver::run()` on a
 * background thread, dispatching prepareMirror/drain against the injected
 * `MooncakeKvReceiver` (which has already registered its full-table mirror as
 * the one Mooncake segment the sender writes into). The transport is injected
 * via a factory for the same decoupling/testability reason as the connector.
 *
 * Lifetime: `receiver` must outlive this server. stop() (also called by the
 * dtor) tears the transport down — which unblocks the receive loop — and joins
 * the thread; it is idempotent.
 */
class KvMigrationReceiverServer {
 public:
  /// Create a SERVER transport bound + listening on `port` (already
  /// initializeAsServer()'d + start()'ed), or nullptr on failure.
  using ServerTransportFactory =
      std::function<std::shared_ptr<sockets::ISocketTransport>(uint16_t port)>;

  KvMigrationReceiverServer(
      uint16_t port, ServerTransportFactory factory, MooncakeKvReceiver& receiver,
      std::chrono::milliseconds receiveTimeout =
          KvControlChannel::kDefaultReceiveTimeout,
      std::chrono::milliseconds pollInterval =
          KvControlChannel::kDefaultPollInterval);

  ~KvMigrationReceiverServer();

  KvMigrationReceiverServer(const KvMigrationReceiverServer&) = delete;
  KvMigrationReceiverServer& operator=(const KvMigrationReceiverServer&) =
      delete;

  /// Build the transport + channel + receiver orchestrator and spawn the loop.
  /// @return false if the transport factory failed (nothing started).
  bool start();

  /// Stop the transport (unblocks the loop) and join the thread. Idempotent.
  void stop();

  bool running() const { return running_; }

 private:
  uint16_t port_;
  ServerTransportFactory factory_;
  MooncakeKvReceiver& receiver_;
  std::chrono::milliseconds receive_timeout_;
  std::chrono::milliseconds poll_interval_;

  std::shared_ptr<sockets::ISocketTransport> transport_;
  std::unique_ptr<KvControlChannel> channel_;
  std::unique_ptr<KvMigrationReceiver> orchestrator_;
  std::thread thread_;
  bool running_ = false;
};

}  // namespace tt::transport
