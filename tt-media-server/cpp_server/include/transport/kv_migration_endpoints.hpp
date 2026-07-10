// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "sockets/i_socket_transport.hpp"
#include "transport/kv_control_channel.hpp"
#include "transport/kv_migration_orchestrator.hpp"
#include "transport/mooncake_kv_receiver.hpp"

namespace tt::transport {

/**
 * @brief Resolution → live channels: turns a `host → endpoint` map into the
 *        per-host `KvControlChannel`s that `KvMigrationMultiHostSender`
 * consumes.
 *
 * The multi-host sender separates ROUTING (which hosts a request touches —
 * table-driven) from RESOLUTION (host → control channel — injected). This is
 * the resolution provider for the prefill side: given the decode hosts'
 * endpoints, it opens one client control channel per host and owns their
 * lifetime.
 *
 * Socket creation is injected as a factory so this stays independent of
 * `src/sockets` (and unit-testable over a loopback fake): production passes a
 * `TcpSocketTransport` factory; tests pass a fake. `openChannels()` only
 * CREATES the channels — the actual TCP connect runs in the transport's
 * background loop and completes asynchronously. Callers that must not act until
 * the peers are reachable (e.g. the prefill worker before it starts consuming
 * Kafka) should follow openChannels() with awaitConnected().
 *
 * Scope: this handles the STARTUP connect only. A channel that drops mid-run is
 * surfaced per-migration (that host's `migrate()` fails and acks FAILED);
 * steady-state re-connection / peer-drop recovery is out of scope here.
 *
 * Lifetime: the connector owns the channels (and, via them, the transports), so
 * it must outlive the `KvMigrationMultiHostSender` built from `channels()`.
 */
class KvControlChannelConnector {
 public:
  struct Endpoint {
    std::string host;  ///< IP / DNS name to connect to.
    uint16_t port = 0;
  };

  /// Create a CLIENT transport aimed at `endpoint` (already
  /// initializeAsClient()'d + start()'ed), or nullptr on failure.
  using TransportFactory =
      std::function<std::shared_ptr<sockets::ISocketTransport>(
          const Endpoint& endpoint)>;

  KvControlChannelConnector(std::unordered_map<std::string, Endpoint> endpoints,
                            TransportFactory factory,
                            std::chrono::milliseconds receiveTimeout =
                                KvControlChannel::kDefaultReceiveTimeout,
                            std::chrono::milliseconds pollInterval =
                                KvControlChannel::kDefaultPollInterval);

  /**
   * @brief Create a control channel for every endpoint (does NOT wait for TCP).
   * @return true iff every endpoint's transport was created. A host whose
   *         factory returns nullptr is skipped (absent from channels()); the
   *         rest are still wired — same comprehensive-report contract as
   *         KvMigrationMultiHostSender (a missing host fails only its own
   * slice).
   *
   * The transport connects asynchronously in its own background loop, so a
   * `true` return means the channels exist, NOT that any peer is reachable yet.
   * Use awaitConnected() to block until they are.
   */
  bool openChannels();

  /**
   * @brief Block until every created channel reports a live connection, or the
   *        timeout elapses.
   * @param timeout maximum time to wait for the asynchronous TCP connects.
   * @return the number of channels that are connected when this returns. Equals
   *         channels().size() if all came up before the timeout; a smaller
   *         value means the wait timed out with some peers still unreachable
   *         (the caller may proceed degraded — migrations to those hosts fail
   *         and ack FAILED). Returns immediately if no channels were created.
   *
   * Startup barrier only: it does not track peers that drop AFTER connecting.
   */
  std::size_t awaitConnected(std::chrono::milliseconds timeout);

  /// host → channel, for KvMigrationMultiHostSender. Contains only the hosts
  /// whose transport was created by openChannels().
  std::unordered_map<std::string, KvControlChannel*> channels() const;

  /// Number of channels created by openChannels() (NOT the number currently
  /// TCP-connected — see awaitConnected()).
  std::size_t channelCount() const { return channels_.size(); }

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
 * @brief Decode-side control server: accepts prefills, runs receiver protocol.
 *
 * Production (TcpSocketTransport): multi-accept — every connecting prefill gets
 * its own control session (channel + KvMigrationReceiver::run thread) so
 * TABLE_EXCHANGE and migrate work under N-prefill × 1-decode discovery.
 *
 * Unit-test fakes: factory returns an already-connected peer transport; the
 * server keeps the historical single-session path.
 *
 * Lifetime: `receiver` must outlive this server. stop() (also called by the
 * dtor) tears listen + all sessions down and joins threads; it is idempotent.
 */
class KvMigrationReceiverServer {
 public:
  /// Create a SERVER transport bound + listening on `port`. For TCP, leave it
  /// un-started — this server sets the multi-accept handler then start()s it.
  /// Test fakes may return an already-connected peer (single-session mode).
  using ServerTransportFactory =
      std::function<std::shared_ptr<sockets::ISocketTransport>(uint16_t port)>;

  /// @param localTableBlob decode `.pb` bytes for init-time TABLE_EXCHANGE
  ///        replies (empty = migrate-only; no table provisioning).
  KvMigrationReceiverServer(uint16_t port, ServerTransportFactory factory,
                            MooncakeKvReceiver& receiver,
                            std::vector<uint8_t> localTableBlob = {},
                            std::chrono::milliseconds receiveTimeout =
                                KvControlChannel::kDefaultReceiveTimeout,
                            std::chrono::milliseconds pollInterval =
                                KvControlChannel::kDefaultPollInterval);

  ~KvMigrationReceiverServer();

  KvMigrationReceiverServer(const KvMigrationReceiverServer&) = delete;
  KvMigrationReceiverServer& operator=(const KvMigrationReceiverServer&) =
      delete;

  /// Build the listen transport (or single fake session) and spawn accept /
  /// serve loops. @return false if the transport factory failed.
  bool start();

  /// Stop listen + all sessions and join threads. Idempotent.
  void stop();

  bool running() const { return running_; }

 private:
  struct Session {
    std::shared_ptr<sockets::ISocketTransport> transport;
    std::unique_ptr<KvControlChannel> channel;
    std::unique_ptr<KvMigrationReceiver> orchestrator;
    std::thread thread;
  };

  void startSingleSession(std::shared_ptr<sockets::ISocketTransport> transport);
  void onAccept(std::shared_ptr<sockets::ISocketTransport> peer);

  uint16_t port_;
  ServerTransportFactory factory_;
  MooncakeKvReceiver& receiver_;
  std::vector<uint8_t> local_table_blob_;
  std::chrono::milliseconds receive_timeout_;
  std::chrono::milliseconds poll_interval_;

  std::shared_ptr<sockets::ISocketTransport> listenTransport_;
  // Single-session (fake) path.
  std::unique_ptr<KvControlChannel> channel_;
  std::unique_ptr<KvMigrationReceiver> orchestrator_;
  std::thread thread_;

  std::mutex sessionsMutex_;
  std::vector<std::unique_ptr<Session>> sessions_;
  bool running_ = false;
};

}  // namespace tt::transport
