// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
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
 * Scope: openChannels() / openChannel() / replaceChannel() create client
 * transports; TCP connect runs asynchronously. Production prefill uses a
 * fail-closed startup barrier, then a mesh watch that re-resolves
 * kv_control/<name> from metadata when a peer drops so a restarted decode with
 * a new host:port gets a fresh channel (not only sticky TCP reconnect).
 *
 * Lifetime: channels are shared_ptr-owned. The connector keeps one strong
 * reference per peer; callers (e.g. KvMigrationMultiHostSender::migrate) that
 * snapshot a channel keep it alive across replaceChannel() so an in-flight
 * migration cannot dereference a destroyed object. openChannel() /
 * replaceChannel() are thread-safe vs channels() / channelCount() /
 * awaitConnected().
 */
class KvControlChannelConnector {
 public:
  struct Endpoint {
    std::string host;  ///< IP / DNS name to connect to.
    uint16_t port = 0;

    bool operator==(const Endpoint& other) const {
      return host == other.host && port == other.port;
    }
    bool operator!=(const Endpoint& other) const { return !(*this == other); }
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
   * @brief Create (or no-op if already present) a control channel for one peer.
   * @return true if the channel exists after the call (created now or earlier).
   *         false if the factory failed for a new peer.
   *
   * Does not replace an existing channel — use replaceChannel() when metadata
   * republishes a different host:port after a peer restart.
   */
  bool openChannel(const std::string& name, const Endpoint& endpoint);

  /**
   * @brief Tear down any existing channel for @p name and open @p endpoint.
   * @return true if a channel to @p endpoint exists after the call.
   *
   * Used by the post-Ready mesh watch when kv_control/<name> moves (new IP).
   */
  bool replaceChannel(const std::string& name, const Endpoint& endpoint);

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

  /// host → channel. Contains only hosts whose transport was created.
  /// Shared ownership: holding a returned pointer keeps that channel alive
  /// even if replaceChannel() later drops the connector's reference.
  std::unordered_map<std::string, std::shared_ptr<KvControlChannel>> channels()
      const;

  /// Last endpoint registered for @p name, if any.
  std::optional<Endpoint> endpoint(const std::string& name) const;

  /// Number of channels created (NOT the number currently TCP-connected —
  /// see awaitConnected() / connectedCount()).
  std::size_t channelCount() const;

  /// How many created channels currently report isConnected().
  std::size_t connectedCount() const;

 private:
  bool openChannelLocked(const std::string& name, const Endpoint& endpoint);
  bool replaceChannelLocked(const std::string& name, const Endpoint& endpoint);

  mutable std::mutex mutex_;
  std::unordered_map<std::string, Endpoint> endpoints_;
  TransportFactory factory_;
  std::chrono::milliseconds receive_timeout_;
  std::chrono::milliseconds poll_interval_;
  // Shared so migrate() / mesh-watch can keep a channel alive across
  // replaceChannel() without coordinating with the connector mutex.
  std::unordered_map<std::string, std::shared_ptr<KvControlChannel>> channels_;
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
 * Lifetime: a non-null `receiver` must outlive this server. A null receiver
 * enables control-only dry-run mode. stop() (also called by the dtor) tears
 * listen + all sessions down and joins threads; it is idempotent.
 */
class KvMigrationReceiverServer {
 public:
  /// Create a SERVER transport bound + listening on `port`. For TCP, leave it
  /// un-started — this server sets the multi-accept handler then start()s it.
  /// Test fakes may return an already-connected peer (single-session mode).
  using ServerTransportFactory =
      std::function<std::shared_ptr<sockets::ISocketTransport>(uint16_t port)>;

  /// @param localTableBlob decode `.pb` bytes for init-time TABLE_EXCHANGE
  ///        replies (empty = migrate-only; no table provisioning). Held once
  ///        and shared by every accepted prefill session.
  KvMigrationReceiverServer(uint16_t port, ServerTransportFactory factory,
                            MooncakeKvReceiver& receiver,
                            std::vector<uint8_t> localTableBlob = {},
                            std::chrono::milliseconds receiveTimeout =
                                KvControlChannel::kDefaultReceiveTimeout,
                            std::chrono::milliseconds pollInterval =
                                KvControlChannel::kDefaultPollInterval);
  KvMigrationReceiverServer(uint16_t port, ServerTransportFactory factory,
                            MooncakeKvReceiver* receiver,
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

  /// Live (not yet finished) multi-accept sessions. Test/observability aid.
  std::size_t activeSessionCount() const;

 private:
  struct Session {
    std::shared_ptr<sockets::ISocketTransport> transport;
    std::unique_ptr<KvControlChannel> channel;
    std::unique_ptr<KvMigrationReceiver> orchestrator;
    std::thread thread;
    std::atomic<bool> finished{false};
  };

  void startSingleSession(std::shared_ptr<sockets::ISocketTransport> transport);
  void onAccept(std::shared_ptr<sockets::ISocketTransport> peer);
  /// Join + erase finished sessions. Skips the calling thread's own session
  /// (detach+erase that one). Caller may hold sessionsMutex_ via the Locked
  /// overload; the public entry takes the lock.
  void reapFinishedSessions();
  void reapFinishedSessionsLocked();

  uint16_t port_;
  ServerTransportFactory factory_;
  MooncakeKvReceiver* receiver_;
  std::shared_ptr<const std::vector<uint8_t>> local_table_blob_;
  std::chrono::milliseconds receive_timeout_;
  std::chrono::milliseconds poll_interval_;

  std::shared_ptr<sockets::ISocketTransport> listenTransport_;
  // Single-session (fake) path.
  std::unique_ptr<KvControlChannel> channel_;
  std::unique_ptr<KvMigrationReceiver> orchestrator_;
  std::thread thread_;

  mutable std::mutex sessionsMutex_;
  std::vector<std::unique_ptr<Session>> sessions_;
  bool running_ = false;
};

}  // namespace tt::transport
