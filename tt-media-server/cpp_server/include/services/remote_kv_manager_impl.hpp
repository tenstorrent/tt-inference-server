// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "messaging/i_kafka_consumer.hpp"
#include "messaging/i_kafka_producer.hpp"
#include "services/remote_kv_manager.hpp"

namespace tt::services {

/**
 * Production implementation of IRemoteKVManager backed by Kafka.
 *
 * migrate() publishes a MigrationRequestMessage onto the request topic and
 * returns immediately with a fresh uint64 id. A single background thread
 * drains acknowledgments from the ack topic and updates an in-memory
 * status map. The same thread periodically sweeps the map and marks any
 * migration whose request was issued more than `timeout` ago and is still
 * IN_PROGRESS as FAILED, so callers eventually observe a terminal state
 * even if a worker disappears.
 *
 * Thread-safety: migrate() / getStatus() are safe to call from any thread.
 */
class RemoteKVManagerImpl : public IRemoteKVManager {
 public:
  /**
   * @param requestProducer       Kafka producer wired to the migration-
   *   request topic. Ownership is taken.
   * @param ackConsumer           Kafka consumer subscribed to the
   *   migration-ack topic, with a unique group.id. Ownership is taken.
   * @param migrationWorkerPoolSize  Number of migration workers the manager
   *   will fan download requests out to. Used by the download path's
   *   COMPLETION rule ("we wait for all N workers to ack"). The migrate()
   *   path is unaffected — it targets a single peer per request.
   *   Must be >= 1; values <= 0 are treated as 1 with a warning.
   * @param timeout               Max age of an IN_PROGRESS migration /
   *   download before the sweeper marks it FAILED. Default 60s.
   * @param sweepInterval         How often the drain thread runs the
   *   timeout sweep. Default 5s. Tests can pass a small value to force
   *   fast resolution.
   * @param drainPollMs           Per-iteration poll timeout passed to the
   *   consumer. Default 100ms. Lower values trade CPU for responsiveness.
   */
  RemoteKVManagerImpl(
      std::unique_ptr<tt::messaging::IKafkaProducer> requestProducer,
      std::unique_ptr<tt::messaging::IKafkaConsumer> ackConsumer,
      std::size_t migrationWorkerPoolSize,
      std::chrono::milliseconds timeout = std::chrono::seconds(60),
      std::chrono::milliseconds sweepInterval = std::chrono::seconds(5),
      int drainPollMs = 100);

  ~RemoteKVManagerImpl() override;

  RemoteKVManagerImpl(const RemoteKVManagerImpl&) = delete;
  RemoteKVManagerImpl& operator=(const RemoteKVManagerImpl&) = delete;

  [[nodiscard]] uint64_t migrate(const MigrationRequest& request) override;
  MigrationStatus getStatus(uint64_t migrationId) const override;

  // Mooncake-store path. The current implementation is a no-op: the
  // request payloads are NOT yet published to Kafka and no fan-out /
  // aggregation logic exists. downloadFromStore() returns a fresh id
  // and tracks the entry as IN_PROGRESS so the timeout sweep can
  // eventually flip it to FAILED — keeping the lifecycle observable
  // without lying about completion. offloadToStore() simply logs and
  // returns an id for correlation; nothing is tracked past return.
  [[nodiscard]] uint64_t downloadFromStore(
      const DownloadKVRequest& request) override;
  KVTransferResult getDownloadResult(uint64_t transferId) const override;
  uint64_t offloadToStore(const OffloadKVRequest& request) override;

 private:
  void drainLoop();
  // Caller must hold mtx.
  void sweepLocked(std::chrono::steady_clock::time_point now);

  struct MigrationState {
    MigrationStatus status;
    std::chrono::steady_clock::time_point submittedAt;
  };

  struct DownloadState {
    KVTransferStatus status;
    uint32_t usablePrefixCount;
    std::chrono::steady_clock::time_point submittedAt;
  };

  std::unique_ptr<tt::messaging::IKafkaProducer> requestProducer;
  std::unique_ptr<tt::messaging::IKafkaConsumer> ackConsumer;
  std::size_t migrationWorkerPoolSize;
  std::chrono::milliseconds timeout;
  std::chrono::milliseconds sweepInterval;
  int drainPollMs;

  mutable std::mutex mtx;
  std::unordered_map<uint64_t, MigrationState> migrations;
  std::unordered_map<uint64_t, DownloadState> downloads;

  std::atomic<bool> running{false};
  std::thread drainThread;
  std::chrono::steady_clock::time_point lastSweep{};
};

}  // namespace tt::services
