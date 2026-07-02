// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "messaging/i_kafka_consumer.hpp"
#include "messaging/i_kafka_producer.hpp"
#include "messaging/migration_message.hpp"
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
   * Maps a MigrationRequest's layer_id to the Kafka partition that owns
   * that layer's work. Returning a negative value falls back to the
   * broker-picked partition (i.e. legacy un-partitioned behavior).
   */
  using LayerToPartition = std::function<int32_t(uint32_t layerId)>;

  /**
   * @param requestProducer       Kafka producer wired to the migration-
   *   request topic. Ownership is taken.
   * @param ackConsumer           Kafka consumer subscribed to the
   *   migration-ack topic, with a unique group.id. Ownership is taken.
   * @param migrationWorkerPoolSize  Number of migration workers the manager
   *   fans download requests out to. A download reaches COMPLETED once
   *   this many workers have ack'd. Must be >= 1.
   * @param timeout               Max age of an IN_PROGRESS migration /
   *   download before the sweeper marks it FAILED. Default 60s.
   * @param sweepInterval         How often the drain thread runs the
   *   timeout sweep. Default 5s.
   * @param drainPollMs           Per-iteration poll timeout passed to the
   *   consumer. Default 100ms.
   * @param downloadRequestProducer  Kafka producer for Mooncake-store
   *   download requests. Optional; if null, downloadFromStore() rolls
   *   straight to FAILED.
   * @param downloadAckConsumer   Kafka consumer for download acks, with a
   *   unique group.id. Optional; if null, downloads stay IN_PROGRESS
   *   until the sweeper times them out.
   * @param offloadRequestProducer  Kafka producer for offload requests.
   *   Optional; offload is fire-and-forget, no ack consumer needed.
   */
  RemoteKVManagerImpl(
      std::unique_ptr<tt::messaging::IKafkaProducer> requestProducer,
      std::unique_ptr<tt::messaging::IKafkaConsumer> ackConsumer,
      std::size_t migrationWorkerPoolSize,
      std::chrono::milliseconds timeout = std::chrono::seconds(60),
      std::chrono::milliseconds sweepInterval = std::chrono::seconds(5),
      int drainPollMs = 100,
      std::unique_ptr<tt::messaging::IKafkaProducer> downloadRequestProducer =
          nullptr,
      std::unique_ptr<tt::messaging::IKafkaConsumer> downloadAckConsumer =
          nullptr,
      std::unique_ptr<tt::messaging::IKafkaProducer> offloadRequestProducer =
          nullptr,
      std::unique_ptr<tt::messaging::IKafkaConsumer> offloadAckConsumer =
          nullptr,
      LayerToPartition layerToPartition = nullptr);

  ~RemoteKVManagerImpl() override;

  RemoteKVManagerImpl(const RemoteKVManagerImpl&) = delete;
  RemoteKVManagerImpl& operator=(const RemoteKVManagerImpl&) = delete;

  [[nodiscard]] uint64_t migrate(const MigrationRequest& request) override;
  MigrationStatus getMigrationStatus(uint64_t migrationId) const override;

  [[nodiscard]] uint64_t downloadFromStore(
      const DownloadKVRequest& request) override;
  DownloadKVResult getDownloadResult(uint64_t transferId) const override;
  [[nodiscard]] uint64_t offloadToStore(
      const OffloadKVRequest& request) override;
  MigrationStatus getOffloadStatus(uint64_t transferId) const override;

 private:
  void drainLoop();
  // Caller must hold mtx.
  void sweepLocked(std::chrono::steady_clock::time_point now);
  // Takes the lock internally.
  void applyDownloadAck(const tt::messaging::DownloadResponseMessage& ack);

  struct MigrationState {
    MigrationStatus status;
    std::chrono::steady_clock::time_point submittedAt;
  };

  struct DownloadState {
    MigrationStatus status;
    std::vector<uint64_t> downloadedBlockHashes;
    std::chrono::steady_clock::time_point submittedAt;
    std::size_t successfulAcksReceived{0};
  };

  struct OffloadState {
    MigrationStatus status;
    std::chrono::steady_clock::time_point submittedAt;
  };

  std::unique_ptr<tt::messaging::IKafkaProducer> requestProducer;
  std::unique_ptr<tt::messaging::IKafkaConsumer> ackConsumer;
  std::unique_ptr<tt::messaging::IKafkaProducer> downloadRequestProducer;
  std::unique_ptr<tt::messaging::IKafkaConsumer> downloadAckConsumer;
  std::unique_ptr<tt::messaging::IKafkaProducer> offloadRequestProducer;
  std::unique_ptr<tt::messaging::IKafkaConsumer> offloadAckConsumer;
  LayerToPartition layerToPartition;
  std::size_t migrationWorkerPoolSize;
  std::chrono::milliseconds timeout;
  std::chrono::milliseconds sweepInterval;
  int drainPollMs;

  // All state is protected by this mutex.
  // This is temporary solution until performance becomes a priority.
  mutable std::mutex mtx;
  std::unordered_map<uint64_t, MigrationState> migrations;
  std::unordered_map<uint64_t, DownloadState> downloads;
  std::unordered_map<uint64_t, OffloadState> offloads;

  std::atomic<bool> running{false};
  std::thread drainThread;
  std::chrono::steady_clock::time_point lastSweep{};
};

}  // namespace tt::services
