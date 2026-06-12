// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "messaging/kafka_consumer.hpp"
#include "transport/i_transfer_engine.hpp"

namespace tt::worker {

/**
 * Configuration for MigrationWorker.
 */
struct MigrationWorkerConfig {
  std::string brokers;
  std::string topic;
  std::string group_id;
  /// Max KV cache size for a single transfer. The receive buffer is
  /// pre-allocated to this size at startup. Default 64MB.
  std::size_t max_transfer_size = 64 * 1024 * 1024;
};

/**
 * Parsed migration request from a Kafka message.
 */
struct MigrationRequest {
  std::string action;
  std::string sessionId;
  uint64_t migrationId = 0;
  std::string sourceHost;
  uint16_t sourcePort = 0;
  uint64_t sourceSegmentAddr = 0;
  uint64_t kvSizeBytes = 0;
  uint32_t sourceSlotId = 0;
  uint32_t numCachedTokens = 0;
  int64_t timestampUs = 0;
};

class MigrationWorker {
 public:
  /**
   * Construct a MigrationWorker.
   *
   * @param config  Kafka consumer configuration.
   * @param engine  Optional transfer engine for executing RDMA migrations.
   *               When null, requests are logged but no transfer is attempted.
   */
  explicit MigrationWorker(
      MigrationWorkerConfig config,
      std::shared_ptr<tt::transport::ITransferEngine> engine = nullptr);

  ~MigrationWorker();

  MigrationWorker(const MigrationWorker&) = delete;
  MigrationWorker& operator=(const MigrationWorker&) = delete;

  /**
   * Starts the worker thread that polls Kafka for messages.
   *
   * Safe to call multiple times (subsequent calls are no-ops).
   */
  void start();

  /**
   * Stops the worker thread gracefully.
   *
   * Signals the worker thread to exit and waits for it to finish.
   * Safe to call multiple times (subsequent calls are no-ops).
   */
  void stop();

 private:
  void consumerLoop();

  void processOffloadRequest(const std::string& message,
                             std::chrono::system_clock::time_point receiveTime);

  MigrationRequest parseRequest(const std::string& message);

  /**
   * Execute an RDMA pull from the source host.
   * No-op if no transfer engine is configured.
   */
  void executeMigration(const MigrationRequest& req);

  MigrationWorkerConfig config_;
  std::unique_ptr<tt::messaging::KafkaConsumer> consumer_;
  std::shared_ptr<tt::transport::ITransferEngine> engine_;
  std::atomic<bool> running_{false};
  std::thread workerThread_;

  /// Pre-allocated receive buffer (avoids per-transfer alloc + page faults).
  std::vector<uint8_t> recvBuffer_;
  bool bufferRegistered_{false};
};
}  // namespace tt::worker
