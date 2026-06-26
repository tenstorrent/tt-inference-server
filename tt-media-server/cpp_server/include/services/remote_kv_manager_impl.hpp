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
   * @param requestProducer  Kafka producer wired to the migration-request
   *   topic. Ownership is taken.
   * @param ackConsumer      Kafka consumer subscribed to the migration-ack
   *   topic, with a unique group.id. Ownership is taken.
   * @param timeout          Max age of an IN_PROGRESS migration before the
   *   sweeper marks it FAILED. Default 60s.
   * @param sweepInterval    How often the drain thread runs the timeout
   *   sweep. Default 5s. Tests can pass a small value to force fast
   *   resolution.
   * @param drainPollMs      Per-iteration poll timeout passed to the
   *   consumer. Default 100ms. Lower values trade CPU for responsiveness.
   */
  RemoteKVManagerImpl(
      std::unique_ptr<tt::messaging::IKafkaProducer> requestProducer,
      std::unique_ptr<tt::messaging::IKafkaConsumer> ackConsumer,
      std::chrono::milliseconds timeout = std::chrono::seconds(60),
      std::chrono::milliseconds sweepInterval = std::chrono::seconds(5),
      int drainPollMs = 100);

  ~RemoteKVManagerImpl() override;

  RemoteKVManagerImpl(const RemoteKVManagerImpl&) = delete;
  RemoteKVManagerImpl& operator=(const RemoteKVManagerImpl&) = delete;

  [[nodiscard]] uint64_t migrate(const MigrationRequest& request) override;
  MigrationStatus getStatus(uint64_t migrationId) const override;

 private:
  void drainLoop();
  // Caller must hold mtx.
  void sweepLocked(std::chrono::steady_clock::time_point now);

  struct MigrationState {
    MigrationStatus status;
    std::chrono::steady_clock::time_point submittedAt;
  };

  std::unique_ptr<tt::messaging::IKafkaProducer> requestProducer;
  std::unique_ptr<tt::messaging::IKafkaConsumer> ackConsumer;
  std::chrono::milliseconds timeout;
  std::chrono::milliseconds sweepInterval;
  int drainPollMs;

  mutable std::mutex mtx;
  std::unordered_map<uint64_t, MigrationState> migrations;

  std::atomic<bool> running{false};
  std::thread drainThread;
  std::chrono::steady_clock::time_point lastSweep{};
};

}  // namespace tt::services
