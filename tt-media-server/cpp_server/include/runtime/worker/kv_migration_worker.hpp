// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

#include "messaging/i_kafka_consumer.hpp"
#include "messaging/i_kafka_producer.hpp"
#include "runtime/worker/migration_executor.hpp"
#include "services/remote_kv_manager.hpp"

namespace tt::worker {

/**
 * Kafka consumer that turns MigrationRequestMessages into IMigrationExecutor
 * calls and publishes a MigrationResponseMessage when each migration completes.
 *
 * Worker thread model:
 *   - One thread polls requestConsumer and submits each parsed request to
 *     the executor. The thread returns to the poll loop immediately
 *     (execute() is non-blocking by contract), so a slow migration does
 *     not block dequeue of the next one.
 *   - When the executor invokes its DoneCallback the worker serializes a
 *     MigrationResponseMessage and publishes it via ackProducer. The
 *     callback can fire from any thread, so ack publication is mutex-
 *     protected.
 *
 * Lifecycle: start() spins the polling thread, stop() asks it to exit and
 * joins. Safe to call either repeatedly. Destruction implies stop().
 *
 * Ownership: the worker takes ownership of all three injected
 * dependencies. On destruction it stops the poll thread and then destroys the
 * executor explicitly (before ackMutex / ackProducer tear down), so an async
 * executor's in-flight DoneCallback (-> publishAck) drains against still-valid
 * members. The DoneCallback captures `this`, so the executor must not outlive
 * the worker — which this ordering guarantees.
 */
class KvMigrationWorker {
 public:
  KvMigrationWorker(
      std::unique_ptr<tt::messaging::IKafkaConsumer> requestConsumer,
      std::unique_ptr<tt::messaging::IKafkaProducer> ackProducer,
      std::unique_ptr<IMigrationExecutor> executor, int pollTimeoutMs = 100);

  ~KvMigrationWorker();

  KvMigrationWorker(const KvMigrationWorker&) = delete;
  KvMigrationWorker& operator=(const KvMigrationWorker&) = delete;

  void start();
  void stop();

 private:
  void consumerLoop();
  void publishAck(uint64_t migrationId, tt::services::MigrationStatus status);

  std::unique_ptr<tt::messaging::IKafkaConsumer> requestConsumer;
  std::unique_ptr<tt::messaging::IKafkaProducer> ackProducer;
  std::unique_ptr<IMigrationExecutor> executor;
  int pollTimeoutMs;

  std::mutex ackMutex;
  std::atomic<bool> running{false};
  std::thread workerThread;
};

}  // namespace tt::worker
