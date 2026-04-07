// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "messaging/kafka_client.hpp"

namespace tt::worker {

/**
 * Configuration for MigrationWorker.
 */
struct MigrationWorkerConfig {
  std::string brokers;    ///< Kafka broker addresses (e.g., "localhost:9092")
  std::string topic;      ///< Topic to consume from (e.g., "session-offload")
  std::string group_id;   ///< Consumer group ID (e.g., "migration-workers")
};

/**
 * Worker that listens to Kafka for session offload requests.
 *
 * This worker runs in a separate thread and continuously polls Kafka for
 * offload messages from the main server. When a message is received, it
 * prints the message and calculates the messaging overhead.
 *
 * Purpose: Measure the latency between producer sending and consumer receiving
 * offload signals. This validates whether Kafka is fast enough for real-time
 * session migration coordination.
 *
 * Thread-safety: This class manages its own thread internally. Safe to call
 * start()/stop() from any thread.
 *
 * Example usage:
 * @code
 *   MigrationWorker worker({
 *     .brokers = "localhost:9092",
 *     .topic = "session-offload",
 *     .group_id = "migration-workers"
 *   });
 *
 *   worker.start();  // Begins polling in background thread
 *   // ... worker runs ...
 *   worker.stop();   // Graceful shutdown
 * @endcode
 */
class MigrationWorker {
 public:
  /**
   * Creates a MigrationWorker with the given configuration.
   *
   * @param config Kafka connection and topic configuration
   *
   * @note Constructor creates the Kafka consumer but does not start polling.
   *       Call start() to begin consuming messages.
   */
  explicit MigrationWorker(MigrationWorkerConfig config);

  /**
   * Destructor ensures worker thread is stopped and joined.
   */
  ~MigrationWorker();

  MigrationWorker(const MigrationWorker&) = delete;
  MigrationWorker& operator=(const MigrationWorker&) = delete;

  /**
   * Starts the worker thread that polls Kafka for messages.
   *
   * Creates a background thread that continuously polls the Kafka topic.
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
  /**
   * Main loop that polls Kafka and processes messages.
   * Runs in worker thread until stopped.
   */
  void consumerLoop();

  /**
   * Processes a single offload request message.
   *
   * Parses the JSON message, extracts the timestamp, calculates overhead,
   * and logs the results.
   *
   * Expected message format:
   * {
   *   "timestamp_us": 1234567890,
   *   "action": "offload",
   *   "current_session_count": 850,
   *   "max_sessions": 1000
   * }
   *
   * @param message JSON string containing offload request
   */
  void processOffloadRequest(const std::string& message);

  MigrationWorkerConfig config_;
  std::unique_ptr<tt::messaging::KafkaConsumer> consumer_;
  std::atomic<bool> running_{false};
  std::thread workerThread_;
};

}  // namespace tt::worker
