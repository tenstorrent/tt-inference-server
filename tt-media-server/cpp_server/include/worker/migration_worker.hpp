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
  std::string brokers;
  std::string topic;
  std::string group_id;
};

class MigrationWorker {
 public:

  explicit MigrationWorker(MigrationWorkerConfig config);

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
  void processOffloadRequest(const std::string& message, auto receiveTime);

  MigrationWorkerConfig config_;
  std::unique_ptr<tt::messaging::KafkaConsumer> consumer_;
  std::atomic<bool> running_{false};
  std::thread workerThread_;
};
}
