// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <tt_llm_engine/scheduler/migration_client_interface.hpp>

#include "messaging/i_kafka_consumer.hpp"
#include "messaging/i_kafka_producer.hpp"

namespace tt::services {

/**
 * Factory for creating migration clients.
 *
 * Returns a MigrationClientInterface that only implements the single migrate()
 * path for ALLOCATE prefix copies. Burst methods throw std::runtime_error.
 */
class MigrationClientFactory {
 public:
  /**
   * Create a Kafka-backed migration client.
   *
   * @param requestProducer Kafka producer for migration requests
   * @param ackConsumer Kafka consumer for migration acknowledgments
   * @param migrationTimeout Per-migration ACK timeout (default 60s)
   * @param shutdownTimeout Max drain wait on shutdown (default 30s)
   * @return MigrationClientInterface for use with PrefillScheduler
   */
  static std::unique_ptr<tt_llm_engine::scheduler::MigrationClientInterface>
  createKafkaClient(
      std::unique_ptr<tt::messaging::IKafkaProducer> requestProducer,
      std::unique_ptr<tt::messaging::IKafkaConsumer> ackConsumer,
      std::chrono::milliseconds migrationTimeout = std::chrono::seconds(60),
      std::chrono::milliseconds shutdownTimeout = std::chrono::seconds(30));

  /**
   * Create a Kafka-backed migration client with topic configuration.
   *
   * @param brokers Kafka broker list (e.g., "localhost:9092")
   * @param requestTopic Topic for migration requests
   * @param ackTopic Topic for migration acknowledgments
   * @param groupId Consumer group ID
   * @param migrationTimeout Per-migration ACK timeout (default 60s)
   * @param shutdownTimeout Max drain wait on shutdown (default 30s)
   * @return MigrationClientInterface for use with PrefillScheduler
   */
  static std::unique_ptr<tt_llm_engine::scheduler::MigrationClientInterface>
  createKafkaClient(
      const std::string& brokers, const std::string& requestTopic,
      const std::string& ackTopic, const std::string& groupId,
      std::chrono::milliseconds migrationTimeout = std::chrono::seconds(60),
      std::chrono::milliseconds shutdownTimeout = std::chrono::seconds(30));
};

}  // namespace tt::services
