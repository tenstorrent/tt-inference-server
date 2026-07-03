// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>
#include <tt_llm_engine/scheduler/migration_client_interface.hpp>

#include "services/remote_kv_manager_adapter.hpp"
#include "services/remote_kv_manager_impl.hpp"

namespace tt::services {

/**
 * Factory for creating migration clients.
 *
 * This provides a unified way to create either the MPI-based
 * MigrationLayerClientAdapter (from tt-llm-engine) or the Kafka-based
 * RemoteKVManagerAdapter for PrefillScheduler integration.
 */
class MigrationClientFactory {
 public:
  /**
   * Create a Kafka-backed migration client using RemoteKVManagerAdapter.
   *
   * @param requestProducer Kafka producer for migration requests
   * @param ackConsumer Kafka consumer for migration acknowledgments
   * @param layersPerChunk Number of layers per chunk (from SchedulerParams)
   * @param timeout Migration timeout (default 60s)
   * @return unique_ptr to MigrationClientInterface-compatible adapter
   *
   * Usage in driver:
   *   auto producer = std::make_unique<KafkaProducer>(...);
   *   auto consumer = std::make_unique<KafkaConsumer>(...);
   *   auto ml = MigrationClientFactory::createKafkaClient(
   *       std::move(producer), std::move(consumer), params.layers_per_chunk);
   *   PrefillScheduler scheduler(h2d_cfg, ack_cfg, params, std::move(ml));
   */
  static std::unique_ptr<RemoteKVManagerAdapter> createKafkaClient(
      std::unique_ptr<tt::messaging::IKafkaProducer> requestProducer,
      std::unique_ptr<tt::messaging::IKafkaConsumer> ackConsumer,
      uint32_t layersPerChunk,
      std::chrono::milliseconds timeout = std::chrono::seconds(60)) {
    auto kvManager = std::make_unique<RemoteKVManagerImpl>(
        std::move(requestProducer), std::move(ackConsumer), timeout);
    return std::make_unique<RemoteKVManagerAdapter>(std::move(kvManager),
                                                    layersPerChunk);
  }

  /**
   * Create a Kafka-backed migration client with topic configuration.
   *
   * @param brokers Kafka broker list (e.g., "localhost:9092")
   * @param requestTopic Topic for migration requests
   * @param ackTopic Topic for migration acknowledgments
   * @param groupId Consumer group ID
   * @param layersPerChunk Number of layers per chunk
   * @return unique_ptr to MigrationClientInterface-compatible adapter
   */
  static std::unique_ptr<RemoteKVManagerAdapter> createKafkaClient(
      const std::string& brokers, const std::string& requestTopic,
      const std::string& ackTopic, const std::string& groupId,
      uint32_t layersPerChunk);
};

}  // namespace tt::services
