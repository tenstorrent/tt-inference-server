// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/migration_client_factory.hpp"

#include "messaging/kafka_consumer.hpp"
#include "messaging/kafka_producer.hpp"
#include "services/remote_kv_manager_adapter.hpp"
#include "services/remote_kv_manager_impl.hpp"

namespace tt::services {

std::unique_ptr<tt_llm_engine::scheduler::MigrationClientInterface>
MigrationClientFactory::createKafkaClient(
    std::unique_ptr<tt::messaging::IKafkaProducer> requestProducer,
    std::unique_ptr<tt::messaging::IKafkaConsumer> ackConsumer,
    std::chrono::milliseconds migrationTimeout,
    std::chrono::milliseconds shutdownTimeout) {
  auto kvManager = std::make_unique<RemoteKVManagerImpl>(
      std::move(requestProducer), std::move(ackConsumer), migrationTimeout);
  return std::make_unique<RemoteKVManagerAdapter>(std::move(kvManager),
                                                  shutdownTimeout);
}

std::unique_ptr<tt_llm_engine::scheduler::MigrationClientInterface>
MigrationClientFactory::createKafkaClient(
    const std::string& brokers, const std::string& requestTopic,
    const std::string& ackTopic, const std::string& groupId,
    std::chrono::milliseconds migrationTimeout,
    std::chrono::milliseconds shutdownTimeout) {
  auto producer = std::make_unique<tt::messaging::KafkaProducer>(
      tt::messaging::KafkaProducerConfig{brokers, requestTopic});
  auto consumer = std::make_unique<tt::messaging::KafkaConsumer>(
      tt::messaging::KafkaConsumerConfig{brokers, ackTopic, groupId,
                                         std::nullopt});

  return createKafkaClient(std::move(producer), std::move(consumer),
                           migrationTimeout, shutdownTimeout);
}

}  // namespace tt::services
