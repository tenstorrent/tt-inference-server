// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/migration_client_factory.hpp"

#include "messaging/kafka_consumer.hpp"
#include "messaging/kafka_producer.hpp"

namespace tt::services {

std::unique_ptr<RemoteKVManagerAdapter> MigrationClientFactory::createKafkaClient(
    const std::string& brokers, const std::string& requestTopic,
    const std::string& ackTopic, const std::string& groupId,
    uint32_t layersPerChunk) {
  auto producer =
      std::make_unique<tt::messaging::KafkaProducer>(brokers, requestTopic);
  auto consumer =
      std::make_unique<tt::messaging::KafkaConsumer>(brokers, ackTopic, groupId);

  return createKafkaClient(std::move(producer), std::move(consumer),
                           layersPerChunk);
}

}  // namespace tt::services
