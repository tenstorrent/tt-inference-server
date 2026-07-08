// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "messaging/i_kafka_consumer.hpp"

namespace tt::messaging {

struct KafkaConsumerConfig {
  std::string brokers;
  std::string topic;
  std::string group_id;
  // When set, the consumer uses rd_kafka_assign() to pin itself to this
  // partition instead of joining the group's rebalance. This guarantees
  // "worker k always reads partition k" semantics but disables automatic
  // failover and group-managed offset commits. auto.offset.reset=latest
  // still applies, so requests in flight during a restart are lost --
  // callers must reconcile via a higher-level timeout/sweeper.
  std::optional<int32_t> partition;
};

class KafkaConsumer : public IKafkaConsumer {
 public:
  explicit KafkaConsumer(KafkaConsumerConfig config);
  ~KafkaConsumer() override;

  KafkaConsumer(const KafkaConsumer&) = delete;
  KafkaConsumer& operator=(const KafkaConsumer&) = delete;

  /**
   * Polls for the next message from Kafka.
   *
   * Blocks for up to timeout_ms milliseconds waiting for a message. Returns
   * immediately if a message is available.
   *
   * @param timeoutMs Maximum time to wait for a message in milliseconds
   *
   * @return Message payload as string if available, nullopt if:
   *         - Timeout occurred with no messages
   *         - Non-fatal error (e.g., partition EOF)
   *         - Message has no payload
   *
   * @note Fatal errors are logged but also return nullopt. Check logs for
   * details.
   * @note This method must be called regularly (at least every few seconds) to
   *       maintain the consumer's liveness in the consumer group.
   */
  std::optional<std::string> receive(int timeoutMs) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::messaging
