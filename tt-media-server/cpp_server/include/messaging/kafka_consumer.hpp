// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <string>
#include <memory>
#include <optional>

namespace tt::messaging {
    /**
 * Configuration for Kafka consumer.
 */
struct KafkaConsumerConfig {
    std::string brokers; 
    std::string topic;
    std::string group_id;
  };
  
  
  class KafkaConsumer {
   public:
  
    explicit KafkaConsumer(KafkaConsumerConfig config);
    ~KafkaConsumer();
  
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
     * @note Fatal errors are logged but also return nullopt. Check logs for details.
     * @note This method must be called regularly (at least every few seconds) to
     *       maintain the consumer's liveness in the consumer group.
     */
    std::optional<std::string> pollPayload(int timeoutMs);
  
   private:
    struct Impl;
    std::unique_ptr<Impl> impl_; 
  };
  }
  