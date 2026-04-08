// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace tt::messaging {

/**
 * Configuration for Kafka producer.
 */
struct KafkaProducerConfig {
  std::string brokers;
  std::string topic;
};

class KafkaProducer {
 public:
  explicit KafkaProducer(KafkaProducerConfig config);
  ~KafkaProducer();

  KafkaProducer(const KafkaProducer&) = delete;
  KafkaProducer& operator=(const KafkaProducer&) = delete;

  /**
   * Sends a message to the configured Kafka topic.
   *
   * Makes a copy of the payload, so the caller can safely free/modify the
   * string after this call returns.
   *
   * @param payload Message content to send
   * @param errorMessage Optional output parameter for error details. If
   * provided and send fails, contains human-readable error description.
   *
   * @return true if message was successfully queued for sending, false
   * otherwise
   *
   * @note This is asynchronous - the message may not have been delivered when
   *       this function returns. Use flush() or destructor for guaranteed
   * delivery.
   */
  bool sendCopy(std::string_view payload, std::string* errorMessage = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

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
   * @note Fatal errors are logged but also return nullopt. Check logs for
   * details.
   * @note This method must be called regularly (at least every few seconds) to
   *       maintain the consumer's liveness in the consumer group.
   */
  std::optional<std::string> pollPayload(int timeoutMs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::messaging
