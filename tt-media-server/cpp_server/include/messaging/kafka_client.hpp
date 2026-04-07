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
  std::string brokers;  ///< Comma-separated list of broker addresses (e.g., "localhost:9092")
  std::string topic;    ///< Topic name to produce messages to
};

/**
 * Minimal Kafka producer wrapper around librdkafka.
 *
 * Provides a simple interface for sending string messages to a Kafka topic.
 * Uses the librdkafka C API internally for high-performance message production.
 *
 * Thread-safety: This class is NOT thread-safe. Create separate instances per thread
 * or add external synchronization.
 *
 * Example usage:
 * @code
 *   KafkaProducer producer({
 *     .brokers = "localhost:9092",
 *     .topic = "my-topic"
 *   });
 *
 *   std::string error;
 *   if (!producer.send_copy("Hello Kafka", &error)) {
 *     std::cerr << "Failed to send: " << error << std::endl;
 *   }
 * @endcode
 */
class KafkaProducer {
 public:
  /**
   * Creates a Kafka producer and connects to the broker.
   *
   * @param config Configuration containing broker address and topic name
   *
   * @note Constructor does not throw. Check send_copy() return value for errors.
   */
  explicit KafkaProducer(KafkaProducerConfig config);

  /**
   * Destructor flushes pending messages and cleans up resources.
   *
   * Blocks for up to 10 seconds to flush any queued messages before destroying
   * the producer handle.
   */
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
   * @param error_message Optional output parameter for error details. If provided
   *                      and send fails, contains human-readable error description.
   *
   * @return true if message was successfully queued for sending, false otherwise
   *
   * @note This is asynchronous - the message may not have been delivered when
   *       this function returns. Use flush() or destructor for guaranteed delivery.
   */
  bool send_copy(std::string_view payload, std::string* error_message = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;  ///< PIMPL to hide librdkafka types from header
};

/**
 * Configuration for Kafka consumer.
 */
struct KafkaConsumerConfig {
  std::string brokers;   ///< Comma-separated list of broker addresses (e.g., "localhost:9092")
  std::string topic;     ///< Topic name to consume messages from
  std::string group_id;  ///< Consumer group ID for coordinated consumption
};

/**
 * Minimal Kafka consumer wrapper around librdkafka.
 *
 * Provides a simple polling interface for consuming string messages from a Kafka topic.
 * Uses the high-level consumer API with automatic partition assignment and offset management.
 *
 * Consumer groups: Multiple consumers with the same group_id will automatically
 * share partitions. Each message is delivered to only one consumer in the group.
 *
 * Offset management: Offsets are committed automatically. Consumer will resume
 * from last committed position after restart.
 *
 * Thread-safety: This class is NOT thread-safe. Create separate instances per thread
 * or add external synchronization.
 *
 * Example usage:
 * @code
 *   KafkaConsumer consumer({
 *     .brokers = "localhost:9092",
 *     .topic = "my-topic",
 *     .group_id = "my-consumer-group"
 *   });
 *
 *   while (running) {
 *     auto msg = consumer.poll_payload(1000);  // 1 second timeout
 *     if (msg.has_value()) {
 *       process_message(*msg);
 *     }
 *   }
 * @endcode
 */
class KafkaConsumer {
 public:
  /**
   * Creates a Kafka consumer and subscribes to the topic.
   *
   * @param config Configuration containing broker address, topic name, and consumer group
   *
   * @note Constructor does not throw. Check poll_payload() return value for errors.
   *       Subscription happens in constructor - consumer is ready to poll immediately.
   */
  explicit KafkaConsumer(KafkaConsumerConfig config);

  /**
   * Destructor closes the consumer and leaves the consumer group.
   *
   * Triggers rebalancing in the consumer group so other consumers can take over
   * this consumer's partitions.
   */
  ~KafkaConsumer();

  KafkaConsumer(const KafkaConsumer&) = delete;
  KafkaConsumer& operator=(const KafkaConsumer&) = delete;

  /**
   * Polls for the next message from Kafka.
   *
   * Blocks for up to timeout_ms milliseconds waiting for a message. Returns
   * immediately if a message is available.
   *
   * @param timeout_ms Maximum time to wait for a message in milliseconds
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
  std::optional<std::string> poll_payload(int timeout_ms);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;  ///< PIMPL to hide librdkafka types from header
};

}  // namespace tt::messaging
