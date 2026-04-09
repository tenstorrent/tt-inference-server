// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <string_view>

namespace tt::messaging {

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
  bool send(std::string_view payload, std::string* errorMessage = nullptr);
  /**
   * Flushes queued messages and waits for delivery.
   *
   * @param timeoutMs Maximum time to wait in milliseconds.
   * @param errorMessage Optional output parameter for error details.
   * @return true when all queued messages are delivered before timeout.
   */
  bool flush(int timeoutMs = 10 * 1000, std::string* errorMessage = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::messaging
