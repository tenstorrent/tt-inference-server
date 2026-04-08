// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <string>
#include <string_view>
#include <memory>

namespace tt::messaging {

    /**
     * Configuration for Kafka producer.
     */
    struct KafkaProducerConfig {
      std::string brokers;
      std::string topic;
      /** If non-empty, used as Kafka message key for partition selection. */
      std::string message_key;
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
       * @param errorMessage Optional output parameter for error details. If provided
       *                     and send fails, contains human-readable error description.
       *
       * @return true if message was successfully queued for sending, false otherwise
       *
       * @note This is asynchronous - the message may not have been delivered when
       *       this function returns. Use flush() or destructor for guaranteed delivery.
       */
      bool sendCopy(std::string_view payload, std::string* errorMessage = nullptr);
    
     private:
      struct Impl;
      std::unique_ptr<Impl> impl_;
    };
}
