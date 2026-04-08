// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "messaging/kafka_producer.hpp"
#include <librdkafka/rdkafka.h>
#include "utils/logger.hpp"

namespace tt::messaging {

namespace {

    // Thread-local error buffer for librdkafka API calls
    char errstr[512];
    
    
    bool setConfigOrLog(rd_kafka_conf_t* conf, const char* name, const char* value) {
      if (rd_kafka_conf_set(conf, name, value, errstr, sizeof(errstr)) !=
          RD_KAFKA_CONF_OK) {
        TT_LOG_ERROR("[Kafka] config {}={} failed: {}", name, value, errstr);
        return false;
      }
      return true;
    }
    
    }  // namespace
    
    struct KafkaProducer::Impl {
      rd_kafka_t* kafka_handle{nullptr};        // Main Kafka producer connection
      rd_kafka_topic_t* topic_handle{nullptr};  // Specific topic to publish to
      std::string message_key;
    
      ~Impl() {
        if (topic_handle) {
          rd_kafka_topic_destroy(topic_handle);
          topic_handle = nullptr;
        }
        if (kafka_handle) {
          rd_kafka_flush(kafka_handle, 10 * 1000);   // Flush before destruction
          rd_kafka_destroy(kafka_handle);
          kafka_handle = nullptr;
        }
      }
    };
    
    KafkaProducer::KafkaProducer(KafkaProducerConfig config) : impl_(std::make_unique<Impl>()) {
      rd_kafka_conf_t* conf = rd_kafka_conf_new();
      if (!conf) {
        TT_LOG_ERROR("[Kafka] rd_kafka_conf_new failed");
        return;
      }
      if (!setConfigOrLog(conf, "bootstrap.servers", config.brokers.c_str())) {
        rd_kafka_conf_destroy(conf);
        return;
      }
    
      // Low-latency producer settings
      setConfigOrLog(conf, "linger.ms", "0");
      setConfigOrLog(conf, "compression.type", "none");
      setConfigOrLog(conf, "socket.nagle.disable", "true");
    
      // rd_kafka_new() takes ownership of conf on success, leaves it on failure
      rd_kafka_t* kafkaHandle =
          rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
      if (!kafkaHandle) {
        TT_LOG_ERROR("[Kafka] rd_kafka_new (producer) failed: {}", errstr);
        rd_kafka_conf_destroy(conf);  // Only destroy on failure
        return;
      }
      // Do NOT destroy conf here - ownership transferred to kafkaHandle
    
      rd_kafka_topic_t* topicHandle =
          rd_kafka_topic_new(kafkaHandle, config.topic.c_str(), nullptr);
      if (!topicHandle) {
        TT_LOG_ERROR("[Kafka] rd_kafka_topic_new failed: {}",
                     rd_kafka_err2str(rd_kafka_last_error()));
        rd_kafka_destroy(kafkaHandle);
        return;
      }
    
      impl_->kafka_handle = kafkaHandle;
      impl_->topic_handle = topicHandle;
      impl_->message_key = std::move(config.message_key);
    }
    
    KafkaProducer::~KafkaProducer() = default;
    
    bool KafkaProducer::sendCopy(std::string_view payload, std::string* errorMessage) {
      if (!impl_ || !impl_->kafka_handle || !impl_->topic_handle) {
        if (errorMessage) {
          *errorMessage = "Kafka producer not initialized";
        }
        return false;
      }
    
      const std::string& key = impl_->message_key;
      const void* keyPtr = key.empty() ? nullptr : key.data();
      size_t keyLen = key.empty() ? 0 : key.size();
    
      if (rd_kafka_produce(
              impl_->topic_handle, RD_KAFKA_PARTITION_UA, RD_KAFKA_MSG_F_COPY,
              const_cast<char*>(payload.data()), payload.size(),
              keyPtr, keyLen,
              nullptr) != 0) {
    
        std::string err = rd_kafka_err2str(rd_kafka_last_error());
    
        TT_LOG_ERROR("[Kafka] rd_kafka_produce failed: {}", err);
        if (errorMessage) {
          *errorMessage = std::move(err);
        }
        return false;
      }
    
      rd_kafka_poll(impl_->kafka_handle, 0);
      return true;
    }
}    