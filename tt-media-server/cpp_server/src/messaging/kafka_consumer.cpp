// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "messaging/kafka_consumer.hpp"

#include <librdkafka/rdkafka.h>

#include "utils/logger.hpp"

namespace tt::messaging {

struct KafkaConsumer::Impl {
    rd_kafka_t* kafka_handle{nullptr};  // Main Kafka consumer connection
  
    ~Impl() {
      if (kafka_handle) {
        rd_kafka_consumer_close(kafka_handle);
        rd_kafka_destroy(kafka_handle);
        kafka_handle = nullptr;
      }
    }
  };
  
  KafkaConsumer::KafkaConsumer(KafkaConsumerConfig config)
      : impl_(std::make_unique<Impl>()) {
    rd_kafka_conf_t* conf = rd_kafka_conf_new();
    if (!conf) {
      TT_LOG_ERROR("[Kafka] rd_kafka_conf_new failed");
      return;
    }
  
    // Low-latency consumer configuration for ~1ms overhead
    if (!setConfigOrLog(conf, "bootstrap.servers", config.brokers.c_str()) ||
        !setConfigOrLog(conf, "group.id", config.group_id.c_str()) ||
        !setConfigOrLog(conf, "auto.offset.reset", "latest") ||  // Only read NEW messages, not old ones
        !setConfigOrLog(conf, "enable.partition.eof", "false") ||
        // Critical: reduce fetch wait from default 500ms to 1ms for low latency
        !setConfigOrLog(conf, "fetch.wait.max.ms", "1") ||
        !setConfigOrLog(conf, "fetch.min.bytes", "1") ||
        // Auto-commit to reduce coordination overhead
        !setConfigOrLog(conf, "enable.auto.commit", "true") ||
        !setConfigOrLog(conf, "auto.commit.interval.ms", "100")) {
      rd_kafka_conf_destroy(conf);
      return;
    }
  
    // rd_kafka_new() takes ownership of conf on success, leaves it on failure
    rd_kafka_t* kafka_handle =
        rd_kafka_new(RD_KAFKA_CONSUMER, conf, errstr, sizeof(errstr));
    if (!kafka_handle) {
      TT_LOG_ERROR("[Kafka] rd_kafka_new (consumer) failed: {}", errstr);
      rd_kafka_conf_destroy(conf);  // Only destroy on failure
      return;
    }
    // Do NOT destroy conf here - ownership transferred to kafka_handle
  
    rd_kafka_poll_set_consumer(kafka_handle);
  
    rd_kafka_topic_partition_list_t* subscription_list = rd_kafka_topic_partition_list_new(1);
    rd_kafka_topic_partition_list_add(subscription_list, config.topic.c_str(),
                                      RD_KAFKA_PARTITION_UA);
    rd_kafka_resp_err_t err = rd_kafka_subscribe(kafka_handle, subscription_list);
    rd_kafka_topic_partition_list_destroy(subscription_list);
  
    if (err) {
      TT_LOG_ERROR("[Kafka] rd_kafka_subscribe failed: {}",
                   rd_kafka_err2str(err));
      rd_kafka_destroy(kafka_handle);
      return;
    }
  
    impl_->kafka_handle = kafka_handle;
  }
  
  KafkaConsumer::~KafkaConsumer() = default;
  
  std::optional<std::string> KafkaConsumer::pollPayload(int timeoutMs) {
    if (!impl_ || !impl_->kafka_handle) {
      return std::nullopt;
    }
  
    rd_kafka_message_t* message =
        rd_kafka_consumer_poll(impl_->kafka_handle, timeoutMs);
    if (!message) {
      return std::nullopt;
    }
  
    struct MessageGuard {
      rd_kafka_message_t* msg;
      ~MessageGuard() { rd_kafka_message_destroy(msg); }
    } guard{message};
  
    if (message->err) {
      if (message->err != RD_KAFKA_RESP_ERR__PARTITION_EOF &&
          message->err != RD_KAFKA_RESP_ERR__TIMED_OUT) {
        TT_LOG_WARN("[Kafka] consumer poll error: {}",
                    rd_kafka_message_errstr(message));
      }
      return std::nullopt;
    }
  
    if (!message->payload || message->len == 0) {
      return std::nullopt;
    }
  
    return std::string(static_cast<const char*>(message->payload), message->len);
  }
  
  }  // namespace tt::messaging
  