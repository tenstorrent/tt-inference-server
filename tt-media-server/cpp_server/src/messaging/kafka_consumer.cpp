// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "messaging/kafka_consumer.hpp"

#include <librdkafka/rdkafka.h>

#include <string>

#include "messaging/utils/kafka_utils.hpp"
#include "utils/logger.hpp"

namespace tt::messaging {

struct KafkaConsumer::Impl {
  rd_kafka_t* kafka_handle{nullptr};

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

  if (!kafka_utils::setConfigOrLog(conf, "bootstrap.servers",
                                   config.brokers.c_str()) ||
      !kafka_utils::setConfigOrLog(conf, "group.id", config.group_id.c_str()) ||
      !kafka_utils::setConfigOrLog(conf, "auto.offset.reset", "latest") ||
      !kafka_utils::setConfigOrLog(conf, "enable.partition.eof", "false") ||
      !kafka_utils::setConfigOrLog(conf, "fetch.wait.max.ms", "1") ||
      !kafka_utils::setConfigOrLog(conf, "fetch.min.bytes", "1") ||
      !kafka_utils::setConfigOrLog(conf, "enable.auto.commit", "true") ||
      !kafka_utils::setConfigOrLog(conf, "auto.commit.interval.ms", "100")) {
    rd_kafka_conf_destroy(conf);
    return;
  }

  std::string errStr;
  rd_kafka_t* kafkaHandle =
      kafka_utils::newKafkaHandle(RD_KAFKA_CONSUMER, conf, errStr);
  if (!kafkaHandle) {
    TT_LOG_ERROR("[Kafka] rd_kafka_new (consumer) failed: {}", errStr);
    rd_kafka_conf_destroy(conf);
    return;
  }

  rd_kafka_poll_set_consumer(kafkaHandle);

  rd_kafka_topic_partition_list_t* partitionList =
      rd_kafka_topic_partition_list_new(1);
  const int32_t partition = config.partition.value_or(RD_KAFKA_PARTITION_UA);
  rd_kafka_topic_partition_list_add(partitionList, config.topic.c_str(),
                                    partition);
  rd_kafka_resp_err_t err = RD_KAFKA_RESP_ERR_NO_ERROR;
  if (config.partition.has_value()) {
    err = rd_kafka_assign(kafkaHandle, partitionList);
  } else {
    err = rd_kafka_subscribe(kafkaHandle, partitionList);
  }
  rd_kafka_topic_partition_list_destroy(partitionList);

  if (err) {
    TT_LOG_ERROR(
        "[Kafka] {} failed: {}",
        config.partition.has_value() ? "rd_kafka_assign" : "rd_kafka_subscribe",
        rd_kafka_err2str(err));
    rd_kafka_destroy(kafkaHandle);
    return;
  }

  impl_->kafka_handle = kafkaHandle;
}

KafkaConsumer::~KafkaConsumer() = default;

std::optional<std::string> KafkaConsumer::receive(int timeoutMs) {
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
