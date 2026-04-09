// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "messaging/kafka_producer.hpp"

#include <librdkafka/rdkafka.h>
#include <string>

#include "messaging/utils/kafka_utils.hpp"
#include "utils/logger.hpp"

namespace tt::messaging {

struct KafkaProducer::Impl {
  rd_kafka_t* kafka_handle{nullptr};
  rd_kafka_topic_t* topic_handle{nullptr};

  ~Impl() {
    if (topic_handle) {
      rd_kafka_topic_destroy(topic_handle);
      topic_handle = nullptr;
    }
    if (kafka_handle) {
      rd_kafka_flush(kafka_handle, 80 * 1000);
      rd_kafka_destroy(kafka_handle);
      kafka_handle = nullptr;
    }
  }
};

KafkaProducer::KafkaProducer(KafkaProducerConfig config)
    : impl_(std::make_unique<Impl>()) {
  rd_kafka_conf_t* conf = rd_kafka_conf_new();
  if (!conf) {
    TT_LOG_ERROR("[Kafka] rd_kafka_conf_new failed");
    return;
  }
  if (!kafka_utils::setConfigOrLog(conf, "bootstrap.servers", config.brokers.c_str())) {
    rd_kafka_conf_destroy(conf);
    return;
  }

  kafka_utils::setConfigOrLog(conf, "linger.ms", "0");
  kafka_utils::setConfigOrLog(conf, "compression.type", "none");
  kafka_utils::setConfigOrLog(conf, "socket.nagle.disable", "true");

  std::string errStr;
  rd_kafka_t* kafkaHandle =
      kafka_utils::newKafkaHandle(RD_KAFKA_PRODUCER, conf, errStr);
  if (!kafkaHandle) {
    TT_LOG_ERROR("[Kafka] rd_kafka_new (producer) failed: {}", errStr);
    rd_kafka_conf_destroy(conf);
    return;
  }

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
}

KafkaProducer::~KafkaProducer() = default;

bool KafkaProducer::send(std::string_view payload, std::string* errorMessage) {
  if (!impl_ || !impl_->kafka_handle || !impl_->topic_handle) {
    if (errorMessage) {
      *errorMessage = "Kafka producer not initialized";
    }
    return false;
  }

  if (rd_kafka_produce(impl_->topic_handle, RD_KAFKA_PARTITION_UA, RD_KAFKA_MSG_F_COPY,
                       const_cast<char*>(payload.data()), payload.size(), nullptr, 0,
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

bool KafkaProducer::flush(int timeoutMs, std::string* errorMessage) {
  if (!impl_ || !impl_->kafka_handle) {
    if (errorMessage) {
      *errorMessage = "Kafka producer not initialized";
    }
    return false;
  }

  const auto flushErr = rd_kafka_flush(impl_->kafka_handle, timeoutMs);
  if (flushErr != RD_KAFKA_RESP_ERR_NO_ERROR) {
    if (errorMessage) {
      *errorMessage = rd_kafka_err2str(flushErr);
    }
    TT_LOG_ERROR("[Kafka] rd_kafka_flush failed: {}", rd_kafka_err2str(flushErr));
    return false;
  }
  return true;
}

}  // namespace tt::messaging
