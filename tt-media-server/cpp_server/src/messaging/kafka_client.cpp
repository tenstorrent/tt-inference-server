// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


#include "messaging/kafka_client.hpp"

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

}

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

  setConfigOrLog(conf, "linger.ms", "0");
  setConfigOrLog(conf, "compression.type", "none");
  setConfigOrLog(conf, "socket.nagle.disable", "true");

  rd_kafka_t* kafka_handle =
      rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
  if (!kafka_handle) {
    TT_LOG_ERROR("[Kafka] rd_kafka_new (producer) failed: {}", errstr);
    rd_kafka_conf_destroy(conf);
    return;
  }
  rd_kafka_topic_t* topic_handle = rd_kafka_topic_new(kafka_handle, config.topic.c_str(), nullptr);
  if (!topic_handle) {
    TT_LOG_ERROR("[Kafka] rd_kafka_topic_new failed: {}",
                 rd_kafka_err2str(rd_kafka_last_error()));
    rd_kafka_destroy(kafka_handle);
    return;
  }

  impl_->kafka_handle = kafka_handle;
  impl_->topic_handle = topic_handle;
}

KafkaProducer::~KafkaProducer() = default;

bool KafkaProducer::sendCopy(std::string_view payload, std::string* errorMessage) {
  if (!impl_ || !impl_->kafka_handle || !impl_->topic_handle) {
    if (errorMessage) {
      *errorMessage = "Kafka producer not initialized";
    }
    return false;
  }

  if (rd_kafka_produce(
          impl_->topic_handle, RD_KAFKA_PARTITION_UA, RD_KAFKA_MSG_F_COPY,
          const_cast<char*>(payload.data()), payload.size(),
          nullptr, 0,
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

  if (!setConfigOrLog(conf, "bootstrap.servers", config.brokers.c_str()) ||
      !setConfigOrLog(conf, "group.id", config.group_id.c_str()) ||
      !setConfigOrLog(conf, "auto.offset.reset", "latest") ||
      !setConfigOrLog(conf, "enable.partition.eof", "false") ||
      !setConfigOrLog(conf, "fetch.wait.max.ms", "1") ||
      !setConfigOrLog(conf, "fetch.min.bytes", "1") ||
      !setConfigOrLog(conf, "enable.auto.commit", "true") ||
      !setConfigOrLog(conf, "auto.commit.interval.ms", "100")) {
    rd_kafka_conf_destroy(conf);
    return;
  }

  rd_kafka_t* kafka_handle =
      rd_kafka_new(RD_KAFKA_CONSUMER, conf, errstr, sizeof(errstr));
  if (!kafka_handle) {
    TT_LOG_ERROR("[Kafka] rd_kafka_new (consumer) failed: {}", errstr);
    rd_kafka_conf_destroy(conf);
    return;
  }

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

}
