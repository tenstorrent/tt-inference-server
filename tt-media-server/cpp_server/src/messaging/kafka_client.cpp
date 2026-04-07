// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

/**
 * @file kafka_client.cpp
 * @brief Kafka producer and consumer implementation using librdkafka.
 *
 * This file provides thin wrappers around librdkafka's C API to enable
 * message-based communication between distributed components. It's designed
 * for simplicity and ease of use while maintaining good performance.
 *
 * Key design decisions:
 * - Uses PIMPL pattern to hide librdkafka types from public API
 * - Automatically handles resource cleanup via RAII
 * - Logs errors using project's logging framework
 * - Producer uses MSG_F_COPY for safety (caller can free data immediately)
 * - Consumer uses high-level API with automatic partition assignment
 */

#include "messaging/kafka_client.hpp"

#include <librdkafka/rdkafka.h>

#include "utils/logger.hpp"

namespace tt::messaging {

namespace {

// Thread-local error buffer for librdkafka API calls
char errstr[512];

/**
 * Helper function to set Kafka configuration and log errors.
 *
 * @param conf Kafka configuration object
 * @param name Configuration parameter name
 * @param value Configuration parameter value
 * @return true if configuration was set successfully, false otherwise
 */
bool setConfOrLog(rd_kafka_conf_t* conf, const char* name, const char* value) {
  if (rd_kafka_conf_set(conf, name, value, errstr, sizeof(errstr)) !=
      RD_KAFKA_CONF_OK) {
    TT_LOG_ERROR("[Kafka] config {}={} failed: {}", name, value, errstr);
    return false;
  }
  return true;
}

}  // namespace

struct KafkaProducer::Impl {
  rd_kafka_t* rk{nullptr};
  rd_kafka_topic_t* rkt{nullptr};

  ~Impl() {
    if (rkt) {
      rd_kafka_topic_destroy(rkt);
      rkt = nullptr;
    }
    if (rk) {
      rd_kafka_flush(rk, 10 * 1000);
      rd_kafka_destroy(rk);
      rk = nullptr;
    }
  }
};

KafkaProducer::KafkaProducer(KafkaProducerConfig config) : impl_(std::make_unique<Impl>()) {
  rd_kafka_conf_t* conf = rd_kafka_conf_new();
  if (!conf) {
    TT_LOG_ERROR("[Kafka] rd_kafka_conf_new failed");
    return;
  }
  if (!setConfOrLog(conf, "bootstrap.servers", config.brokers.c_str())) {
    rd_kafka_conf_destroy(conf);
    return;
  }

  // rd_kafka_new() takes ownership of conf on success, leaves it on failure
  rd_kafka_t* rk =
      rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
  if (!rk) {
    TT_LOG_ERROR("[Kafka] rd_kafka_new (producer) failed: {}", errstr);
    rd_kafka_conf_destroy(conf);  // Only destroy on failure
    return;
  }
  // Do NOT destroy conf here - ownership transferred to rk

  rd_kafka_topic_t* rkt = rd_kafka_topic_new(rk, config.topic.c_str(), nullptr);
  if (!rkt) {
    TT_LOG_ERROR("[Kafka] rd_kafka_topic_new failed: {}",
                 rd_kafka_err2str(rd_kafka_last_error()));
    rd_kafka_destroy(rk);
    return;
  }

  impl_->rk = rk;
  impl_->rkt = rkt;
}

KafkaProducer::~KafkaProducer() = default;

bool KafkaProducer::send_copy(std::string_view payload,
                              std::string* error_message) {
  if (!impl_ || !impl_->rk || !impl_->rkt) {
    if (error_message) {
      *error_message = "Kafka producer not initialized";
    }
    return false;
  }

  if (rd_kafka_produce(
          impl_->rkt, RD_KAFKA_PARTITION_UA, RD_KAFKA_MSG_F_COPY,
          const_cast<char*>(payload.data()), payload.size(), nullptr, 0,
          nullptr) != 0) {

    std::string err = rd_kafka_err2str(rd_kafka_last_error());
    
    TT_LOG_ERROR("[Kafka] rd_kafka_produce failed: {}", err);
    if (error_message) {
      *error_message = std::move(err);
    }
    return false;
  }

  rd_kafka_poll(impl_->rk, 0);
  return true;
}

struct KafkaConsumer::Impl {
  rd_kafka_t* rk{nullptr};

  ~Impl() {
    if (rk) {
      rd_kafka_consumer_close(rk);
      rd_kafka_destroy(rk);
      rk = nullptr;
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
  if (!setConfOrLog(conf, "bootstrap.servers", config.brokers.c_str()) ||
      !setConfOrLog(conf, "group.id", config.group_id.c_str()) ||
      !setConfOrLog(conf, "auto.offset.reset", "earliest") ||
      !setConfOrLog(conf, "enable.partition.eof", "false")) {
    rd_kafka_conf_destroy(conf);
    return;
  }

  // rd_kafka_new() takes ownership of conf on success, leaves it on failure
  rd_kafka_t* rk =
      rd_kafka_new(RD_KAFKA_CONSUMER, conf, errstr, sizeof(errstr));
  if (!rk) {
    TT_LOG_ERROR("[Kafka] rd_kafka_new (consumer) failed: {}", errstr);
    rd_kafka_conf_destroy(conf);  // Only destroy on failure
    return;
  }
  // Do NOT destroy conf here - ownership transferred to rk

  rd_kafka_poll_set_consumer(rk);

  rd_kafka_topic_partition_list_t* sub = rd_kafka_topic_partition_list_new(1);
  rd_kafka_topic_partition_list_add(sub, config.topic.c_str(),
                                    RD_KAFKA_PARTITION_UA);
  rd_kafka_resp_err_t err = rd_kafka_subscribe(rk, sub);
  rd_kafka_topic_partition_list_destroy(sub);

  if (err) {
    TT_LOG_ERROR("[Kafka] rd_kafka_subscribe failed: {}",
                 rd_kafka_err2str(err));
    rd_kafka_destroy(rk);
    return;
  }

  impl_->rk = rk;
}

KafkaConsumer::~KafkaConsumer() = default;

std::optional<std::string> KafkaConsumer::poll_payload(int timeout_ms) {
  if (!impl_ || !impl_->rk) {
    return std::nullopt;
  }

  rd_kafka_message_t* msg =
      rd_kafka_consumer_poll(impl_->rk, timeout_ms);
  if (!msg) {
    return std::nullopt;
  }

  struct MsgGuard {
    rd_kafka_message_t* m;
    ~MsgGuard() { rd_kafka_message_destroy(m); }
  } guard{msg};

  if (msg->err) {
    if (msg->err != RD_KAFKA_RESP_ERR__PARTITION_EOF &&
        msg->err != RD_KAFKA_RESP_ERR__TIMED_OUT) {
      TT_LOG_WARN("[Kafka] consumer poll error: {}",
                  rd_kafka_message_errstr(msg));
    }
    return std::nullopt;
  }

  if (!msg->payload || msg->len == 0) {
    return std::nullopt;
  }

  return std::string(static_cast<const char*>(msg->payload), msg->len);
}

}  // namespace tt::messaging
