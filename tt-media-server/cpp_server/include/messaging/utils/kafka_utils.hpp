// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <librdkafka/rdkafka.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "utils/logger.hpp"

namespace tt::messaging::kafka_utils {

inline constexpr std::size_t kRdkafkaErrstrBytes = 512;

inline void trimNullTerminated(std::string& s) {
  s.resize(std::strlen(s.c_str()));
}

inline bool setConfigOrLog(rd_kafka_conf_t* conf, const char* name,
                           const char* value) {
  char errstr[kRdkafkaErrstrBytes];
  if (rd_kafka_conf_set(conf, name, value, errstr, sizeof(errstr)) !=
      RD_KAFKA_CONF_OK) {
    TT_LOG_ERROR("[Kafka] config {}={} failed: {}", name, value, errstr);
    return false;
  }
  return true;
}

inline rd_kafka_t* newKafkaHandle(rd_kafka_type_t type, rd_kafka_conf_t* conf,
                                  std::string& errStr) {
  errStr.assign(kRdkafkaErrstrBytes, '\0');
  rd_kafka_t* handle = rd_kafka_new(type, conf, errStr.data(), errStr.size());
  if (!handle) {
    trimNullTerminated(errStr);
  } else {
    errStr.clear();
  }
  return handle;
}

/**
 * Best-effort synchronous topic creation via the admin API. Returns true if
 * the topic exists at partition count >= numPartitions on completion (either
 * because we created it or because it already existed).
 *
 * Useful for tests that need N partitions up-front rather than relying on
 * broker auto-creation (which defaults to num.partitions=1). The `brokers`
 * string is the same bootstrap.servers value used by producers/consumers.
 * `timeoutMs` bounds the round-trip to the broker.
 */
inline bool createTopicWithPartitions(const std::string& brokers,
                                      const std::string& topic,
                                      int32_t numPartitions,
                                      int timeoutMs = 10 * 1000) {
  rd_kafka_conf_t* conf = rd_kafka_conf_new();
  if (!conf) {
    TT_LOG_ERROR("[Kafka admin] rd_kafka_conf_new failed");
    return false;
  }
  if (!setConfigOrLog(conf, "bootstrap.servers", brokers.c_str())) {
    rd_kafka_conf_destroy(conf);
    return false;
  }

  std::string errStr;
  rd_kafka_t* handle = newKafkaHandle(RD_KAFKA_PRODUCER, conf, errStr);
  if (!handle) {
    TT_LOG_ERROR("[Kafka admin] rd_kafka_new failed: {}", errStr);
    return false;
  }

  rd_kafka_NewTopic_t* newTopic = rd_kafka_NewTopic_new(
      topic.c_str(), numPartitions, /*replication_factor=*/1, nullptr, 0);
  if (!newTopic) {
    TT_LOG_ERROR("[Kafka admin] rd_kafka_NewTopic_new failed");
    rd_kafka_destroy(handle);
    return false;
  }

  rd_kafka_queue_t* queue = rd_kafka_queue_new(handle);
  rd_kafka_AdminOptions_t* options =
      rd_kafka_AdminOptions_new(handle, RD_KAFKA_ADMIN_OP_CREATETOPICS);
  rd_kafka_CreateTopics(handle, &newTopic, 1, options, queue);

  bool ok = false;
  rd_kafka_event_t* event = rd_kafka_queue_poll(queue, timeoutMs);
  if (!event) {
    TT_LOG_ERROR("[Kafka admin] CreateTopics timed out after {}ms", timeoutMs);
  } else {
    const rd_kafka_CreateTopics_result_t* result =
        rd_kafka_event_CreateTopics_result(event);
    std::size_t count = 0;
    const rd_kafka_topic_result_t** results =
        rd_kafka_CreateTopics_result_topics(result, &count);
    for (std::size_t i = 0; i < count; ++i) {
      const rd_kafka_resp_err_t err = rd_kafka_topic_result_error(results[i]);
      if (err == RD_KAFKA_RESP_ERR_NO_ERROR ||
          err == RD_KAFKA_RESP_ERR_TOPIC_ALREADY_EXISTS) {
        ok = true;
      } else {
        TT_LOG_ERROR("[Kafka admin] create '{}' failed: {}",
                     rd_kafka_topic_result_name(results[i]),
                     rd_kafka_topic_result_error_string(results[i]));
      }
    }
    rd_kafka_event_destroy(event);
  }

  rd_kafka_AdminOptions_destroy(options);
  rd_kafka_NewTopic_destroy(newTopic);
  rd_kafka_queue_destroy(queue);
  rd_kafka_destroy(handle);
  return ok;
}

}  // namespace tt::messaging::kafka_utils
