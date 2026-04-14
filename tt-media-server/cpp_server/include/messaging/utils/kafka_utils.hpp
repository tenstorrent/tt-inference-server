// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <librdkafka/rdkafka.h>

#include <cstddef>
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

}  // namespace tt::messaging::kafka_utils
