// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <string>

namespace tt::dynamo::json_value {

inline std::optional<uint64_t> optionalUInt64(const Json::Value& obj,
                                              const char* field) {
  if (!obj.isMember(field) || obj[field].isNull()) return std::nullopt;
  if (obj[field].isUInt64()) return obj[field].asUInt64();
  if (obj[field].isUInt()) return obj[field].asUInt();
  if (obj[field].isString()) {
    try {
      return static_cast<uint64_t>(std::stoull(obj[field].asString()));
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

inline std::optional<uint32_t> optionalUInt32(const Json::Value& obj,
                                              const char* field) {
  auto value = optionalUInt64(obj, field);
  if (!value.has_value() ||
      *value > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(*value);
}

inline std::optional<int> optionalInt(const Json::Value& obj,
                                      const char* field) {
  if (!obj.isMember(field) || obj[field].isNull()) return std::nullopt;
  if (obj[field].isInt()) return obj[field].asInt();
  if (obj[field].isString()) {
    try {
      return std::stoi(obj[field].asString());
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

inline void setOptional(Json::Value& obj, const char* field,
                        const std::optional<int>& value) {
  if (value.has_value()) {
    obj[field] = *value;
  } else {
    obj[field] = Json::Value::null;
  }
}

inline void setOptional(Json::Value& obj, const char* field,
                        const std::optional<uint32_t>& value) {
  if (value.has_value()) {
    obj[field] = *value;
  } else {
    obj[field] = Json::Value::null;
  }
}

inline void setOptional(Json::Value& obj, const char* field,
                        const std::optional<uint64_t>& value) {
  if (value.has_value()) {
    obj[field] = static_cast<Json::UInt64>(*value);
  } else {
    obj[field] = Json::Value::null;
  }
}

}  // namespace tt::dynamo::json_value
