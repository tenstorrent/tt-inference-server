// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace tt::dynamo::json_value {

inline std::string compactJson(const Json::Value& value) {
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  writer["emitUTF8"] = true;
  return Json::writeString(writer, value);
}

inline Json::Value parseJson(const std::string& data) {
  Json::Value out;
  Json::CharReaderBuilder builder;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  std::string errs;
  if (!reader->parse(data.data(), data.data() + data.size(), &out, &errs)) {
    throw std::runtime_error("invalid JSON: " + errs);
  }
  return out;
}

inline std::optional<uint64_t> asOptionalUInt64(const Json::Value& value) {
  if (value.isUInt64()) return value.asUInt64();
  if (value.isUInt()) return value.asUInt();
  if (value.isInt64() && value.asInt64() >= 0) {
    return static_cast<uint64_t>(value.asInt64());
  }
  if (value.isString()) {
    try {
      return static_cast<uint64_t>(std::stoull(value.asString()));
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

inline std::optional<uint32_t> asOptionalUInt32(const Json::Value& value) {
  auto parsed = asOptionalUInt64(value);
  if (!parsed.has_value() ||
      *parsed > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(*parsed);
}

inline std::optional<uint64_t> optionalUInt64(const Json::Value& obj,
                                              const char* field) {
  if (!obj.isMember(field) || obj[field].isNull()) return std::nullopt;
  return asOptionalUInt64(obj[field]);
}

inline std::optional<uint32_t> optionalUInt32(const Json::Value& obj,
                                              const char* field) {
  if (!obj.isMember(field) || obj[field].isNull()) return std::nullopt;
  return asOptionalUInt32(obj[field]);
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
