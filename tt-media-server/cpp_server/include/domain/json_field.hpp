// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <stdexcept>
#include <string>

namespace tt::domain::json_field {

inline bool getBool(const Json::Value& val, const char* field) {
  if (!val.isBool())
    throw std::invalid_argument(std::string(field) + " must be a boolean");
  return val.asBool();
}

inline int getInt(const Json::Value& val, const char* field) {
  if (!val.isIntegral())
    throw std::invalid_argument(std::string(field) + " must be an integer");
  return val.asInt();
}

inline float getFloat(const Json::Value& val, const char* field) {
  if (!val.isNumeric())
    throw std::invalid_argument(std::string(field) + " must be a number");
  return val.asFloat();
}

inline std::string getString(const Json::Value& val, const char* field) {
  if (!val.isString())
    throw std::invalid_argument(std::string(field) + " must be a string");
  return val.asString();
}

}  // namespace tt::domain::json_field
