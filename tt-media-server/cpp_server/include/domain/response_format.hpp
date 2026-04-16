// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "config/types.hpp"
#include "domain/json_field.hpp"

namespace tt::domain {

using tt::config::ResponseFormatType;

struct ResponseFormat {
  ResponseFormatType type = ResponseFormatType::TEXT;
  std::optional<std::string> json_schema_name;
  std::optional<std::string> json_schema_str;
  bool strict = false;

  static ResponseFormat fromJson(const Json::Value& json) {
    using namespace json_field;
    checkObject(json, "response_format");

    ResponseFormat fmt;
    std::string typeStr = getString(json["type"], "response_format.type");

    if (typeStr == "text") {
      fmt.type = ResponseFormatType::TEXT;
    } else if (typeStr == "json_object") {
      fmt.type = ResponseFormatType::JSON_OBJECT;
    } else if (typeStr == "json_schema") {
      fmt.type = ResponseFormatType::JSON_SCHEMA;

      if (!json.isMember("json_schema") || json["json_schema"].isNull()) {
        throw std::invalid_argument(
            "response_format.json_schema is required when type is "
            "'json_schema'");
      }
      checkObject(json["json_schema"], "response_format.json_schema");
      const auto& schemaObj = json["json_schema"];

      if (schemaObj.isMember("name") && !schemaObj["name"].isNull()) {
        fmt.json_schema_name =
            getString(schemaObj["name"], "response_format.json_schema.name");
      }

      if (schemaObj.isMember("strict") && !schemaObj["strict"].isNull()) {
        fmt.strict =
            getBool(schemaObj["strict"], "response_format.json_schema.strict");
      }

      if (!schemaObj.isMember("schema") || schemaObj["schema"].isNull()) {
        throw std::invalid_argument(
            "response_format.json_schema.schema is required");
      }
      checkObject(schemaObj["schema"], "response_format.json_schema.schema");

      Json::StreamWriterBuilder writer;
      writer["indentation"] = "";
      fmt.json_schema_str = Json::writeString(writer, schemaObj["schema"]);
    } else {
      throw std::invalid_argument(
          "response_format.type must be 'text', 'json_object', or "
          "'json_schema'");
    }

    return fmt;
  }
};

}  // namespace tt::domain
