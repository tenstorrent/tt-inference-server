#pragma once

#include <json/json.h>

#include <optional>
#include <string>

namespace tt::domain::tool_calls {

struct ToolChoice {
  std::string type;
  std::optional<std::string>
      function;  // Used when type = "function", name of the function to call

  static ToolChoice fromJson(const Json::Value& json) {
    ToolChoice choice;
    if (json.isString()) {
      choice.type = json.asString();
    } else if (json.isObject()) {
      if (json.isMember("type") && !json["type"].isNull()) {
        choice.type = json["type"].asString();
      }
      if (json.isMember("function") && json["function"].isObject() &&
          json["function"].isMember("name") &&
          !json["function"]["name"].isNull() &&
          !json["function"]["name"].asString().empty()) {
        choice.function = json["function"]["name"].asString();
      }
    }
    return choice;
  }

  Json::Value toJson() const {
    if (type == "none" || type == "auto" || type == "required") {
      return Json::Value(type);
    } else if (type == "function") {
      Json::Value json;
      json["type"] = type;
      if (function.has_value()) {
        json["function"] = function.value();
      }
      return json;
    }
    return Json::nullValue;
  }
};
}  // namespace tt::domain::tool_calls