#pragma once

#include <json/json.h>

#include <optional>
#include <stdexcept>
#include <string>

namespace tt::domain::tool_calls {

struct ToolChoice {
  std::string type;
  std::optional<std::string> function;

  static ToolChoice fromJson(const Json::Value& json) {
    ToolChoice choice;
    if (json.isString()) {
      choice.type = json.asString();
    } else if (json.isObject()) {
      if (!json.isMember("type") || !json["type"].isString()) {
        throw std::invalid_argument(
            "tool_choice object must have a string 'type' field");
      }
      choice.type = json["type"].asString();
      if (json.isMember("function") && json["function"].isObject() &&
          json["function"].isMember("name")) {
        choice.function = json["function"]["name"].asString();
      }
    } else {
      throw std::invalid_argument("tool_choice must be a string or an object");
    }
    return choice;
  }

  Json::Value toJson() const {
    if (type == "function") {
      Json::Value json;
      json["type"] = type;
      if (function.has_value()) {
        json["function"] = function.value();
      }
      return json;
    }
    return Json::Value(type);
  }
};

}  // namespace tt::domain::tool_calls
