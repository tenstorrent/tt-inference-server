#pragma once

#include <json/json.h>
#include <json/value.h>

#include <optional>
#include <stdexcept>
#include <string>

namespace tt::domain::tool_calls {

struct ToolChoice {
  std::string type;
  std::optional<std::string> function;

  static bool isSpecValidType(const std::string& type) {
    return type == "none" || type == "auto" || type == "required" ||
           type == "function";
  }

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
    if (!isSpecValidType(choice.type)) {
      throw std::invalid_argument(
          "tool_choice 'type' must be one of 'none', 'auto', 'required', or "
          "'function' (got '" +
          choice.type + "')");
    }
    return choice;
  }

  Json::Value toJson() const {
    if (isSpecValidType(type)) {
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
