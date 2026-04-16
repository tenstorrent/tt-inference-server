#pragma once

#include <json/json.h>

#include <string>

#include "function_definiton.hpp"

namespace tt::domain::tool_calls {

struct Tool {
  std::string type = "function";  // For now, only function tools are supported
  FunctionDefinition functionDefinition;

  static Tool fromJson(const Json::Value& json) {
    Tool tool;
    if (json.isMember("type") && !json["type"].isNull()) {
      tool.type = json["type"].asString();
    }
    if (json.isMember("function") && !json["function"].isNull()) {
      tool.functionDefinition = FunctionDefinition::fromJson(json["function"]);
    }
    return tool;
  }

  Json::Value toJson() const {
    Json::Value json;
    json["type"] = type;
    json["function"] = functionDefinition.toJson();
    return json;
  }
};
}  // namespace tt::domain::tool_calls