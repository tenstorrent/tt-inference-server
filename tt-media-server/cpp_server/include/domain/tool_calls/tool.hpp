#pragma once

#include <json/json.h>

#include <string>

#include "function_definition.hpp"

namespace tt::domain::tool_calls {

struct Tool {
  std::string type = "function";  // For now, only function tools are supported
  FunctionDefinition functionDefinition;

  static Tool fromJson(const Json::Value& json) {
    Tool tool;
    if (const auto& typeVal = json["type"]; !typeVal.isNull()) {
      tool.type = typeVal.asString();
    }
    if (const auto& funcVal = json["function"]; !funcVal.isNull()) {
      tool.functionDefinition = FunctionDefinition::fromJson(funcVal);
    }
    return tool;
  }

  Json::Value toJson() const {
    Json::Value json;
    json["type"] = type;
    json["function"] = functionDefinition.toJson();
    return json;
  }

  void writeTo(std::ostream& out) const {
    out << "{\"type\":\"" << type << "\",\"function\":";
    functionDefinition.writeTo(out);
    out << "}";
  }
};
}  // namespace tt::domain::tool_calls