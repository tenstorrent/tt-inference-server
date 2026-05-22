#pragma once

#include <json/json.h>
#include <json/value.h>

#include <string>

#include "function_call.hpp"

namespace tt::domain::tool_calls {
struct ToolCall {
  std::string id;
  std::string type = "function";
  FunctionCall functionCall;

  static ToolCall fromJson(const Json::Value& json) {
    ToolCall call;
    if (json.isMember("id") && !json["id"].isNull()) {
      call.id = json["id"].asString();
    }
    if (json.isMember("type") && !json["type"].isNull()) {
      call.type = json["type"].asString();
    }
    if (json.isMember("function") && !json["function"].isNull()) {
      call.functionCall = FunctionCall::fromJson(json["function"]);
    }
    return call;
  }

  Json::Value toJson() const {
    Json::Value json;
    json["id"] = id;
    json["type"] = type;
    json["function"] = functionCall.toJson();
    return json;
  }
};
}  // namespace tt::domain::tool_calls