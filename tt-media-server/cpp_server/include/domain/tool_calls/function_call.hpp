#pragma once

#include <json/json.h>
#include <json/value.h>

#include <string>

namespace tt::domain::tool_calls {

struct FunctionCall {
  std::string name;
  Json::Value arguments;

  static FunctionCall fromJson(const Json::Value& json) {
    FunctionCall call;
    if (json.isMember("name") && !json["name"].isNull()) {
      call.name = json["name"].asString();
    }
    if (json.isMember("arguments") && !json["arguments"].isNull()) {
      call.arguments = json["arguments"];
    }
    return call;
  }

  Json::Value toJson() const {
    Json::Value json;
    json["name"] = name;
    json["arguments"] = arguments;
    return json;
  }
};
}  // namespace tt::domain::tool_calls