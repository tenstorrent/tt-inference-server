#pragma once

#include <json/json.h>
#include <json/value.h>

#include <optional>
#include <string>

namespace tt::domain::tool_calls {

struct FunctionDefinition {
  std::string name;
  std::optional<std::string> description;
  Json::Value parameters;

  static FunctionDefinition fromJson(const Json::Value& json) {
    FunctionDefinition func;
    if (json.isMember("name") && !json["name"].isNull()) {
      func.name = json["name"].asString();
    }
    if (json.isMember("description") && !json["description"].isNull()) {
      func.description = json["description"].asString();
    }
    if (json.isMember("parameters") && !json["parameters"].isNull()) {
      func.parameters = json["parameters"];
    }
    return func;
  }

  Json::Value toJson() const {
    Json::Value json;
    json["name"] = name;
    if (description.has_value()) {
      json["description"] = description.value();
    }
    if (!parameters.isNull()) {
      json["parameters"] = parameters;
    }
    return json;
  }
};

}  // namespace tt::domain::tool_calls