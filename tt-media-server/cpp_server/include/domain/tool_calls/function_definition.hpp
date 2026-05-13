#pragma once

#include <json/json.h>
#include <json/value.h>

#include <optional>
#include <ostream>
#include <string>

namespace tt::domain::tool_calls {

struct FunctionDefinition {
  std::string name;
  std::optional<std::string> description;
  Json::Value parameters;

  static FunctionDefinition fromJson(const Json::Value& json) {
    FunctionDefinition func;
    if (const auto& nameVal = json["name"]; !nameVal.isNull()) {
      func.name = nameVal.asString();
    }
    if (const auto& descVal = json["description"]; !descVal.isNull()) {
      func.description = descVal.asString();
    }
    if (const auto& paramsVal = json["parameters"]; !paramsVal.isNull()) {
      func.parameters = paramsVal;
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

  void writeTo(std::ostream& out) const {
    static Json::StreamWriterBuilder builder = []() {
      Json::StreamWriterBuilder b;
      b["indentation"] = "";
      return b;
    }();
    out << "{\"name\":\"" << name << "\"";
    if (description.has_value()) {
      out << ",\"description\":\"" << description.value() << "\"";
    }
    if (!parameters.isNull()) {
      out << ",\"parameters\":" << Json::writeString(builder, parameters);
    }
    out << "}";
  }
};

}  // namespace tt::domain::tool_calls