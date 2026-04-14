// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <optional>
#include <string>
#include <vector>

namespace tt::domain {

/**
 * Function definition within a tool.
 */
struct FunctionDefinition {
  std::string name;
  std::optional<std::string> description;
  Json::Value parameters;  // JSON schema object

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

/**
 * Tool definition (currently only supports function tools).
 */
struct Tool {
  std::string type = "function";
  FunctionDefinition function;

  static Tool fromJson(const Json::Value& json) {
    Tool tool;
    if (json.isMember("type") && !json["type"].isNull()) {
      tool.type = json["type"].asString();
    }
    if (json.isMember("function") && !json["function"].isNull()) {
      tool.function = FunctionDefinition::fromJson(json["function"]);
    }
    return tool;
  }

  Json::Value toJson() const {
    Json::Value json;
    json["type"] = type;
    json["function"] = function.toJson();
    return json;
  }
};

/**
 * Tool choice specification (controls which tools the model can call).
 */
struct ToolChoice {
  std::string type;  // "none", "auto", "required", or "function"
  std::optional<std::string>
      function_name;  // Set when type="function" for specific tool

  static ToolChoice fromJson(const Json::Value& json) {
    ToolChoice choice;
    if (json.isString()) {
      choice.type = json.asString();
    } else if (json.isObject()) {
      if (json.isMember("type") && !json["type"].isNull()) {
        choice.type = json["type"].asString();
      }
      if (json.isMember("function") && json["function"].isObject() &&
          json["function"].isMember("name")) {
        choice.function_name = json["function"]["name"].asString();
      }
    }
    return choice;
  }

  Json::Value toJson() const {
    if (type == "none" || type == "auto" || type == "required") {
      return Json::Value(type);
    } else {
      Json::Value json;
      json["type"] = type;
      if (function_name.has_value()) {
        json["function"]["name"] = function_name.value();
      }
      return json;
    }
  }
};

/**
 * A function call made by the model.
 */
struct FunctionCall {
  std::string name;
  std::string arguments;  // JSON-encoded string

  static FunctionCall fromJson(const Json::Value& json) {
    FunctionCall call;
    if (json.isMember("name") && !json["name"].isNull()) {
      call.name = json["name"].asString();
    }
    if (json.isMember("arguments") && !json["arguments"].isNull()) {
      call.arguments = json["arguments"].asString();
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

/**
 * A tool call made by the model in a response.
 */
struct ToolCall {
  std::string id;
  std::string type = "function";
  FunctionCall function;

  static ToolCall fromJson(const Json::Value& json) {
    ToolCall call;
    if (json.isMember("id") && !json["id"].isNull()) {
      call.id = json["id"].asString();
    }
    if (json.isMember("type") && !json["type"].isNull()) {
      call.type = json["type"].asString();
    }
    if (json.isMember("function") && !json["function"].isNull()) {
      call.function = FunctionCall::fromJson(json["function"]);
    }
    return call;
  }

  Json::Value toJson() const {
    Json::Value json;
    json["id"] = id;
    json["type"] = type;
    json["function"] = function.toJson();
    return json;
  }
};

}  // namespace tt::domain
