// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>
#include <json/json.h>

#include "domain/tool_calls/function_definition.hpp"
#include "domain/tool_calls/tool.hpp"

namespace tt::runners {
// Forward declaration of buildToolSchema for testing
Json::Value buildToolSchema(const tt::domain::tool_calls::Tool& tool);
}  // namespace tt::runners

namespace {

// ---------------------------------------------------------------------------
// Tool Schema Construction Tests
// ---------------------------------------------------------------------------

TEST(ToolSchemaTest, BuildToolSchemaForSingleTool) {
  // Create a tool
  tt::domain::tool_calls::FunctionDefinition funcDef;
  funcDef.name = "get_weather";
  funcDef.description = "Get current weather";

  Json::Value params;
  params["type"] = "object";
  params["properties"]["location"]["type"] = "string";
  params["properties"]["units"]["type"] = "string";
  params["properties"]["units"]["enum"].append("celsius");
  params["properties"]["units"]["enum"].append("fahrenheit");
  params["required"].append("location");
  params["additionalProperties"] = false;
  funcDef.parameters = params;

  tt::domain::tool_calls::Tool tool;
  tool.functionDefinition = funcDef;

  // Build schema
  Json::Value schema = tt::runners::buildToolSchema(tool);

  // Verify schema structure
  EXPECT_EQ(schema["type"].asString(), "object");
  EXPECT_EQ(schema["properties"]["name"]["const"].asString(), "get_weather");
  EXPECT_EQ(schema["properties"]["arguments"]["type"].asString(), "object");
  EXPECT_EQ(
      schema["properties"]["arguments"]["properties"]["location"]["type"]
          .asString(),
      "string");
  EXPECT_FALSE(
      schema["properties"]["arguments"]["additionalProperties"].asBool());
  ASSERT_EQ(schema["required"].size(), 2);
  EXPECT_EQ(schema["required"][0].asString(), "name");
  EXPECT_EQ(schema["required"][1].asString(), "arguments");
}

TEST(ToolSchemaTest, BuildWrappedSchemaWithAnyOfForMultipleTools) {
  // Create tools
  std::vector<tt::domain::tool_calls::Tool> tools;

  // Tool 1: get_weather
  tt::domain::tool_calls::FunctionDefinition funcDef1;
  funcDef1.name = "get_weather";
  Json::Value params1;
  params1["type"] = "object";
  params1["properties"]["location"]["type"] = "string";
  params1["required"].append("location");
  params1["additionalProperties"] = false;
  funcDef1.parameters = params1;

  tt::domain::tool_calls::Tool tool1;
  tool1.functionDefinition = funcDef1;
  tools.push_back(tool1);

  // Tool 2: get_time
  tt::domain::tool_calls::FunctionDefinition funcDef2;
  funcDef2.name = "get_time";
  Json::Value params2;
  params2["type"] = "object";
  params2["properties"]["timezone"]["type"] = "string";
  params2["required"].append("timezone");
  params2["additionalProperties"] = false;
  funcDef2.parameters = params2;

  tt::domain::tool_calls::Tool tool2;
  tool2.functionDefinition = funcDef2;
  tools.push_back(tool2);

  // Build wrapped schema with anyOf
  Json::Value wrappedSchema;
  wrappedSchema["type"] = "object";
  Json::Value anyOf(Json::arrayValue);
  for (const auto& tool : tools) {
    anyOf.append(tt::runners::buildToolSchema(tool));
  }
  wrappedSchema["anyOf"] = anyOf;

  // Verify wrapped schema structure
  EXPECT_EQ(wrappedSchema["type"].asString(), "object");
  ASSERT_EQ(wrappedSchema["anyOf"].size(), 2);

  // Verify first tool schema
  EXPECT_EQ(wrappedSchema["anyOf"][0]["properties"]["name"]["const"].asString(),
            "get_weather");
  EXPECT_EQ(wrappedSchema["anyOf"][0]["properties"]["arguments"]["properties"]
                        ["location"]["type"]
                            .asString(),
            "string");

  // Verify second tool schema
  EXPECT_EQ(wrappedSchema["anyOf"][1]["properties"]["name"]["const"].asString(),
            "get_time");
  EXPECT_EQ(wrappedSchema["anyOf"][1]["properties"]["arguments"]["properties"]
                        ["timezone"]["type"]
                            .asString(),
            "string");
}

}  // namespace
