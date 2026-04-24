// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <gtest/gtest.h>
#include <json/json.h>

#include "domain/chat_completion_request.hpp"
#include "domain/chat_message.hpp"
#include "domain/tool_calls/tool.hpp"
#include "domain/tool_calls/tool_choice.hpp"

using namespace tt::domain;
using namespace tt::domain::tool_calls;

class NamedFunctionCallingTest : public ::testing::Test {
 protected:
  Json::Value createBasicRequestJson() {
    Json::Value json;
    json["model"] = "test-model";

    Json::Value msg;
    msg["role"] = "user";
    msg["content"] = "What's the weather in San Francisco?";
    json["messages"].append(msg);

    return json;
  }

  Json::Value createWeatherTool() {
    Json::Value tool;
    tool["type"] = "function";
    tool["function"]["name"] = "get_weather";
    tool["function"]["description"] = "Get weather for a location";

    Json::Value params;
    params["type"] = "object";
    params["properties"]["location"]["type"] = "string";
    params["properties"]["location"]["description"] = "The city name";
    params["required"].append("location");
    tool["function"]["parameters"] = params;

    return tool;
  }

  Json::Value createTimeTool() {
    Json::Value tool;
    tool["type"] = "function";
    tool["function"]["name"] = "get_time";
    tool["function"]["description"] = "Get current time";

    Json::Value params;
    params["type"] = "object";
    params["properties"]["timezone"]["type"] = "string";
    params["required"].append("timezone");
    tool["function"]["parameters"] = params;

    return tool;
  }
};

TEST_F(NamedFunctionCallingTest, ToolChoiceFunctionParsing) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createWeatherTool());
  json["tools"].append(createTimeTool());

  // Set tool_choice to specific function
  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"]["name"] = "get_weather";
  json["tool_choice"] = toolChoice;

  auto request = ChatCompletionRequest::fromJson(json, 1);

  // Verify tool_choice is parsed correctly
  ASSERT_TRUE(request.tool_choice.has_value());
  EXPECT_EQ(request.tool_choice->type, "function");
  ASSERT_TRUE(request.tool_choice->function.has_value());
  EXPECT_EQ(request.tool_choice->function.value(), "get_weather");
}

TEST_F(NamedFunctionCallingTest, ToolChoiceFunctionValidation) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createWeatherTool());

  // Valid: function exists in tools
  Json::Value validToolChoice;
  validToolChoice["type"] = "function";
  validToolChoice["function"]["name"] = "get_weather";
  json["tool_choice"] = validToolChoice;

  EXPECT_NO_THROW({
    auto request = ChatCompletionRequest::fromJson(json, 1);
    EXPECT_TRUE(request.tool_choice.has_value());
    EXPECT_EQ(request.tool_choice->type, "function");
  });

  // Invalid: function doesn't exist in tools
  Json::Value invalidToolChoice;
  invalidToolChoice["type"] = "function";
  invalidToolChoice["function"]["name"] = "nonexistent_function";
  json["tool_choice"] = invalidToolChoice;

  EXPECT_THROW(
      { auto request = ChatCompletionRequest::fromJson(json, 2); },
      std::invalid_argument);
}

TEST_F(NamedFunctionCallingTest, ToolChoiceFunctionRequiresFunctionName) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createWeatherTool());

  // Invalid: type is "function" but no function name provided
  Json::Value toolChoice;
  toolChoice["type"] = "function";
  // Missing: toolChoice["function"]["name"]
  json["tool_choice"] = toolChoice;

  EXPECT_THROW(
      { auto request = ChatCompletionRequest::fromJson(json, 1); },
      std::invalid_argument);
}

TEST_F(NamedFunctionCallingTest, ToolChoiceFunctionCreatesStructuredOutput) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createWeatherTool());

  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"]["name"] = "get_weather";
  json["tool_choice"] = toolChoice;

  auto request = ChatCompletionRequest::fromJson(json, 1);

  // Convert to LLMRequest
  auto llmRequest = request.toLLMRequest();

  // Verify tool_choice info is copied
  ASSERT_TRUE(llmRequest.tool_choice_type.has_value());
  EXPECT_EQ(llmRequest.tool_choice_type.value(), "function");

  ASSERT_TRUE(llmRequest.tool_choice_function_name.has_value());
  EXPECT_EQ(llmRequest.tool_choice_function_name.value(), "get_weather");

  // Verify response_format is set to JSON_SCHEMA
  ASSERT_TRUE(llmRequest.response_format.has_value());
  EXPECT_EQ(llmRequest.response_format->type,
            tt::config::ResponseFormatType::JSON_SCHEMA);
  EXPECT_TRUE(llmRequest.response_format->strict);

  // Verify schema name matches function name
  ASSERT_TRUE(llmRequest.response_format->json_schema_name.has_value());
  EXPECT_EQ(llmRequest.response_format->json_schema_name.value(),
            "get_weather");

  // Verify schema string contains the parameters
  ASSERT_TRUE(llmRequest.response_format->json_schema_str.has_value());
  std::string schema = llmRequest.response_format->json_schema_str.value();
  EXPECT_TRUE(schema.find("location") != std::string::npos);
  EXPECT_TRUE(schema.find("string") != std::string::npos);
}

TEST_F(NamedFunctionCallingTest, MultipleToolsSelectSpecificFunction) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createWeatherTool());
  json["tools"].append(createTimeTool());

  // Request specific function
  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"]["name"] = "get_time";
  json["tool_choice"] = toolChoice;

  auto request = ChatCompletionRequest::fromJson(json, 1);
  auto llmRequest = request.toLLMRequest();

  // Verify the correct function's schema is used
  ASSERT_TRUE(llmRequest.response_format.has_value());
  ASSERT_TRUE(llmRequest.response_format->json_schema_name.has_value());
  EXPECT_EQ(llmRequest.response_format->json_schema_name.value(), "get_time");

  ASSERT_TRUE(llmRequest.response_format->json_schema_str.has_value());
  std::string schema = llmRequest.response_format->json_schema_str.value();
  EXPECT_TRUE(schema.find("timezone") != std::string::npos);
  // Should NOT contain parameters from other function
  EXPECT_TRUE(schema.find("location") == std::string::npos);
}

// Main function for running tests
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
