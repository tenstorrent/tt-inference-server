// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <gtest/gtest.h>
#include <json/json.h>

#include <iostream>

#include "domain/chat_completion_request.hpp"
#include "domain/tool_calls/tool_choice.hpp"

using namespace tt::domain;
using namespace tt::domain::tool_calls;

class ToolChoiceParsingTest : public ::testing::Test {
 protected:
  Json::Value createBasicRequest() {
    Json::Value json;
    json["model"] = "test-model";

    Json::Value msg;
    msg["role"] = "user";
    msg["content"] = "Test message";
    json["messages"].append(msg);

    // Add a tool
    Json::Value tool;
    tool["type"] = "function";
    tool["function"]["name"] = "get_weather";
    tool["function"]["description"] = "Get weather";
    Json::Value params;
    params["type"] = "object";
    params["properties"]["location"]["type"] = "string";
    tool["function"]["parameters"] = params;
    json["tools"].append(tool);

    return json;
  }

  void printJson(const std::string& label, const Json::Value& json) {
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "  ";
    std::cout << label << ":\n" << Json::writeString(writer, json) << std::endl;
  }
};

TEST_F(ToolChoiceParsingTest, CorrectFunctionFormat) {
  Json::Value json = createBasicRequest();

  // CORRECT FORMAT for tool_choice with function
  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"]["name"] = "get_weather";
  json["tool_choice"] = toolChoice;

  printJson("Valid tool_choice JSON", toolChoice);

  // Should parse successfully
  EXPECT_NO_THROW({
    auto request = ChatCompletionRequest::fromJson(json, 1);
    ASSERT_TRUE(request.tool_choice.has_value());
    EXPECT_EQ(request.tool_choice->type, "function");
    ASSERT_TRUE(request.tool_choice->function.has_value());
    EXPECT_EQ(request.tool_choice->function.value(), "get_weather");
  });
}

TEST_F(ToolChoiceParsingTest, MissingFunctionObject) {
  Json::Value json = createBasicRequest();

  // WRONG: type is "function" but no function object
  Json::Value toolChoice;
  toolChoice["type"] = "function";
  // Missing: toolChoice["function"]
  json["tool_choice"] = toolChoice;

  printJson("Invalid: missing function object", toolChoice);

  EXPECT_THROW(
      { auto request = ChatCompletionRequest::fromJson(json, 1); },
      std::invalid_argument);
}

TEST_F(ToolChoiceParsingTest, MissingFunctionName) {
  Json::Value json = createBasicRequest();

  // WRONG: function object exists but no name
  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"] = Json::objectValue;  // Empty object
  json["tool_choice"] = toolChoice;

  printJson("Invalid: empty function object", toolChoice);

  EXPECT_THROW(
      { auto request = ChatCompletionRequest::fromJson(json, 1); },
      std::invalid_argument);
}

TEST_F(ToolChoiceParsingTest, NullFunctionName) {
  Json::Value json = createBasicRequest();

  // WRONG: function name is null
  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"]["name"] = Json::nullValue;
  json["tool_choice"] = toolChoice;

  printJson("Invalid: null function name", toolChoice);

  EXPECT_THROW(
      { auto request = ChatCompletionRequest::fromJson(json, 1); },
      std::invalid_argument);
}

TEST_F(ToolChoiceParsingTest, EmptyFunctionName) {
  Json::Value json = createBasicRequest();

  // WRONG: function name is empty string
  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"]["name"] = "";
  json["tool_choice"] = toolChoice;

  printJson("Invalid: empty function name", toolChoice);

  EXPECT_THROW(
      { auto request = ChatCompletionRequest::fromJson(json, 1); },
      std::invalid_argument);
}

TEST_F(ToolChoiceParsingTest, StringFormatAuto) {
  Json::Value json = createBasicRequest();

  // CORRECT: string format for "auto"
  json["tool_choice"] = "auto";

  EXPECT_NO_THROW({
    auto request = ChatCompletionRequest::fromJson(json, 1);
    ASSERT_TRUE(request.tool_choice.has_value());
    EXPECT_EQ(request.tool_choice->type, "auto");
    EXPECT_FALSE(request.tool_choice->function.has_value());
  });
}

TEST_F(ToolChoiceParsingTest, StringFormatNone) {
  Json::Value json = createBasicRequest();

  // CORRECT: string format for "none"
  json["tool_choice"] = "none";

  EXPECT_NO_THROW({
    auto request = ChatCompletionRequest::fromJson(json, 1);
    ASSERT_TRUE(request.tool_choice.has_value());
    EXPECT_EQ(request.tool_choice->type, "none");
    EXPECT_FALSE(request.tool_choice->function.has_value());
  });
}

TEST_F(ToolChoiceParsingTest, FunctionNotInTools) {
  Json::Value json = createBasicRequest();

  // Function name doesn't match any tool
  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"]["name"] = "nonexistent_function";
  json["tool_choice"] = toolChoice;

  printJson("Invalid: function not in tools", toolChoice);

  EXPECT_THROW(
      {
        try {
          auto request = ChatCompletionRequest::fromJson(json, 1);
        } catch (const std::invalid_argument& e) {
          std::cout << "Expected error: " << e.what() << std::endl;
          throw;
        }
      },
      std::invalid_argument);
}

TEST_F(ToolChoiceParsingTest, CorrectFormatFromRawJSON) {
  // Test parsing from a raw JSON string (simulates API request)
  std::string jsonStr = R"({
    "model": "test-model",
    "messages": [{"role": "user", "content": "What's the weather?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    }],
    "tool_choice": {
      "type": "function",
      "function": {
        "name": "get_weather"
      }
    }
  })";

  Json::Reader reader;
  Json::Value json;
  ASSERT_TRUE(reader.parse(jsonStr, json)) << "Failed to parse JSON string";

  printJson("Full request JSON", json);
  printJson("tool_choice portion", json["tool_choice"]);

  EXPECT_NO_THROW({
    auto request = ChatCompletionRequest::fromJson(json, 1);
    ASSERT_TRUE(request.tool_choice.has_value());
    EXPECT_EQ(request.tool_choice->type, "function");
    ASSERT_TRUE(request.tool_choice->function.has_value());
    EXPECT_EQ(request.tool_choice->function.value(), "get_weather");
  });
}

// Main function for running tests
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
