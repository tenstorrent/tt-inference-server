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

class ChatCompletionRequestToolTest : public ::testing::Test {
 protected:
  // Helper to create a basic valid request JSON
  Json::Value createBasicRequestJson() {
    Json::Value json;
    json["model"] = "test-model";

    Json::Value msg;
    msg["role"] = "user";
    msg["content"] = "Hello";
    json["messages"].append(msg);

    return json;
  }

  // Helper to create a tool
  Json::Value createToolJson(const std::string& name,
                             const std::string& description) {
    Json::Value tool;
    tool["type"] = "function";
    tool["function"]["name"] = name;
    tool["function"]["description"] = description;

    Json::Value params;
    params["type"] = "object";
    params["properties"]["location"]["type"] = "string";
    params["required"].append("location");
    tool["function"]["parameters"] = params;

    return tool;
  }
};

// ==================== Tool Parsing Tests ====================

TEST_F(ChatCompletionRequestToolTest, ParseRequestWithTools) {
  Json::Value json = createBasicRequestJson();

  json["tools"].append(createToolJson("get_weather", "Get weather info"));
  json["tools"].append(createToolJson("get_time", "Get current time"));

  auto request = ChatCompletionRequest::fromJson(json, 1);

  ASSERT_TRUE(request.tools.has_value());
  EXPECT_EQ(request.tools->size(), 2);
  EXPECT_EQ(request.tools->at(0).functionDefinition.name, "get_weather");
  EXPECT_EQ(request.tools->at(1).functionDefinition.name, "get_time");
}

// ==================== tool_choice Tests ====================

TEST_F(ChatCompletionRequestToolTest, ToolChoiceNone) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather"));
  json["tool_choice"] = "none";

  auto request = ChatCompletionRequest::fromJson(json, 1);

  // When tool_choice is "none", tools should be cleared
  EXPECT_FALSE(request.tools.has_value() || request.tools->empty())
      << "Tools should be empty when tool_choice is 'none'";

  // tool_choice should still be parsed
  ASSERT_TRUE(request.tool_choice.has_value());
  EXPECT_EQ(request.tool_choice->type, "none");
}

TEST_F(ChatCompletionRequestToolTest, ToolChoiceAuto) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather"));
  json["tool_choice"] = "auto";

  auto request = ChatCompletionRequest::fromJson(json, 1);

  // With tool_choice "auto", tools should remain
  ASSERT_TRUE(request.tools.has_value());
  EXPECT_FALSE(request.tools->empty());

  ASSERT_TRUE(request.tool_choice.has_value());
  EXPECT_EQ(request.tool_choice->type, "auto");
}

TEST_F(ChatCompletionRequestToolTest, ToolChoiceNoneWithoutTools) {
  Json::Value json = createBasicRequestJson();
  json["tool_choice"] = "none";
  // No tools provided

  // Should not throw - tool_choice "none" without tools is valid
  EXPECT_NO_THROW({
    auto request = ChatCompletionRequest::fromJson(json, 1);
    EXPECT_TRUE(request.tool_choice.has_value());
    EXPECT_EQ(request.tool_choice->type, "none");
  });
}

TEST_F(ChatCompletionRequestToolTest, ToolChoiceNoneWithEmptyToolsArray) {
  Json::Value json = createBasicRequestJson();
  json["tools"] = Json::arrayValue;
  json["tool_choice"] = "none";

  auto request = ChatCompletionRequest::fromJson(json, 1);

  EXPECT_TRUE(request.tool_choice.has_value());
  EXPECT_EQ(request.tool_choice->type, "none");
}

// Main function for running tests
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
