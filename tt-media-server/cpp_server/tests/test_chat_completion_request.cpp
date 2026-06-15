// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <gtest/gtest.h>
#include <json/json.h>

#include <stdexcept>
#include <string>

#include "domain/llm/chat_completion_request.hpp"

using namespace tt::domain;
using namespace tt::domain::llm;

namespace {

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

Json::Value createUserMessage(const std::string& content) {
  Json::Value msg;
  msg["role"] = "user";
  msg["content"] = content;
  return msg;
}

Json::Value createAssistantMessage(const std::string& content) {
  Json::Value msg;
  msg["role"] = "assistant";
  msg["content"] = content;
  return msg;
}

Json::Value createAssistantWithToolCall(const std::string& callId,
                                        const std::string& functionName,
                                        const std::string& arguments) {
  Json::Value msg;
  msg["role"] = "assistant";
  msg["content"] = "";

  Json::Value toolCall;
  toolCall["id"] = callId;
  toolCall["type"] = "function";
  toolCall["function"]["name"] = functionName;
  toolCall["function"]["arguments"] = arguments;
  msg["tool_calls"].append(toolCall);

  return msg;
}

// Helper to create a tool response message
Json::Value createToolMessage(const std::string& toolCallId,
                              const std::string& content) {
  Json::Value msg;
  msg["role"] = "tool";
  msg["tool_call_id"] = toolCallId;
  msg["content"] = content;
  return msg;
}

// Tool Parsing Tests

TEST(ChatCompletionRequestTest, ParseRequestWithTools) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather info"));
  json["tools"].append(createToolJson("get_time", "Get current time"));

  auto request = ChatCompletionRequest::fromJson(json, 1);

  ASSERT_TRUE(request.tools.has_value());
  ASSERT_EQ(request.tools->size(), 2);
  EXPECT_EQ(request.tools->at(0).functionDefinition.name, "get_weather");
  EXPECT_EQ(request.tools->at(1).functionDefinition.name, "get_time");
}

// tool_choice Tests

TEST(ChatCompletionRequestTest, ToolChoiceNone) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather"));
  json["tool_choice"] = "none";

  auto request = ChatCompletionRequest::fromJson(json, 1);

  // When tool_choice is "none", tools should still be parsed
  ASSERT_TRUE(request.tools.has_value());
  EXPECT_FALSE(request.tools->empty())
      << "Tools should be kept when tool_choice is 'none'";

  ASSERT_TRUE(request.tool_choice.has_value());
  EXPECT_EQ(request.tool_choice->type, "none");

  // Verify tool_choice is copied to LLMRequest
  auto llmRequest = request.toLLMRequest();
  ASSERT_TRUE(llmRequest.tool_choice.has_value());
  EXPECT_EQ(llmRequest.tool_choice->type, "none");
}

TEST(ChatCompletionRequestTest, ToolChoiceAuto) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather"));
  json["tool_choice"] = "auto";

  auto request = ChatCompletionRequest::fromJson(json, 1);

  ASSERT_TRUE(request.tools.has_value());
  EXPECT_FALSE(request.tools->empty());
  ASSERT_TRUE(request.tool_choice.has_value());
  EXPECT_EQ(request.tool_choice->type, "auto");
}

TEST(ChatCompletionRequestTest, ToolChoiceNoneWithoutTools) {
  Json::Value json = createBasicRequestJson();
  json["tool_choice"] = "none";

  auto request = ChatCompletionRequest::fromJson(json, 1);
  ASSERT_TRUE(request.tool_choice.has_value());
  EXPECT_EQ(request.tool_choice->type, "none");
}

TEST(ChatCompletionRequestTest, ToolChoiceNoneWithEmptyToolsArray) {
  Json::Value json = createBasicRequestJson();
  json["tools"] = Json::arrayValue;
  json["tool_choice"] = "none";

  auto request = ChatCompletionRequest::fromJson(json, 1);

  ASSERT_TRUE(request.tool_choice.has_value());
  EXPECT_EQ(request.tool_choice->type, "none");
}

TEST(ChatCompletionRequestTest, ToolChoiceAutoWithoutToolsRejected) {
  Json::Value json = createBasicRequestJson();
  json["tool_choice"] = "auto";

  EXPECT_THROW(ChatCompletionRequest::fromJson(json, 1), std::invalid_argument)
      << "Should throw invalid_argument for tool_choice=auto without tools";
}

TEST(ChatCompletionRequestTest, ToolChoiceUnknownStringRejected) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather"));
  json["tool_choice"] = "bogus";

  EXPECT_THROW(ChatCompletionRequest::fromJson(json, 1), std::invalid_argument)
      << "Should throw invalid_argument for unknown tool_choice value";
}

TEST(ChatCompletionRequestTest, ToolChoiceFunction) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather"));
  json["tools"].append(createToolJson("get_time", "Get time"));

  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"]["name"] = "get_weather";
  json["tool_choice"] = toolChoice;

  auto request = ChatCompletionRequest::fromJson(json, 1);

  ASSERT_TRUE(request.tool_choice.has_value());
  EXPECT_EQ(request.tool_choice->type, "function");
  ASSERT_TRUE(request.tool_choice->function.has_value());
  EXPECT_EQ(request.tool_choice->function.value(), "get_weather");

  auto llmRequest = request.toLLMRequest();
  ASSERT_TRUE(llmRequest.tool_choice.has_value());
  EXPECT_EQ(llmRequest.tool_choice->type, "function");
  EXPECT_EQ(llmRequest.tool_choice->function.value(), "get_weather");
}

TEST(ChatCompletionRequestTest, ToolChoiceFunctionMissingNameRejected) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather"));

  Json::Value toolChoice;
  toolChoice["type"] = "function";
  // No "function" field
  json["tool_choice"] = toolChoice;

  EXPECT_THROW(
      {
        try {
          ChatCompletionRequest::fromJson(json, 1);
        } catch (const std::invalid_argument& e) {
          EXPECT_NE(std::string(e.what()).find(
                        "tool_choice.function.name is required"),
                    std::string::npos);
          throw;
        }
      },
      std::invalid_argument)
      << "Should throw when tool_choice=function has no function.name";
}

TEST(ChatCompletionRequestTest, ToolChoiceFunctionUnknownNameRejected) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather"));

  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"]["name"] = "missing_tool";
  json["tool_choice"] = toolChoice;

  EXPECT_THROW(
      {
        try {
          ChatCompletionRequest::fromJson(json, 1);
        } catch (const std::invalid_argument& e) {
          std::string errorMsg = e.what();
          EXPECT_NE(errorMsg.find("not found in tools"), std::string::npos);
          EXPECT_NE(errorMsg.find("missing_tool"), std::string::npos);
          throw;
        }
      },
      std::invalid_argument)
      << "Should throw when tool_choice.function.name not in tools";
}

TEST(ChatCompletionRequestTest, ToolChoiceRequired) {
  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather"));
  json["tools"].append(createToolJson("get_time", "Get time"));
  json["tool_choice"] = "required";

  auto request = ChatCompletionRequest::fromJson(json, 1);

  ASSERT_TRUE(request.tools.has_value());
  ASSERT_EQ(request.tools->size(), 2);
  ASSERT_TRUE(request.tool_choice.has_value());
  EXPECT_EQ(request.tool_choice->type, "required");
  EXPECT_FALSE(request.tool_choice->function.has_value());

  auto llmRequest = request.toLLMRequest();
  ASSERT_TRUE(llmRequest.tool_choice.has_value());
  EXPECT_EQ(llmRequest.tool_choice->type, "required");
}

// validateToolMessages Tests

TEST(ChatCompletionRequestTest, ValidToolMessageSequence) {
  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));
  json["messages"].append(createAssistantWithToolCall(
      "call_abc123", "get_weather", "{\"location\":\"NYC\"}"));
  json["messages"].append(createToolMessage("call_abc123", "Sunny, 72°F"));

  auto request = ChatCompletionRequest::fromJson(json, 1);

  ASSERT_EQ(request.messages.size(), 3);
  ASSERT_TRUE(request.messages[1].tool_calls.has_value());
  EXPECT_EQ(request.messages[1].tool_calls->at(0).id, "call_abc123");
  EXPECT_EQ(request.messages[2].role, "tool");
  EXPECT_EQ(request.messages[2].tool_call_id.value(), "call_abc123");
}

TEST(ChatCompletionRequestTest, ToolMessageMissingAfterToolCalls) {
  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));
  json["messages"].append(
      createAssistantWithToolCall("call_abc123", "get_weather", "{}"));
  json["messages"].append(createUserMessage("Never mind"));

  EXPECT_THROW(
      {
        try {
          ChatCompletionRequest::fromJson(json, 1);
        } catch (const std::invalid_argument& e) {
          std::string errorMsg = e.what();
          EXPECT_TRUE(errorMsg.find("Incomplete tool call conversation") !=
                          std::string::npos ||
                      errorMsg.find("Expected message with role='tool'") !=
                          std::string::npos);
          throw;
        }
      },
      std::invalid_argument)
      << "Should throw when tool message is missing after tool_calls";
}

TEST(ChatCompletionRequestTest, ToolMessageMissingToolCallId) {
  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));
  json["messages"].append(
      createAssistantWithToolCall("call_abc123", "get_weather", "{}"));

  // Add tool message without tool_call_id
  Json::Value toolMsg;
  toolMsg["role"] = "tool";
  toolMsg["content"] = "Sunny";
  // Missing tool_call_id
  json["messages"].append(toolMsg);

  EXPECT_THROW(
      {
        try {
          ChatCompletionRequest::fromJson(json, 1);
        } catch (const std::invalid_argument& e) {
          EXPECT_NE(
              std::string(e.what()).find("must include 'tool_call_id' field"),
              std::string::npos);
          throw;
        }
      },
      std::invalid_argument)
      << "Should throw when tool message is missing tool_call_id";
}

TEST(ChatCompletionRequestTest, ToolMessageMismatchedCallId) {
  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));
  json["messages"].append(
      createAssistantWithToolCall("call_abc123", "get_weather", "{}"));
  json["messages"].append(
      createToolMessage("call_xyz789", "Sunny"));  // Wrong ID

  EXPECT_THROW(
      {
        try {
          ChatCompletionRequest::fromJson(json, 1);
        } catch (const std::invalid_argument& e) {
          std::string errorMsg = e.what();
          EXPECT_TRUE(
              errorMsg.find("Missing tool response") != std::string::npos ||
              errorMsg.find("Unknown tool_call_id") != std::string::npos ||
              errorMsg.find("does not match") != std::string::npos);
          throw;
        }
      },
      std::invalid_argument)
      << "Should throw when tool_call_id doesn't match";
}

TEST(ChatCompletionRequestTest, FewerOutputsThanToolCalls) {
  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather and time?"));

  // Assistant requests 3 tool calls
  Json::Value assistantMsg;
  assistantMsg["role"] = "assistant";
  assistantMsg["content"] = "";

  Json::Value toolCall1, toolCall2, toolCall3;
  toolCall1["id"] = "call_abc123";
  toolCall1["type"] = "function";
  toolCall1["function"]["name"] = "get_weather";
  toolCall1["function"]["arguments"] = "{\"location\":\"NYC\"}";

  toolCall2["id"] = "call_def456";
  toolCall2["type"] = "function";
  toolCall2["function"]["name"] = "get_time";
  toolCall2["function"]["arguments"] = "{}";

  toolCall3["id"] = "call_ghi789";
  toolCall3["type"] = "function";
  toolCall3["function"]["name"] = "get_date";
  toolCall3["function"]["arguments"] = "{}";

  assistantMsg["tool_calls"].append(toolCall1);
  assistantMsg["tool_calls"].append(toolCall2);
  assistantMsg["tool_calls"].append(toolCall3);
  json["messages"].append(assistantMsg);

  // Client only provides 2 outputs (missing call_ghi789)
  json["messages"].append(createToolMessage("call_abc123", "Sunny, 72°F"));
  json["messages"].append(createToolMessage("call_def456", "3:45 PM"));

  EXPECT_THROW(
      {
        try {
          ChatCompletionRequest::fromJson(json, 1);
        } catch (const std::invalid_argument& e) {
          std::string errorMsg = e.what();
          EXPECT_TRUE(errorMsg.find("Incomplete tool call conversation") !=
                          std::string::npos ||
                      errorMsg.find("requested 3") != std::string::npos);
          EXPECT_NE(errorMsg.find("call_ghi789"), std::string::npos);
          throw;
        }
      },
      std::invalid_argument)
      << "Should throw when fewer outputs than tool calls";
}

TEST(ChatCompletionRequestTest, MoreOutputsThanToolCalls) {
  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));

  // Assistant requests 2 tool calls
  Json::Value assistantMsg;
  assistantMsg["role"] = "assistant";
  assistantMsg["content"] = "";

  Json::Value toolCall1, toolCall2;
  toolCall1["id"] = "call_abc123";
  toolCall1["type"] = "function";
  toolCall1["function"]["name"] = "get_weather";
  toolCall1["function"]["arguments"] = "{\"location\":\"NYC\"}";

  toolCall2["id"] = "call_def456";
  toolCall2["type"] = "function";
  toolCall2["function"]["name"] = "get_time";
  toolCall2["function"]["arguments"] = "{}";

  assistantMsg["tool_calls"].append(toolCall1);
  assistantMsg["tool_calls"].append(toolCall2);
  json["messages"].append(assistantMsg);

  // Client provides 3 outputs (extra call_ghi789)
  json["messages"].append(createToolMessage("call_abc123", "Sunny, 72°F"));
  json["messages"].append(createToolMessage("call_def456", "3:45 PM"));
  json["messages"].append(
      createToolMessage("call_ghi789", "Extra output"));  // Extra!

  EXPECT_THROW(
      {
        try {
          ChatCompletionRequest::fromJson(json, 1);
        } catch (const std::invalid_argument& e) {
          std::string errorMsg = e.what();
          EXPECT_TRUE(errorMsg.find("Too many tool call responses") !=
                          std::string::npos ||
                      errorMsg.find("requested 2") != std::string::npos);
          EXPECT_NE(errorMsg.find("call_ghi789"), std::string::npos);
          throw;
        }
      },
      std::invalid_argument)
      << "Should throw when more outputs than tool calls";
}

TEST(ChatCompletionRequestTest, OneToolCallMultipleOutputs) {
  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));
  json["messages"].append(
      createAssistantWithToolCall("call_abc123", "get_weather", "{}"));

  // Client provides 2 outputs when only 1 was requested
  json["messages"].append(createToolMessage("call_abc123", "Sunny, 72°F"));
  json["messages"].append(createToolMessage("call_def456", "Extra output"));

  EXPECT_THROW(
      {
        try {
          ChatCompletionRequest::fromJson(json, 1);
        } catch (const std::invalid_argument& e) {
          EXPECT_NE(std::string(e.what()).find("Too many tool call responses"),
                    std::string::npos);
          throw;
        }
      },
      std::invalid_argument)
      << "Should throw when multiple outputs for single tool call";
}

TEST(ChatCompletionRequestTest, ZeroToolCallOutputs) {
  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));
  json["messages"].append(
      createAssistantWithToolCall("call_abc123", "get_weather", "{}"));
  // No tool message follows

  EXPECT_THROW(
      {
        try {
          ChatCompletionRequest::fromJson(json, 1);
        } catch (const std::invalid_argument& e) {
          EXPECT_NE(
              std::string(e.what()).find("Expected message with role='tool'"),
              std::string::npos);
          throw;
        }
      },
      std::invalid_argument)
      << "Should throw when no outputs after tool calls";
}

TEST(ChatCompletionRequestTest, DuplicateToolCallIds) {
  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));

  // Assistant requests 2 different tool calls
  Json::Value assistantMsg;
  assistantMsg["role"] = "assistant";
  assistantMsg["content"] = "";

  Json::Value toolCall1, toolCall2;
  toolCall1["id"] = "call_abc123";
  toolCall1["type"] = "function";
  toolCall1["function"]["name"] = "get_weather";
  toolCall1["function"]["arguments"] = "{}";

  toolCall2["id"] = "call_def456";
  toolCall2["type"] = "function";
  toolCall2["function"]["name"] = "get_time";
  toolCall2["function"]["arguments"] = "{}";

  assistantMsg["tool_calls"].append(toolCall1);
  assistantMsg["tool_calls"].append(toolCall2);
  json["messages"].append(assistantMsg);

  // Client provides duplicate response for call_abc123
  json["messages"].append(createToolMessage("call_abc123", "Sunny"));
  json["messages"].append(
      createToolMessage("call_abc123", "Duplicate!"));  // Duplicate!

  EXPECT_THROW(
      {
        try {
          ChatCompletionRequest::fromJson(json, 1);
        } catch (const std::invalid_argument& e) {
          std::string errorMsg = e.what();
          EXPECT_TRUE(errorMsg.find("Duplicate tool response") !=
                          std::string::npos ||
                      errorMsg.find("call_abc123") != std::string::npos);
          throw;
        }
      },
      std::invalid_argument)
      << "Should throw when duplicate tool_call_ids exist";
}

TEST(ChatCompletionRequestTest, MultipleValidToolCallSequence) {
  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather and time?"));

  // Assistant requests 3 tool calls
  Json::Value assistantMsg;
  assistantMsg["role"] = "assistant";
  assistantMsg["content"] = "";

  Json::Value toolCall1, toolCall2, toolCall3;
  toolCall1["id"] = "call_abc123";
  toolCall1["type"] = "function";
  toolCall1["function"]["name"] = "get_weather";
  toolCall1["function"]["arguments"] = "{\"location\":\"NYC\"}";

  toolCall2["id"] = "call_def456";
  toolCall2["type"] = "function";
  toolCall2["function"]["name"] = "get_time";
  toolCall2["function"]["arguments"] = "{}";

  toolCall3["id"] = "call_ghi789";
  toolCall3["type"] = "function";
  toolCall3["function"]["name"] = "get_date";
  toolCall3["function"]["arguments"] = "{}";

  assistantMsg["tool_calls"].append(toolCall1);
  assistantMsg["tool_calls"].append(toolCall2);
  assistantMsg["tool_calls"].append(toolCall3);
  json["messages"].append(assistantMsg);

  // Client provides all 3 outputs in correct order
  json["messages"].append(createToolMessage("call_abc123", "Sunny, 72°F"));
  json["messages"].append(createToolMessage("call_def456", "3:45 PM"));
  json["messages"].append(createToolMessage("call_ghi789", "April 29, 2026"));

  auto request = ChatCompletionRequest::fromJson(json, 1);

  ASSERT_EQ(request.messages.size(), 5);
  ASSERT_TRUE(request.messages[1].tool_calls.has_value());
  EXPECT_EQ(request.messages[1].tool_calls->size(), 3);
}

TEST(ChatCompletionRequestTest, ValidToolCallsFollowedByUserMessage) {
  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));

  // Assistant requests 2 tool calls
  Json::Value assistantMsg;
  assistantMsg["role"] = "assistant";
  assistantMsg["content"] = "";

  Json::Value toolCall1, toolCall2;
  toolCall1["id"] = "call_abc123";
  toolCall1["type"] = "function";
  toolCall1["function"]["name"] = "get_weather";
  toolCall1["function"]["arguments"] = "{}";

  toolCall2["id"] = "call_def456";
  toolCall2["type"] = "function";
  toolCall2["function"]["name"] = "get_time";
  toolCall2["function"]["arguments"] = "{}";

  assistantMsg["tool_calls"].append(toolCall1);
  assistantMsg["tool_calls"].append(toolCall2);
  json["messages"].append(assistantMsg);

  json["messages"].append(createToolMessage("call_abc123", "Sunny"));
  json["messages"].append(createToolMessage("call_def456", "3:45 PM"));

  json["messages"].append(createUserMessage("Thanks!"));

  auto request = ChatCompletionRequest::fromJson(json, 1);

  ASSERT_EQ(request.messages.size(), 5);
  ASSERT_EQ(request.messages[1].tool_calls->size(), 2);
  EXPECT_EQ(request.messages[4].role, "user");
}

}  // namespace
