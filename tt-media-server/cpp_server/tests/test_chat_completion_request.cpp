// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <json/json.h>

#include <cassert>
#include <iostream>
#include <stdexcept>

#include "domain/chat_completion_request.hpp"

using namespace tt::domain;

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

// Helper to create a user message
Json::Value createUserMessage(const std::string& content) {
  Json::Value msg;
  msg["role"] = "user";
  msg["content"] = content;
  return msg;
}

// Helper to create an assistant message
Json::Value createAssistantMessage(const std::string& content) {
  Json::Value msg;
  msg["role"] = "assistant";
  msg["content"] = content;
  return msg;
}

// Helper to create an assistant message with a tool call
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

// ==================== Tool Parsing Tests ====================

void testParseRequestWithTools() {
  std::cout << "\n=== Testing Parse Request With Tools ===\n";

  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather info"));
  json["tools"].append(createToolJson("get_time", "Get current time"));

  auto request = ChatCompletionRequest::fromJson(json, 1);

  assert(request.tools.has_value());
  assert(request.tools->size() == 2);
  assert(request.tools->at(0).functionDefinition.name == "get_weather");
  assert(request.tools->at(1).functionDefinition.name == "get_time");

  std::cout << "✓ Request with multiple tools parsed correctly\n";
  std::cout << "✅ Test passed!\n";
}

// ==================== tool_choice Tests ====================

void testToolChoiceNone() {
  std::cout << "\n=== Testing tool_choice=none ===\n";

  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather"));
  json["tool_choice"] = "none";

  auto request = ChatCompletionRequest::fromJson(json, 1);

  // When tool_choice is "none", tools should still be parsed
  assert(request.tools.has_value());
  assert(!request.tools->empty() &&
         "Tools should be kept when tool_choice is 'none'");

  assert(request.tool_choice.has_value());
  assert(request.tool_choice->type == "none");

  // Verify tool_choice is copied to LLMRequest
  auto llmRequest = request.toLLMRequest();
  assert(llmRequest.tool_choice.has_value());
  assert(llmRequest.tool_choice->type == "none");

  std::cout << "✓ tool_choice=none parsed correctly\n";
  std::cout << "✓ Tools retained even with tool_choice=none\n";
  std::cout << "✓ tool_choice propagated to LLMRequest\n";
  std::cout << "✅ Test passed!\n";
}

void testToolChoiceAuto() {
  std::cout << "\n=== Testing tool_choice=auto ===\n";

  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather"));
  json["tool_choice"] = "auto";

  auto request = ChatCompletionRequest::fromJson(json, 1);

  assert(request.tools.has_value());
  assert(!request.tools->empty());
  assert(request.tool_choice.has_value());
  assert(request.tool_choice->type == "auto");

  std::cout << "✓ tool_choice=auto parsed correctly\n";
  std::cout << "✅ Test passed!\n";
}

void testToolChoiceNoneWithoutTools() {
  std::cout << "\n=== Testing tool_choice=none Without Tools ===\n";

  Json::Value json = createBasicRequestJson();
  json["tool_choice"] = "none";

  auto request = ChatCompletionRequest::fromJson(json, 1);
  assert(request.tool_choice.has_value());
  assert(request.tool_choice->type == "none");

  std::cout << "✓ tool_choice=none without tools accepted\n";
  std::cout << "✅ Test passed!\n";
}

void testToolChoiceNoneWithEmptyToolsArray() {
  std::cout << "\n=== Testing tool_choice=none With Empty Tools Array ===\n";

  Json::Value json = createBasicRequestJson();
  json["tools"] = Json::arrayValue;
  json["tool_choice"] = "none";

  auto request = ChatCompletionRequest::fromJson(json, 1);

  assert(request.tool_choice.has_value());
  assert(request.tool_choice->type == "none");

  std::cout << "✓ tool_choice=none with empty tools array accepted\n";
  std::cout << "✅ Test passed!\n";
}

void testToolChoiceAutoWithoutToolsRejected() {
  std::cout << "\n=== Testing tool_choice=auto Without Tools (Should Reject) "
               "===\n";

  Json::Value json = createBasicRequestJson();
  json["tool_choice"] = "auto";

  bool exceptionThrown = false;
  try {
    ChatCompletionRequest::fromJson(json, 1);
  } catch (const std::invalid_argument&) {
    exceptionThrown = true;
  }

  assert(exceptionThrown &&
         "Should throw invalid_argument for tool_choice=auto without tools");

  std::cout << "✓ tool_choice=auto without tools correctly rejected\n";
  std::cout << "✅ Test passed!\n";
}

void testToolChoiceUnknownStringRejected() {
  std::cout << "\n=== Testing tool_choice With Unknown Value (Should Reject) "
               "===\n";

  Json::Value json = createBasicRequestJson();
  json["tools"].append(createToolJson("get_weather", "Get weather"));
  json["tool_choice"] = "bogus";

  bool exceptionThrown = false;
  try {
    ChatCompletionRequest::fromJson(json, 1);
  } catch (const std::invalid_argument&) {
    exceptionThrown = true;
  }

  assert(exceptionThrown &&
         "Should throw invalid_argument for unknown tool_choice value");

  std::cout << "✓ Unknown tool_choice value correctly rejected\n";
  std::cout << "✅ Test passed!\n";
}

// ==================== validateToolMessages Tests ====================

void testValidToolMessageSequence() {
  std::cout << "\n=== Testing Valid Tool Message Sequence ===\n";

  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));
  json["messages"].append(createAssistantWithToolCall(
      "call_abc123", "get_weather", "{\"location\":\"NYC\"}"));
  json["messages"].append(createToolMessage("call_abc123", "Sunny, 72°F"));

  auto request = ChatCompletionRequest::fromJson(json, 1);

  assert(request.messages.size() == 3);
  assert(request.messages[1].tool_calls.has_value());
  assert(request.messages[1].tool_calls->at(0).id == "call_abc123");
  assert(request.messages[2].role == "tool");
  assert(request.messages[2].tool_call_id.value() == "call_abc123");

  std::cout << "✓ Valid tool message sequence accepted\n";
  std::cout << "✅ Test passed!\n";
}

void testToolMessageMissingAfterToolCalls() {
  std::cout << "\n=== Testing Missing Tool Message After tool_calls (Should "
               "Reject) ===\n";

  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));
  json["messages"].append(
      createAssistantWithToolCall("call_abc123", "get_weather", "{}"));
  json["messages"].append(createUserMessage("Never mind"));

  bool exceptionThrown = false;
  try {
    ChatCompletionRequest::fromJson(json, 1);
  } catch (const std::invalid_argument& e) {
    exceptionThrown = true;
    std::string errorMsg = e.what();
    assert(errorMsg.find("Expected message with role='tool'") !=
           std::string::npos);
  }

  assert(exceptionThrown &&
         "Should throw when tool message is missing after tool_calls");

  std::cout << "✓ Missing tool message correctly rejected\n";
  std::cout << "✅ Test passed!\n";
}

void testToolMessageMissingToolCallId() {
  std::cout
      << "\n=== Testing Tool Message Missing tool_call_id (Should Reject) "
         "===\n";

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

  bool exceptionThrown = false;
  try {
    ChatCompletionRequest::fromJson(json, 1);
  } catch (const std::invalid_argument& e) {
    exceptionThrown = true;
    std::string errorMsg = e.what();
    assert(errorMsg.find("must include 'tool_call_id' field") !=
           std::string::npos);
  }

  assert(exceptionThrown &&
         "Should throw when tool message is missing tool_call_id");

  std::cout << "✓ Missing tool_call_id correctly rejected\n";
  std::cout << "✅ Test passed!\n";
}

void testToolMessageMismatchedCallId() {
  std::cout
      << "\n=== Testing Tool Message With Mismatched tool_call_id (Should "
         "Reject) ===\n";

  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));
  json["messages"].append(
      createAssistantWithToolCall("call_abc123", "get_weather", "{}"));
  json["messages"].append(
      createToolMessage("call_xyz789", "Sunny"));  // Wrong ID

  bool exceptionThrown = false;
  try {
    ChatCompletionRequest::fromJson(json, 1);
  } catch (const std::invalid_argument& e) {
    exceptionThrown = true;
    std::string errorMsg = e.what();
    assert(errorMsg.find("does not match expected call_id") !=
           std::string::npos);
  }

  assert(exceptionThrown && "Should throw when tool_call_id doesn't match");

  std::cout << "✓ Mismatched tool_call_id correctly rejected\n";
  std::cout << "✅ Test passed!\n";
}

// Main function for running tests
int main() {
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║      Chat Completion Request Tool Test Suite            ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";

  try {
    testParseRequestWithTools();
    testToolChoiceNone();
    testToolChoiceAuto();
    testToolChoiceNoneWithoutTools();
    testToolChoiceNoneWithEmptyToolsArray();
    testToolChoiceAutoWithoutToolsRejected();
    testToolChoiceUnknownStringRejected();

    // validateToolMessages tests
    testValidToolMessageSequence();
    testToolMessageMissingAfterToolCalls();
    testToolMessageMissingToolCallId();
    testToolMessageMismatchedCallId();

    std::cout << "\n";
    std::cout
        << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout
        << "║              🎉 ALL TESTS PASSED! 🎉                    ║\n";
    std::cout
        << "╚══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "\n❌ TEST FAILED: " << e.what() << "\n";
    return 1;
  }
}
