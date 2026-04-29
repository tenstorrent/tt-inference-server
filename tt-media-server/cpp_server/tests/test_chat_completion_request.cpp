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
    // New validation gives "Incomplete tool call conversation" error
    assert(errorMsg.find("Incomplete tool call conversation") !=
               std::string::npos ||
           errorMsg.find("Expected message with role='tool'") !=
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
    // New validation reports missing expected ID or unknown actual ID
    assert(errorMsg.find("Missing tool response") != std::string::npos ||
           errorMsg.find("Unknown tool_call_id") != std::string::npos ||
           errorMsg.find("does not match") != std::string::npos);
  }

  assert(exceptionThrown && "Should throw when tool_call_id doesn't match");

  std::cout << "✓ Mismatched tool_call_id correctly rejected\n";
  std::cout << "✅ Test passed!\n";
}

// ==================== New Tool Call ID Validation Tests ====================

void testToolCallFewerOutputsThanExpected() {
  std::cout << "\n=== Testing Scenario 1: Fewer Outputs Than Tool Calls "
               "(Should Reject) ===\n";

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

  bool exceptionThrown = false;
  try {
    ChatCompletionRequest::fromJson(json, 1);
  } catch (const std::invalid_argument& e) {
    exceptionThrown = true;
    std::string errorMsg = e.what();
    assert(errorMsg.find("Incomplete tool call conversation") !=
               std::string::npos ||
           errorMsg.find("requested 3") != std::string::npos);
    assert(errorMsg.find("call_ghi789") != std::string::npos);
    std::cout << "  Error message: " << errorMsg << "\n";
  }

  assert(exceptionThrown &&
         "Should throw when fewer outputs than tool calls");

  std::cout << "✓ Fewer outputs correctly rejected\n";
  std::cout << "✅ Test passed!\n";
}

void testToolCallMoreOutputsThanExpected() {
  std::cout << "\n=== Testing Scenario 2: More Outputs Than Tool Calls (Should "
               "Reject) ===\n";

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

  bool exceptionThrown = false;
  try {
    ChatCompletionRequest::fromJson(json, 1);
  } catch (const std::invalid_argument& e) {
    exceptionThrown = true;
    std::string errorMsg = e.what();
    assert(errorMsg.find("Too many tool call responses") != std::string::npos ||
           errorMsg.find("requested 2") != std::string::npos);
    assert(errorMsg.find("call_ghi789") != std::string::npos);
    std::cout << "  Error message: " << errorMsg << "\n";
  }

  assert(exceptionThrown && "Should throw when more outputs than tool calls");

  std::cout << "✓ Extra outputs correctly rejected\n";
  std::cout << "✅ Test passed!\n";
}

void testToolCallOneRequestMultipleOutputs() {
  std::cout << "\n=== Testing Scenario 3: 1 Tool Call → Multiple Outputs "
               "(Should Reject) ===\n";

  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));
  json["messages"].append(
      createAssistantWithToolCall("call_abc123", "get_weather", "{}"));

  // Client provides 2 outputs when only 1 was requested
  json["messages"].append(createToolMessage("call_abc123", "Sunny, 72°F"));
  json["messages"].append(createToolMessage("call_def456", "Extra output"));

  bool exceptionThrown = false;
  try {
    ChatCompletionRequest::fromJson(json, 1);
  } catch (const std::invalid_argument& e) {
    exceptionThrown = true;
    std::string errorMsg = e.what();
    assert(errorMsg.find("Too many tool call responses") != std::string::npos);
    std::cout << "  Error message: " << errorMsg << "\n";
  }

  assert(exceptionThrown &&
         "Should throw when multiple outputs for single tool call");

  std::cout << "✓ Multiple outputs for single tool call correctly rejected\n";
  std::cout << "✅ Test passed!\n";
}

void testToolCallZeroOutputs() {
  std::cout
      << "\n=== Testing Scenario 4: Tool Calls → 0 Outputs (Should Reject) "
         "===\n";

  Json::Value json = createBasicRequestJson();
  json["messages"].clear();

  json["messages"].append(createUserMessage("What's the weather?"));
  json["messages"].append(
      createAssistantWithToolCall("call_abc123", "get_weather", "{}"));
  // No tool message follows

  bool exceptionThrown = false;
  try {
    ChatCompletionRequest::fromJson(json, 1);
  } catch (const std::invalid_argument& e) {
    exceptionThrown = true;
    std::string errorMsg = e.what();
    assert(errorMsg.find("Expected message with role='tool'") !=
           std::string::npos);
    std::cout << "  Error message: " << errorMsg << "\n";
  }

  assert(exceptionThrown && "Should throw when no outputs after tool calls");

  std::cout << "✓ Zero outputs correctly rejected\n";
  std::cout << "✅ Test passed!\n";
}

void testToolCallDuplicateToolCallIds() {
  std::cout << "\n=== Testing Duplicate tool_call_id in Responses (Should "
               "Reject) ===\n";

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

  bool exceptionThrown = false;
  try {
    ChatCompletionRequest::fromJson(json, 1);
  } catch (const std::invalid_argument& e) {
    exceptionThrown = true;
    std::string errorMsg = e.what();
    // New validation properly detects duplicates
    assert(errorMsg.find("Duplicate tool response") != std::string::npos ||
           errorMsg.find("call_abc123") != std::string::npos);
    std::cout << "  Error message: " << errorMsg << "\n";
  }

  assert(exceptionThrown && "Should throw when duplicate tool_call_ids exist");

  std::cout << "✓ Duplicate tool_call_ids correctly rejected\n";
  std::cout << "✅ Test passed!\n";
}

void testToolCallMultipleValidSequence() {
  std::cout << "\n=== Testing Multiple Valid Tool Calls Sequence ===\n";

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

  assert(request.messages.size() == 5);
  assert(request.messages[1].tool_calls.has_value());
  assert(request.messages[1].tool_calls->size() == 3);

  std::cout << "✓ Multiple valid tool calls sequence accepted\n";
  std::cout << "✅ Test passed!\n";
}

void testToolCallValidSequenceWithSubsequentMessage() {
  std::cout << "\n=== Testing Valid Tool Calls Followed by User Message ===\n";

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

  assert(request.messages.size() == 5);
  assert(request.messages[1].tool_calls->size() == 2);
  assert(request.messages[4].role == "user");

  std::cout
      << "✓ Tool calls followed by user message correctly accepted\n";
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

    // New tool call ID validation tests (OpenAI compatibility)
    testToolCallFewerOutputsThanExpected();
    testToolCallMoreOutputsThanExpected();
    testToolCallOneRequestMultipleOutputs();
    testToolCallZeroOutputs();
    testToolCallDuplicateToolCallIds();
    testToolCallMultipleValidSequence();
    testToolCallValidSequenceWithSubsequentMessage();

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
