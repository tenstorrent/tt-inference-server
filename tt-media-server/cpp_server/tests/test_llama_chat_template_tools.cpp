// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "domain/chat_message.hpp"
#include "domain/tool_calls/tool.hpp"
#include "utils/tokenizers/llama_tokenizer.hpp"

using namespace tt::domain;
using namespace tt::domain::tool_calls;
using namespace tt::utils::tokenizers;

// Create a standard weather tool definition
Tool createWeatherTool() {
  Tool tool;
  tool.type = "function";
  tool.functionDefinition.name = "get_weather";
  tool.functionDefinition.description = "Get weather info";

  Json::Value params;
  params["type"] = "object";
  params["properties"]["location"]["type"] = "string";
  params["required"].append("location");
  tool.functionDefinition.parameters = params;

  return tool;
}

// Create a simple user message
ChatMessage createUserMessage(const std::string& content) {
  ChatMessage msg;
  msg.role = "user";
  msg.content = content;
  return msg;
}

// Create a simple assistant message
ChatMessage createAssistantMessage(const std::string& content) {
  ChatMessage msg;
  msg.role = "assistant";
  msg.content = content;
  return msg;
}

// Create an assistant message with a single tool call
ChatMessage createAssistantWithToolCall(const std::string& content,
                                        const std::string& toolCallId,
                                        const std::string& functionName,
                                        const std::string& arguments) {
  ChatMessage msg;
  msg.role = "assistant";
  msg.content = content;

  ToolCall toolCall;
  toolCall.id = toolCallId;
  toolCall.type = "function";
  toolCall.functionCall.name = functionName;

  // Parse arguments as JSON
  Json::Reader reader;
  if (!reader.parse(arguments, toolCall.functionCall.arguments)) {
    throw std::runtime_error("Failed to parse arguments JSON");
  }

  msg.tool_calls = std::vector<ToolCall>{toolCall};

  return msg;
}

// Create a tool output message
ChatMessage createToolOutputMessage(const std::string& toolCallId,
                                    const std::string& content) {
  ChatMessage msg;
  msg.role = "tool";
  msg.tool_call_id = toolCallId;
  msg.content = content;
  return msg;
}

void testChatTemplateWithoutTools() {
  std::cout << "\n=== Testing Chat Template Without Tools (Llama) ===\n";

  // Create Llama tokenizer directly
  LlamaTokenizer tokenizer("tokenizers/meta-llama/Llama-3.1-8B-Instruct/tokenizer.json");

  std::vector<ChatMessage> messages;
  messages.push_back(createUserMessage("What's the weather like?"));

  std::string result =
      tokenizer.applyChatTemplate(messages, true, std::nullopt);

  // Should not contain tool-related markers when no tools provided
  assert(result.find("ipython") == std::string::npos);
  assert(result.find("Given the following functions") == std::string::npos);

  // Should contain the user message
  assert(result.find("What's the weather like?") != std::string::npos);

  std::cout << "✓ Chat template without tools applied correctly\n";
  std::cout << "✅ Test passed!\n";
}

void testChatTemplateWithSingleTool() {
  std::cout << "\n=== Testing Llama Single Tool Template ===\n";

  LlamaTokenizer tokenizer("tokenizers/meta-llama/Llama-3.1-8B-Instruct/tokenizer.json");

  // Create message
  std::vector<ChatMessage> messages = {createUserMessage("Get weather for SF")};

  // Create tool
  std::vector<Tool> tools = {createWeatherTool()};

  // Get actual result
  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  std::cout << "Generated template:\n" << actual << "\n";

  // Verify key components are present
  assert(actual.find("<|begin_of_text|>") != std::string::npos);
  assert(actual.find("<|start_header_id|>system<|end_header_id|>") != std::string::npos);
  assert(actual.find("Environment: ipython") != std::string::npos);
  assert(actual.find("Given the following functions") != std::string::npos);
  assert(actual.find("get_weather") != std::string::npos);
  assert(actual.find("Get weather for SF") != std::string::npos);
  assert(actual.find("<|start_header_id|>assistant<|end_header_id|>") != std::string::npos);

  std::cout << "✅ Test passed!\n";
}

void testChatTemplateWithToolCall() {
  std::cout << "\n=== Testing Llama Tool Call Response ===\n";

  LlamaTokenizer tokenizer("tokenizers/meta-llama/Llama-3.1-8B-Instruct/tokenizer.json");

  ChatMessage userMsg = createUserMessage("What's the weather in SF?");
  ChatMessage assistantMsg = createAssistantWithToolCall(
      "", "call_123", "get_weather", "{\"location\":\"San Francisco\"}");

  std::vector<ChatMessage> messages = {userMsg, assistantMsg};
  std::vector<Tool> tools = {createWeatherTool()};

  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  std::cout << "Generated template:\n" << actual << "\n";

  // Verify tool call format
  assert(actual.find("{\"name\": \"get_weather\"") != std::string::npos);
  assert(actual.find("\"parameters\":") != std::string::npos);
  assert(actual.find("San Francisco") != std::string::npos);

  std::cout << "✅ Test passed!\n";
}

void testChatTemplateWithToolOutput() {
  std::cout << "\n=== Testing Llama Tool Output ===\n";

  LlamaTokenizer tokenizer("tokenizers/meta-llama/Llama-3.1-8B-Instruct/tokenizer.json");

  ChatMessage userMsg = createUserMessage("What's the weather in SF?");
  ChatMessage assistantMsg = createAssistantWithToolCall(
      "", "call_123", "get_weather", "{\"location\":\"San Francisco\"}");
  ChatMessage toolMsg = createToolOutputMessage(
      "call_123", "{\"temperature\":72,\"conditions\":\"sunny\"}");

  std::vector<ChatMessage> messages = {userMsg, assistantMsg, toolMsg};
  std::vector<Tool> tools = {createWeatherTool()};

  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  std::cout << "Generated template:\n" << actual << "\n";

  // Verify tool output appears in ipython role
  assert(actual.find("<|start_header_id|>ipython<|end_header_id|>") != std::string::npos);
  assert(actual.find("temperature") != std::string::npos);
  assert(actual.find("sunny") != std::string::npos);

  std::cout << "✅ Test passed!\n";
}

void testMultipleToolCallsError() {
  std::cout << "\n=== Testing Multiple Tool Calls Error ===\n";

  LlamaTokenizer tokenizer("tokenizers/meta-llama/Llama-3.1-8B-Instruct/tokenizer.json");

  ChatMessage userMsg = createUserMessage("Get weather for SF and LA");

  // Create assistant message with multiple tool calls
  ChatMessage assistantMsg;
  assistantMsg.role = "assistant";
  assistantMsg.content = "";

  ToolCall toolCall1;
  toolCall1.id = "call_1";
  toolCall1.type = "function";
  toolCall1.functionCall.name = "get_weather";
  Json::Reader reader;
  reader.parse("{\"location\":\"SF\"}", toolCall1.functionCall.arguments);

  ToolCall toolCall2;
  toolCall2.id = "call_2";
  toolCall2.type = "function";
  toolCall2.functionCall.name = "get_weather";
  reader.parse("{\"location\":\"LA\"}", toolCall2.functionCall.arguments);

  assistantMsg.tool_calls = std::vector<ToolCall>{toolCall1, toolCall2};

  std::vector<ChatMessage> messages = {userMsg, assistantMsg};
  std::vector<Tool> tools = {createWeatherTool()};

  // Should throw exception for multiple tool calls
  bool exceptionThrown = false;
  try {
    tokenizer.applyChatTemplate(messages, true, tools);
  } catch (const std::runtime_error& e) {
    std::string errorMsg = e.what();
    if (errorMsg.find("single tool-calls") != std::string::npos) {
      exceptionThrown = true;
    }
  }

  assert(exceptionThrown);
  std::cout << "✓ Correctly throws error for multiple tool calls\n";
  std::cout << "✅ Test passed!\n";
}

int main() {
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║      Llama Tool Calling Template Test Suite             ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";

  try {
    testChatTemplateWithoutTools();
    testChatTemplateWithSingleTool();
    testChatTemplateWithToolCall();
    testChatTemplateWithToolOutput();
    testMultipleToolCallsError();

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
