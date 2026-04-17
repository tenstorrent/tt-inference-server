// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "domain/chat_completion_request.hpp"
#include "domain/chat_message.hpp"
#include "domain/tool_calls/tool.hpp"
#include "utils/tokenizers/tokenizer.hpp"

using namespace tt::domain;
using namespace tt::domain::tool_calls;

// DeepSeek special tokens
namespace DeepSeekTokens {
  const char* BOS = "<｜begin▁of▁sentence｜>";
  const char* USER_TAG = "<｜User｜>";
  const char* ASSISTANT_TAG = "<｜Assistant｜>";
  const char* TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>";
  const char* TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>";
  const char* TOOL_SEP = "<｜tool▁sep｜>";
  const char* TOOL_CALL_END = "<｜tool▁call▁end｜>";
  const char* TOOL_CALLS_END = "<｜tool▁calls▁end｜>";
}

// Helper to build expected tool section
std::string buildExpectedToolSection(const std::vector<Tool>& tools) {
  std::ostringstream out;

  // Tool calling instructions
  out << "You are a helpful assistant with tool calling capabilities. "
      << "When a tool call is needed, you MUST use the following format to issue the call:\n"
      << DeepSeekTokens::TOOL_CALLS_BEGIN
      << DeepSeekTokens::TOOL_CALL_BEGIN
      << "function"
      << DeepSeekTokens::TOOL_SEP
      << "FUNCTION_NAME\n"
      << "```json\n{\"param1\":\"value1\",\"param2\":\"value2\"}\n```"
      << DeepSeekTokens::TOOL_CALL_END
      << DeepSeekTokens::TOOL_CALLS_END
      << "\n\nMake sure the JSON is valid.\n"
      << "## Tools\n\n### Function\n\nYou have the following functions available:\n\n";

  // Tool definitions
  // Note: The tokenizer uses default JSON formatting (with indentation)
  // We need to match that by using the stream operator directly
  for (const auto& tool : tools) {
    out << "- `" << tool.functionDefinition.name << "`:\n```json\n"
        << tool.toJson() << "\n```\n";
  }

  return out.str();
}

void testChatTemplateWithoutTools() {
  std::cout << "\n=== Testing Chat Template Without Tools ===\n";

  auto& tokenizer = tt::utils::tokenizers::activeTokenizer();

  std::vector<ChatMessage> messages;
  ChatMessage msg;
  msg.role = "user";
  msg.content = "What's the weather like?";
  messages.push_back(msg);

  std::string result = tokenizer.applyChatTemplate(messages, true, std::nullopt);

  // Should not contain tool-related markers when no tools provided
  assert(result.find("tools") == std::string::npos ||
         result.find("<｜tool▁calls▁begin｜>") == std::string::npos);

  // Should contain the user message
  assert(result.find(content) != std::string::npos);

  std::cout << "✓ Chat template without tools applied correctly\n";
  std::cout << "✅ Test passed!\n";
}


void testChatTemplateWithSingleTool() {
  std::cout << "\n=== Testing Exact Single Tool Template ===\n";

  auto& tokenizer = tt::utils::tokenizers::activeTokenizer();

  // Create message
  std::vector<ChatMessage> messages;
  ChatMessage msg;
  msg.role = "user";
  msg.content = "Get weather for SF";
  messages.push_back(msg);

  // Create tool
  std::vector<Tool> tools;
  Tool tool;
  tool.type = "function";
  tool.functionDefinition.name = "get_weather";
  tool.functionDefinition.description = "Get weather info";

  Json::Value params;
  params["type"] = "object";
  params["properties"]["location"]["type"] = "string";
  params["required"].append("location");
  tool.functionDefinition.parameters = params;
  tools.push_back(tool);

  // Get actual result
  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  // Build expected result
  std::ostringstream expected;
  expected << DeepSeekTokens::BOS;
  expected << buildExpectedToolSection(tools);
  expected << DeepSeekTokens::USER_TAG << "Get weather for SF";
  expected << DeepSeekTokens::ASSISTANT_TAG;

  // Compare
  if (actual == expected.str()) {
    std::cout << "✅ Exact match! Template is perfect.\n";
  } else {
    std::cout << "❌ Mismatch detected!\n";
    std::cout << "\n=== EXPECTED ===\n" << expected.str() << "\n";
    std::cout << "\n=== ACTUAL ===\n" << actual << "\n";
    std::cout << "\nExpected length: " << expected.str().length() << "\n";
    std::cout << "Actual length: " << actual.length() << "\n";

    // Find first difference
    for (size_t i = 0; i < std::min(expected.str().length(), actual.length()); ++i) {
      if (expected.str()[i] != actual[i]) {
        std::cout << "First difference at position " << i << ":\n";
        std::cout << "  Expected: '" << expected.str()[i] << "' (ASCII " << (int)expected.str()[i] << ")\n";
        std::cout << "  Actual: '" << actual[i] << "' (ASCII " << (int)actual[i] << ")\n";
        break;
      }
    }
    throw std::runtime_error("Template mismatch");
  }

  std::cout << "✅ Test passed!\n";
}

void testChatTemplateWithMultipleTools() {
  std::cout << "\n=== Testing Exact Multiple Tools Template ===\n";

  auto& tokenizer = tt::utils::tokenizers::activeTokenizer();

  // Create message
  std::vector<ChatMessage> messages;
  ChatMessage msg;
  msg.role = "user";
  msg.content = "Check weather and time";
  messages.push_back(msg);

  // Create tools
  std::vector<Tool> tools;

  // Tool 1: get_weather
  Tool weatherTool;
  weatherTool.type = "function";
  weatherTool.functionDefinition.name = "get_weather";
  weatherTool.functionDefinition.description = "Get weather";
  Json::Value weatherParams;
  weatherParams["type"] = "object";
  weatherParams["properties"]["location"]["type"] = "string";
  weatherTool.functionDefinition.parameters = weatherParams;
  tools.push_back(weatherTool);

  // Tool 2: get_time
  Tool timeTool;
  timeTool.type = "function";
  timeTool.functionDefinition.name = "get_time";
  timeTool.functionDefinition.description = "Get time";
  Json::Value timeParams;
  timeParams["type"] = "object";
  timeParams["properties"]["timezone"]["type"] = "string";
  timeTool.functionDefinition.parameters = timeParams;
  tools.push_back(timeTool);

  // Get actual result
  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  // Build expected result
  std::ostringstream expected;
  expected << DeepSeekTokens::BOS;
  expected << buildExpectedToolSection(tools);
  expected << DeepSeekTokens::USER_TAG << "Check weather and time";
  expected << DeepSeekTokens::ASSISTANT_TAG;

  // Compare
  if (actual == expected.str()) {
    std::cout << "✅ Exact match! Template is perfect.\n";
  } else {
    std::cout << "❌ Mismatch detected!\n";
    std::cout << "\n=== EXPECTED ===\n" << expected.str() << "\n";
    std::cout << "\n=== ACTUAL ===\n" << actual << "\n";
    std::cout << "\nExpected length: " << expected.str().length() << "\n";
    std::cout << "Actual length: " << actual.length() << "\n";

    // Find first difference
    for (size_t i = 0; i < std::min(expected.str().length(), actual.length()); ++i) {
      if (expected.str()[i] != actual[i]) {
        std::cout << "First difference at position " << i << ":\n";
        std::cout << "  Expected: '" << expected.str()[i] << "' (ASCII " << (int)expected.str()[i] << ")\n";
        std::cout << "  Actual: '" << actual[i] << "' (ASCII " << (int)actual[i] << ")\n";
        break;
      }
    }
    throw std::runtime_error("Template mismatch for multiple tools");
  }

  std::cout << "✅ Test passed!\n";
}
void testChatTemplateWithConversationHistory() {
  std::cout << "\n=== Testing Exact Conversation History Template ===\n";

  auto& tokenizer = tt::utils::tokenizers::activeTokenizer();

  std::vector<ChatMessage> messages;

  // First user message
  ChatMessage userMsg1;
  userMsg1.role = "user";
  userMsg1.content = "Check SF weather";
  messages.push_back(userMsg1);

  // Assistant response
  ChatMessage assistantMsg;
  assistantMsg.role = "assistant";
  assistantMsg.content = "I'll check for you.";
  messages.push_back(assistantMsg);

  // Follow-up user message
  ChatMessage userMsg2;
  userMsg2.role = "user";
  userMsg2.content = "Also check LA";
  messages.push_back(userMsg2);

  // Tool
  std::vector<Tool> tools;
  Tool weatherTool;
  weatherTool.type = "function";
  weatherTool.functionDefinition.name = "get_weather";
  weatherTool.functionDefinition.description = "Get weather";

  Json::Value params;
  params["type"] = "object";
  params["properties"]["location"]["type"] = "string";
  weatherTool.functionDefinition.parameters = params;
  tools.push_back(weatherTool);

  // Get actual result
  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  // Build expected result
  std::ostringstream expected;
  expected << DeepSeekTokens::BOS;
  expected << buildExpectedToolSection(tools);
  expected << DeepSeekTokens::USER_TAG << "Check SF weather";
  expected << DeepSeekTokens::ASSISTANT_TAG << "I'll check for you.";
  expected << DeepSeekTokens::USER_TAG << "Also check LA";
  expected << DeepSeekTokens::ASSISTANT_TAG;

  // Compare
  if (actual == expected.str()) {
    std::cout << "✅ Exact match! Conversation template is perfect.\n";
  } else {
    std::cout << "❌ Mismatch detected!\n";
    std::cout << "\n=== EXPECTED ===\n" << expected.str() << "\n";
    std::cout << "\n=== ACTUAL ===\n" << actual << "\n";
    std::cout << "\nExpected length: " << expected.str().length() << "\n";
    std::cout << "Actual length: " << actual.length() << "\n";

    // Find first difference
    for (size_t i = 0; i < std::min(expected.str().length(), actual.length()); ++i) {
      if (expected.str()[i] != actual[i]) {
        std::cout << "First difference at position " << i << ":\n";
        std::cout << "  Expected: '" << expected.str()[i] << "' (ASCII " << (int)expected.str()[i] << ")\n";
        std::cout << "  Actual: '" << actual[i] << "' (ASCII " << (int)actual[i] << ")\n";

        // Show context around the difference
        size_t contextStart = (i > 50) ? i - 50 : 0;
        size_t contextEnd = std::min(i + 50, std::min(expected.str().length(), actual.length()));
        std::cout << "\nContext (position " << contextStart << " to " << contextEnd << "):\n";
        std::cout << "Expected: \"" << expected.str().substr(contextStart, contextEnd - contextStart) << "\"\n";
        std::cout << "Actual:   \"" << actual.substr(contextStart, contextEnd - contextStart) << "\"\n";
        break;
      }
    }
    throw std::runtime_error("Conversation template mismatch");
  }

  std::cout << "✅ Test passed!\n";
  std::cout << "  ✓ System message included\n";
  std::cout << "  ✓ Tool section present\n";
  std::cout << "  ✓ All conversation turns formatted correctly\n";
  std::cout << "  ✓ Messages: " << messages.size() << "\n";
}

void testChatTemplateEmptyTools() {
  std::cout << "\n=== Testing Chat Template With Empty Tools Vector ===\n";

  auto& tokenizer = tt::utils::tokenizers::activeTokenizer();

  std::vector<ChatMessage> messages;
  ChatMessage msg;
  msg.role = "user";
  msg.content = "Hello";
  messages.push_back(msg);

  std::vector<Tool> emptyTools;
  std::string result = tokenizer.applyChatTemplate(messages, true, emptyTools);

  // Should behave like no tools provided
  assert(!result.empty());
  std::cout << "✓ Chat template with empty tools vector handled\n";

  std::cout << "✅ Test passed!\n";
}

void testToolStructureValidation() {
  std::cout << "\n=== Testing Tool Structure Validation ===\n";

  // Verify tool JSON structure
  Tool tool;
  tool.type = "function";
  tool.functionDefinition.name = "test_function";
  tool.functionDefinition.description = "A test function";

  Json::Value params;
  params["type"] = "object";
  params["properties"]["arg1"]["type"] = "string";
  params["properties"]["arg1"]["description"] = "First argument";
  params["required"].append("arg1");
  tool.functionDefinition.parameters = params;

  Json::Value toolJson = tool.toJson();

  // Verify structure matches OpenAI spec
  assert(toolJson.isMember("type"));
  assert(toolJson["type"].asString() == "function");
  assert(toolJson.isMember("function"));
  assert(toolJson["function"].isMember("name"));
  assert(toolJson["function"].isMember("description"));
  assert(toolJson["function"].isMember("parameters"));

  auto& funcParams = toolJson["function"]["parameters"];
  assert(funcParams.isMember("type"));
  assert(funcParams.isMember("properties"));
  assert(funcParams.isMember("required"));

  std::cout << "✓ Tool structure matches OpenAI specification\n";
  std::cout << "  - type field present\n";
  std::cout << "  - function.name present\n";
  std::cout << "  - function.description present\n";
  std::cout << "  - function.parameters.properties present\n";
  std::cout << "  - function.parameters.required present\n";

  std::cout << "✅ Test passed!\n";
}



int main() {
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║      Chat Template with Tools Test Suite                ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";

  try {
    testChatTemplateWithoutTools();
    testChatTemplateWithSingleTool();
    testChatTemplateWithMultipleTools();
    testChatTemplateWithConversationHistory();
    testChatTemplateEmptyTools();
    testToolStructureValidation();

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║              🎉 ALL TESTS PASSED! 🎉                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "\n❌ TEST FAILED: " << e.what() << "\n";
    return 1;
  }
}
