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

// Base class for tokenizer-specific template configuration
struct TokenizerTemplateConfig {
  virtual ~TokenizerTemplateConfig() = default;

  // Special tokens
  virtual const char* bos() const = 0;
  virtual const char* userTag() const = 0;
  virtual const char* assistantTag() const = 0;
  virtual const char* toolCallsBegin() const = 0;
  virtual const char* toolCallBegin() const = 0;
  virtual const char* toolSep() const = 0;
  virtual const char* toolCallEnd() const = 0;
  virtual const char* toolCallsEnd() const = 0;

  // Build the tool section for this tokenizer
  virtual std::string buildToolSection(
      const std::vector<Tool>& tools) const = 0;

  // Name for logging
  virtual const char* name() const = 0;
};

struct DeepSeekTemplateConfig : public TokenizerTemplateConfig {
  const char* bos() const override { return "<｜begin▁of▁sentence｜>"; }
  const char* userTag() const override { return "<｜User｜>"; }
  const char* assistantTag() const override { return "<｜Assistant｜>"; }
  const char* toolCallsBegin() const override {
    return "<｜tool▁calls▁begin｜>";
  }
  const char* toolCallBegin() const override { return "<｜tool▁call▁begin｜>"; }
  const char* toolSep() const override { return "<｜tool▁sep｜>"; }
  const char* toolCallEnd() const override { return "<｜tool▁call▁end｜>"; }
  const char* toolCallsEnd() const override { return "<｜tool▁calls▁end｜>"; }
  const char* name() const override { return "DeepSeek"; }

  std::string buildToolSection(const std::vector<Tool>& tools) const override {
    std::ostringstream out;

    out << "You are a helpful assistant with tool calling capabilities. "
        << "When a tool call is needed, you MUST use the following format to "
           "issue the call:\n"
        << toolCallsBegin() << toolCallBegin() << "function" << toolSep()
        << "FUNCTION_NAME\n"
        << "```json\n{\"param1\":\"value1\",\"param2\":\"value2\"}\n```"
        << toolCallEnd() << toolCallsEnd()
        << "\n\nMake sure the JSON is valid.\n"
        << "## Tools\n\n### Function\n\nYou have the following functions "
           "available:\n\n";

    for (const auto& tool : tools) {
      out << "- `" << tool.functionDefinition.name << "`:\n```json\n"
          << tool.toJson() << "\n```\n";
    }

    return out.str();
  }
};

// Get DeepSeek config instance
const TokenizerTemplateConfig* getDeepSeekConfig() {
  static DeepSeekTemplateConfig config;
  return &config;
}

void testChatTemplateWithoutTools(const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Chat Template Without Tools ===\n";

  auto& tokenizer = tt::utils::tokenizers::activeTokenizer();

  std::vector<ChatMessage> messages;
  ChatMessage msg;
  msg.role = "user";
  msg.content = "What's the weather like?";
  messages.push_back(msg);

  std::string result =
      tokenizer.applyChatTemplate(messages, true, std::nullopt);

  // Should not contain tool-related markers when no tools provided
  assert(result.find("tools") == std::string::npos ||
         result.find(config->toolCallsBegin()) == std::string::npos);

  // Should contain the user message
  assert(result.find(msg.content) != std::string::npos);

  std::cout << "✓ Chat template without tools applied correctly\n";
  std::cout << "✅ Test passed!\n";
}

void testChatTemplateWithSingleTool(const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Exact Single Tool Template (" << config->name()
            << ") ===\n";

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

  // Build expected result using tokenizer-specific config
  std::ostringstream expected;
  expected << config->bos();
  expected << config->buildToolSection(tools);
  expected << config->userTag() << "Get weather for SF";
  expected << config->assistantTag();

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
    for (size_t i = 0; i < std::min(expected.str().length(), actual.length());
         ++i) {
      if (expected.str()[i] != actual[i]) {
        std::cout << "First difference at position " << i << ":\n";
        std::cout << "  Expected: '" << expected.str()[i] << "' (ASCII "
                  << (int)expected.str()[i] << ")\n";
        std::cout << "  Actual: '" << actual[i] << "' (ASCII " << (int)actual[i]
                  << ")\n";
        break;
      }
    }
    throw std::runtime_error("Template mismatch");
  }

  std::cout << "✅ Test passed!\n";
}

void testChatTemplateWithMultipleTools(const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Exact Multiple Tools Template (" << config->name()
            << ") ===\n";

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

  // Build expected result using tokenizer-specific config
  std::ostringstream expected;
  expected << config->bos();
  expected << config->buildToolSection(tools);
  expected << config->userTag() << "Check weather and time";
  expected << config->assistantTag();

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
    for (size_t i = 0; i < std::min(expected.str().length(), actual.length());
         ++i) {
      if (expected.str()[i] != actual[i]) {
        std::cout << "First difference at position " << i << ":\n";
        std::cout << "  Expected: '" << expected.str()[i] << "' (ASCII "
                  << (int)expected.str()[i] << ")\n";
        std::cout << "  Actual: '" << actual[i] << "' (ASCII " << (int)actual[i]
                  << ")\n";
        break;
      }
    }
    throw std::runtime_error("Template mismatch for multiple tools");
  }

  std::cout << "✅ Test passed!\n";
}
void testChatTemplateWithConversationHistory(
    const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Exact Conversation History Template ("
            << config->name() << ") ===\n";

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

  // Build expected result using tokenizer-specific config
  // Structure: BOS + SystemMsg + ToolSection + User1 + Assistant1 + User2 +
  // AssistantPrompt
  std::ostringstream expected;
  expected << config->bos();
  expected << config->buildToolSection(tools);
  expected << config->userTag() << "Check SF weather";
  expected << config->assistantTag() << "I'll check for you.";
  // Note: No EOS token between messages in conversation (only if add_eos_token
  // is true)
  expected << config->userTag() << "Also check LA";
  expected << config->assistantTag();

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
    for (size_t i = 0; i < std::min(expected.str().length(), actual.length());
         ++i) {
      if (expected.str()[i] != actual[i]) {
        std::cout << "First difference at position " << i << ":\n";
        std::cout << "  Expected: '" << expected.str()[i] << "' (ASCII "
                  << (int)expected.str()[i] << ")\n";
        std::cout << "  Actual: '" << actual[i] << "' (ASCII " << (int)actual[i]
                  << ")\n";

        // Show context around the difference
        size_t contextStart = (i > 50) ? i - 50 : 0;
        size_t contextEnd = std::min(
            i + 50, std::min(expected.str().length(), actual.length()));
        std::cout << "\nContext (position " << contextStart << " to "
                  << contextEnd << "):\n";
        std::cout << "Expected: \""
                  << expected.str().substr(contextStart,
                                           contextEnd - contextStart)
                  << "\"\n";
        std::cout << "Actual:   \""
                  << actual.substr(contextStart, contextEnd - contextStart)
                  << "\"\n";
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

  assert(!result.empty());
  std::cout << "✓ Chat template with empty tools vector handled\n";

  std::cout << "✅ Test passed!\n";
}

void testToolStructureValidation() {
  std::cout << "\n=== Testing Tool Structure Validation ===\n";

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
    // Get config for the active tokenizer (DeepSeek for now)
    const auto* config = getDeepSeekConfig();

    testChatTemplateWithoutTools(config);
    testChatTemplateWithSingleTool(config);
    testChatTemplateWithMultipleTools(config);
    testChatTemplateWithConversationHistory(config);
    testChatTemplateEmptyTools();
    testToolStructureValidation();

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
