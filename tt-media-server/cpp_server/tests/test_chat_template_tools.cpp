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
  virtual const char* toolOutputsBegin() const = 0;
  virtual const char* toolOutputBegin() const = 0;
  virtual const char* toolOutputEnd() const = 0;
  virtual const char* toolOutputsEnd() const = 0;
  virtual const char* endOfSentence() const = 0;

  // Build sections
  virtual std::string buildToolSection(
      const std::vector<Tool>& tools) const = 0;

  virtual std::string buildAssistantWithToolCalls(
      const ChatMessage& message) const = 0;

  virtual std::string buildToolOutput(const ChatMessage& message) const = 0;

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
  const char* toolOutputsBegin() const override {
    return "<｜tool▁outputs▁begin｜>";
  }
  const char* toolOutputBegin() const override {
    return "<｜tool▁output▁begin｜>";
  }
  const char* toolOutputEnd() const override { return "<｜tool▁output▁end｜>"; }
  const char* toolOutputsEnd() const override {
    return "<｜tool▁outputs▁end｜>";
  }
  const char* endOfSentence() const override { return "<｜end▁of▁sentence｜>"; }
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

  std::string buildAssistantWithToolCalls(
      const ChatMessage& message) const override {
    std::ostringstream out;

    // Optional text content before tool calls
    if (!message.content.empty()) {
      out << assistantTag() << message.content;
    }

    // Tool calls section
    out << toolCallsBegin();
    for (const auto& toolCall : message.tool_calls.value()) {
      out << toolCallBegin() << "function" << toolSep()
          << toolCall.functionCall.name << "\n```json\n"
          << toolCall.functionCall.arguments << "\n```" << toolCallEnd();
    }
    out << toolCallsEnd() << endOfSentence();

    return out.str();
  }

  std::string buildToolOutput(const ChatMessage& message) const override {
    std::ostringstream out;
    out << toolOutputBegin() << message.content << toolOutputEnd();
    return out.str();
  }
};

// Get DeepSeek config instance
const TokenizerTemplateConfig* getDeepSeekConfig() {
  static DeepSeekTemplateConfig config;
  return &config;
}

// ============================================================================
// Common Test Fixtures
// ============================================================================

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

// Create a time tool definition
Tool createTimeTool() {
  Tool tool;
  tool.type = "function";
  tool.functionDefinition.name = "get_time";
  tool.functionDefinition.description = "Get time";

  Json::Value params;
  params["type"] = "object";
  params["properties"]["timezone"]["type"] = "string";
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
  toolCall.functionCall.arguments = arguments;
  msg.tool_calls = std::vector<ToolCall>{toolCall};

  return msg;
}

// Create an assistant message with multiple tool calls
ChatMessage createAssistantWithToolCalls(
    const std::string& content,
    const std::vector<std::tuple<std::string, std::string, std::string>>&
        toolCalls) {
  ChatMessage msg;
  msg.role = "assistant";
  msg.content = content;

  std::vector<ToolCall> calls;
  for (const auto& [id, name, args] : toolCalls) {
    ToolCall toolCall;
    toolCall.id = id;
    toolCall.type = "function";
    toolCall.functionCall.name = name;
    toolCall.functionCall.arguments = args;
    calls.push_back(toolCall);
  }
  msg.tool_calls = calls;

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

// ============================================================================
// Tests
// ============================================================================

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
  assert(result.find(content) != std::string::npos);

  std::cout << "✓ Chat template without tools applied correctly\n";
  std::cout << "✅ Test passed!\n";
}

void testChatTemplateWithSingleTool(const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Exact Single Tool Template (" << config->name()
            << ") ===\n";

  auto& tokenizer = tt::utils::tokenizers::activeTokenizer();

  // Create message
  std::vector<ChatMessage> messages = {createUserMessage("Get weather for SF")};

  // Create tool
  std::vector<Tool> tools = {createWeatherTool()};

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
                  << int(expected.str()[i]) << ")\n";
        std::cout << "  Actual: '" << actual[i] << "' (ASCII "
                  << int(actual[i]) << ")\n";
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
  std::vector<ChatMessage> messages = {
      createUserMessage("Check weather and time")};

  // Create tools
  std::vector<Tool> tools = {createWeatherTool(), createTimeTool()};

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
                  << int(expected.str()[i]) << ")\n";
        std::cout << "  Actual: '" << actual[i] << "' (ASCII "
                  << int(actual[i]) << ")\n";
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

  std::vector<ChatMessage> messages = {
      createUserMessage("Check SF weather"),
      createAssistantMessage("I'll check for you."),
      createUserMessage("Also check LA")};

  std::vector<Tool> tools = {createWeatherTool()};

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
                  << int(expected.str()[i]) << ")\n";
        std::cout << "  Actual: '" << actual[i] << "' (ASCII "
                  << int(actual[i]) << ")\n";

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

void testChatTemplateWithToolOutputs(const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Chat Template With Tool Outputs ("
            << config->name() << ") ===\n";

  auto& tokenizer = tt::utils::tokenizers::activeTokenizer();

  ChatMessage assistantMsg = createAssistantWithToolCall(
      "", "call_123", "get_weather", "{\"location\":\"San Francisco\"}");
  ChatMessage toolMsg = createToolOutputMessage(
      "call_123", "{\"temperature\":72,\"conditions\":\"sunny\"}");

  std::vector<ChatMessage> messages = {
      createUserMessage("What's the weather in SF?"), assistantMsg, toolMsg};

  std::vector<Tool> tools = {createWeatherTool()};

  // Get actual result
  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  // Build expected result
  std::ostringstream expected;
  expected << config->bos();
  expected << config->buildToolSection(tools);
  expected << config->userTag() << "What's the weather in SF?";
  expected << config->buildAssistantWithToolCalls(assistantMsg);
  expected << config->toolOutputsBegin();
  expected << config->buildToolOutput(toolMsg);
  expected << config->toolOutputsEnd();
  expected << config->assistantTag();

  // Compare
  if (actual == expected.str()) {
    std::cout << "✅ Exact match! Tool output template is perfect.\n";
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
                  << int(expected.str()[i]) << ")\n";
        std::cout << "  Actual: '" << actual[i] << "' (ASCII "
                  << int(actual[i]) << ")\n";
        break;
      }
    }
    throw std::runtime_error("Tool output template mismatch");
  }

  std::cout << "✅ Test passed!\n";
}

void testChatTemplateWithMultipleToolOutputs(
    const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Chat Template With Multiple Tool Outputs ("
            << config->name() << ") ===\n";

  auto& tokenizer = tt::utils::tokenizers::activeTokenizer();

  ChatMessage assistantMsg = createAssistantWithToolCalls(
      "", {{"call_1", "get_weather", "{\"location\":\"SF\"}"},
           {"call_2", "get_weather", "{\"location\":\"LA\"}"}});
  ChatMessage toolMsg1 = createToolOutputMessage("call_1", "{\"temp\":72}");
  ChatMessage toolMsg2 = createToolOutputMessage("call_2", "{\"temp\":85}");

  std::vector<ChatMessage> messages = {
      createUserMessage("Get weather for SF and LA"), assistantMsg, toolMsg1,
      toolMsg2};

  std::vector<Tool> tools = {createWeatherTool()};

  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  // Build expected result
  std::ostringstream expected;
  expected << config->bos();
  expected << config->buildToolSection(tools);
  expected << config->userTag() << "Get weather for SF and LA";
  expected << config->buildAssistantWithToolCalls(assistantMsg);
  expected << config->toolOutputsBegin();
  expected << config->buildToolOutput(toolMsg1);
  expected << "\n";  // Newline between multiple tool outputs
  expected << config->buildToolOutput(toolMsg2);
  expected << config->toolOutputsEnd();
  expected << config->assistantTag();

  // Compare
  if (actual == expected.str()) {
    std::cout << "✅ Exact match! Multiple tool outputs template is perfect.\n";
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
                  << int(expected.str()[i]) << ")\n";
        std::cout << "  Actual: '" << actual[i] << "' (ASCII "
                  << int(actual[i]) << ")\n";
        break;
      }
    }
    throw std::runtime_error("Multiple tool outputs template mismatch");
  }

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
    testChatTemplateWithToolOutputs(config);
    testChatTemplateWithMultipleToolOutputs(config);

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
