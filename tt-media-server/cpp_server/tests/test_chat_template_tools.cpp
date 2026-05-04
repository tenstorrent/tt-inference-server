// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <cassert>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "domain/chat_message.hpp"
#include "domain/tool_calls/tool.hpp"
#include "utils/tokenizers/deepseek_tokenizer.hpp"
#include "utils/tokenizers/llama_tokenizer.hpp"
#include "utils/tokenizers/tokenizer.hpp"

using namespace tt::domain;
using namespace tt::domain::tool_calls;
using namespace tt::utils::tokenizers;

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

  virtual std::string buildAssistantWithToolCall(
      const ChatMessage& message) const = 0;

  virtual std::string buildToolOutput(const ChatMessage& message) const = 0;

  virtual std::string buildUserMessage(const std::string& content) const = 0;
  virtual std::string buildAssistantMessage(
      const std::string& content) const = 0;

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

  std::string buildAssistantWithToolCall(
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

  std::string buildUserMessage(const std::string& content) const override {
    std::ostringstream out;
    out << userTag() << content;
    return out.str();
  }

  std::string buildAssistantMessage(const std::string& content) const override {
    std::ostringstream out;
    out << assistantTag() << content;
    return out.str();
  }
};

// Get DeepSeek config instance
const TokenizerTemplateConfig* getDeepSeekConfig() {
  static DeepSeekTemplateConfig config;
  return &config;
}

// ============================================================================
// Llama Template Configuration
// ============================================================================

struct LlamaTemplateConfig : public TokenizerTemplateConfig {
  const char* bos() const override { return "<|begin_of_text|>"; }
  const char* userTag() const override {
    return "<|start_header_id|>user<|end_header_id|>\n\n";
  }
  const char* assistantTag() const override {
    return "<|start_header_id|>assistant<|end_header_id|>\n\n";
  }
  const char* toolCallsBegin() const override { return ""; }
  const char* toolCallBegin() const override { return ""; }
  const char* toolSep() const override { return ""; }
  const char* toolCallEnd() const override { return ""; }
  const char* toolCallsEnd() const override { return ""; }
  const char* toolOutputsBegin() const override { return ""; }
  const char* toolOutputBegin() const override {
    return "<|start_header_id|>tool<|end_header_id|>\n\n";
  }
  const char* toolOutputEnd() const override { return ""; }
  const char* toolOutputsEnd() const override { return ""; }
  const char* endOfSentence() const override { return "<|eot_id|>"; }
  const char* name() const override { return "Llama"; }

  // For Llama, this returns system message + user message with tool definitions
  // (everything between BOS and the first user message content)
  std::string buildToolSection(const std::vector<Tool>& tools) const override {
    std::ostringstream out;

    // System header + environment + preamble + content + EOT
    out << "<|start_header_id|>system<|end_header_id|>\n\n";
    out << "Environment: ipython\n";
    out << "Cutting Knowledge Date: December 2023\n";
    out << "Today Date: 26 Jul 2024\n\n";
    out << "You are a helpful assistant with tool calling capabilities. ";
    out << "Only reply with a tool call if the function exists in the library ";
    out << "provided by the user. If it doesn't exist, just reply directly in ";
    out << "natural language. When you receive a tool call response, use the ";
    out << "output to format an answer to the original user question.";
    out << endOfSentence();

    // User header + tool instructions + tool definitions
    out << "<|start_header_id|>user<|end_header_id|>\n\n";
    out << "Given the following functions, please respond with a JSON for a "
           "function call ";
    out << "with its proper arguments that best answers the given prompt.\n\n";
    out << "Respond in the format {\"name\": function name, \"parameters\": ";
    out << "dictionary of argument name and its value}. ";
    out << "Do not use variables.\n\n";

    for (const auto& tool : tools) {
      out << tool.toJson() << "\n\n";
    }

    return out.str();
  }

  std::string buildAssistantWithToolCall(
      const ChatMessage& message) const override {
    std::ostringstream out;

    out << "<|start_header_id|>assistant<|end_header_id|>\n\n";

    // Llama uses JSON format for tool calls
    if (message.tool_calls.has_value() && !message.tool_calls->empty()) {
      const auto& toolCall = (*message.tool_calls)[0];
      out << "{\"name\": \"" << toolCall.functionCall.name << "\", ";
      out << "\"parameters\": ";

      out << toolCall.functionCall.arguments;
      out << "}";
    }

    out << endOfSentence();
    return out.str();
  }

  std::string buildToolOutput(const ChatMessage& message) const override {
    std::ostringstream out;
    out << "<|start_header_id|>tool<|end_header_id|>\n\n";
    out << message.content;
    out << endOfSentence();
    return out.str();
  }

  std::string buildUserMessage(const std::string& content) const override {
    std::ostringstream out;
    out << userTag() << content << endOfSentence();
    return out.str();
  }

  std::string buildAssistantMessage(const std::string& content) const override {
    std::ostringstream out;
    out << assistantTag() << content << endOfSentence();
    return out.str();
  }
};

// Get Llama config instance
const TokenizerTemplateConfig* getLlamaConfig() {
  static LlamaTemplateConfig config;
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

void testChatTemplateWithoutTools(const Tokenizer& tokenizer,
                                  const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Chat Template Without Tools (" << config->name()
            << ") ===\n";

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

void testChatTemplateWithSingleTool(const Tokenizer& tokenizer,
                                    const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Single Tool Template (" << config->name()
            << ") ===\n";

  std::vector<ChatMessage> messages = {createUserMessage("Get weather for SF")};

  std::vector<Tool> tools = {createWeatherTool()};

  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  std::ostringstream expected;
  expected << config->bos();

  if (std::string(config->name()) == "Llama") {
    // For Llama, buildToolSection includes the user header,
    // so we just append the content + eot + assistant tag
    expected << config->buildToolSection(tools);
    expected << "Get weather for SF" << config->endOfSentence();
    expected << config->assistantTag();
  } else {
    // For other tokenizers, build normally
    expected << config->buildToolSection(tools);
    expected << config->userTag() << "Get weather for SF";
    expected << config->assistantTag();
  }

  // Exact match
  if (actual != expected.str()) {
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
        std::cout << "  Actual: '" << actual[i] << "' (ASCII " << int(actual[i])
                  << ")\n";
        break;
      }
    }
    throw std::runtime_error(std::string(config->name()) +
                             " template exact match failed");
  }
  std::cout << "✅ Test passed!\n";
}

void testChatTemplateWithMultipleTools(const Tokenizer& tokenizer,
                                       const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Multiple Tools Template (" << config->name()
            << ") ===\n";

  std::vector<ChatMessage> messages = {
      createUserMessage("Check weather and time")};

  std::vector<Tool> tools = {createWeatherTool(), createTimeTool()};

  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  std::ostringstream expected;
  expected << config->bos();

  if (std::string(config->name()) == "Llama") {
    // For Llama, buildToolSection includes the user header,
    // so we just append the content + eot + assistant tag
    expected << config->buildToolSection(tools);
    expected << "Check weather and time" << config->endOfSentence();
    expected << config->assistantTag();
  } else {
    // For other tokenizers, build normally
    expected << config->buildToolSection(tools);
    expected << config->userTag() << "Check weather and time";
    expected << config->assistantTag();
  }

  // Exact match
  if (actual != expected.str()) {
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
        std::cout << "  Actual: '" << actual[i] << "' (ASCII " << int(actual[i])
                  << ")\n";
        break;
      }
    }
    throw std::runtime_error(std::string(config->name()) +
                             " multiple tools template exact match failed");
  }
  std::cout << "✅ Test passed!\n";
}
void testChatTemplateWithConversationHistory(
    const Tokenizer& tokenizer, const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Conversation History Template (" << config->name()
            << ") ===\n";

  std::vector<ChatMessage> messages = {
      createUserMessage("Check SF weather"),
      createAssistantMessage("I'll check for you."),
      createUserMessage("Also check LA")};

  std::vector<Tool> tools = {createWeatherTool()};

  // Get actual result
  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  std::ostringstream expected;
  expected << config->bos();

  if (std::string(config->name()) == "Llama") {
    // For Llama, buildToolSection includes the first user header,
    // so we just append the content + eot, then remaining messages
    expected << config->buildToolSection(tools);
    expected << "Check SF weather" << config->endOfSentence();
    expected << config->buildAssistantMessage("I'll check for you.");
    expected << config->buildUserMessage("Also check LA");
    expected << config->assistantTag();
  } else {
    // For other tokenizers, build normally
    expected << config->buildToolSection(tools);
    expected << config->buildUserMessage("Check SF weather");
    expected << config->buildAssistantMessage("I'll check for you.");
    expected << config->buildUserMessage("Also check LA");
    expected << config->assistantTag();
  }

  // Exact match
  if (actual != expected.str()) {
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
        std::cout << "  Actual: '" << actual[i] << "' (ASCII " << int(actual[i])
                  << ")\n";
        break;
      }
    }
    throw std::runtime_error(
        std::string(config->name()) +
        " conversation history template exact match failed");
  }
  std::cout << "✅ Test passed!\n";
}

void testChatTemplateEmptyTools(const Tokenizer& tokenizer,
                                const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Chat Template With Empty Tools Vector ("
            << config->name() << ") ===\n";

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

void testChatTemplateWithToolOutputs(const Tokenizer& tokenizer,
                                     const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Chat Template With Tool Outputs ("
            << config->name() << ") ===\n";

  ChatMessage assistantMsg = createAssistantWithToolCall(
      "", "call_123", "get_weather", "{\"location\":\"San Francisco\"}");
  ChatMessage toolMsg = createToolOutputMessage(
      "call_123", "{\"temperature\":72,\"conditions\":\"sunny\"}");

  std::vector<ChatMessage> messages = {
      createUserMessage("What's the weather in SF?"), assistantMsg, toolMsg};

  std::vector<Tool> tools = {createWeatherTool()};

  // Get actual result
  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  // Build expected output using config - EXACT MATCH FOR ALL TOKENIZERS
  std::ostringstream expected;
  expected << config->bos();

  if (std::string(config->name()) == "Llama") {
    // For Llama, buildToolSection includes the first user header,
    // so we just append the content + eot
    expected << config->buildToolSection(tools);
    expected << "What's the weather in SF?" << config->endOfSentence();
    expected << config->buildAssistantWithToolCall(assistantMsg);
    expected << config->toolOutputsBegin();
    expected << config->buildToolOutput(toolMsg);
    expected << config->toolOutputsEnd();
    expected << config->assistantTag();
  } else {
    // For other tokenizers, build normally
    expected << config->buildToolSection(tools);
    expected << config->userTag() << "What's the weather in SF?";
    expected << config->buildAssistantWithToolCall(assistantMsg);
    expected << config->toolOutputsBegin();
    expected << config->buildToolOutput(toolMsg);
    expected << config->toolOutputsEnd();
    expected << config->assistantTag();
  }

  if (actual != expected.str()) {
    std::cout << "❌ Mismatch detected!\n";
    std::cout << "\n=== EXPECTED ===\n" << expected.str() << "\n";
    std::cout << "\n=== ACTUAL ===\n" << actual << "\n";
    std::cout << "\nExpected length: " << expected.str().length() << "\n";
    std::cout << "Actual length: " << actual.length() << "\n";

    for (size_t i = 0; i < std::min(expected.str().length(), actual.length());
         ++i) {
      if (expected.str()[i] != actual[i]) {
        std::cout << "First difference at position " << i << ":\n";
        std::cout << "  Expected: '" << expected.str()[i] << "' (ASCII "
                  << int(expected.str()[i]) << ")\n";
        std::cout << "  Actual: '" << actual[i] << "' (ASCII " << int(actual[i])
                  << ")\n";
        break;
      }
    }
    throw std::runtime_error(std::string(config->name()) +
                             " tool outputs template exact match failed");
  }
  std::cout << "✅ Test passed!\n";
}

void testChatTemplateWithMultipleToolOutputs(
    const Tokenizer& tokenizer, const TokenizerTemplateConfig* config) {
  std::cout << "\n=== Testing Chat Template With Multiple Tool Outputs ("
            << config->name() << ") ===\n";

  // Llama only supports single tool calls, so skip this test
  if (std::string(config->name()) == "Llama") {
    std::cout << "⊘ Skipped (Llama only supports single tool-calls)\n";
    return;
  }

  ChatMessage assistantMsg1 = createAssistantWithToolCall(
      "", "call_1", "get_weather", "{\"location\":\"SF\"}");
  ChatMessage assistantMsg2 = createAssistantWithToolCall(
      "", "call_2", "get_weather", "{\"location\":\"LA\"}");
  ChatMessage toolMsg1 = createToolOutputMessage("call_1", "{\"temp\":72}");
  ChatMessage toolMsg2 = createToolOutputMessage("call_2", "{\"temp\":85}");

  std::vector<ChatMessage> messages = {
      createUserMessage("Get weather for SF and LA"), assistantMsg1, toolMsg1,
      assistantMsg2, toolMsg2};

  std::vector<Tool> tools = {createWeatherTool()};

  std::string actual = tokenizer.applyChatTemplate(messages, true, tools);

  // Verify key components
  assert(actual.find("Get weather for SF and LA") != std::string::npos);
  assert(actual.find("get_weather") != std::string::npos);
  assert(actual.find("\"temp\":72") != std::string::npos);
  assert(actual.find("\"temp\":85") != std::string::npos);

  std::cout << "✅ Test passed!\n";
}

void runTestSuite(const Tokenizer& tokenizer,
                  const TokenizerTemplateConfig* config) {
  testChatTemplateWithoutTools(tokenizer, config);
  testChatTemplateWithSingleTool(tokenizer, config);
  testChatTemplateWithMultipleTools(tokenizer, config);
  testChatTemplateWithConversationHistory(tokenizer, config);
  testChatTemplateEmptyTools(tokenizer, config);
  testToolStructureValidation();
  testChatTemplateWithToolOutputs(tokenizer, config);
  testChatTemplateWithMultipleToolOutputs(tokenizer, config);
}

struct TokenizerEntry {
  std::string path;
  std::function<std::unique_ptr<Tokenizer>(const std::string&)> factory;
  const TokenizerTemplateConfig* config;
};

int main() {
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║      Chat Template with Tools Test Suite                ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";

  try {
    const std::string deepseekPath =
        std::string(TOKENIZER_DIR) +
        "/deepseek-ai/DeepSeek-R1-0528/tokenizer.json";
    const std::string llamaPath =
        std::string(TOKENIZER_DIR) +
        "/meta-llama/Llama-3.1-8B-Instruct/tokenizer.json";

    std::vector<TokenizerEntry> tokenizers = {
        {deepseekPath,
         [](auto p) { return std::make_unique<DeepseekTokenizer>(p); },
         getDeepSeekConfig()},
        {llamaPath, [](auto p) { return std::make_unique<LlamaTokenizer>(p); },
         getLlamaConfig()},
    };

    for (const auto& entry : tokenizers) {
      if (!std::filesystem::exists(entry.path)) {
        continue;
      }
      auto tok = entry.factory(entry.path);
      runTestSuite(*tok, entry.config);
    }

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
