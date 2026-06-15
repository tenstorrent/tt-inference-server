// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <gtest/gtest.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "domain/llm/chat_message.hpp"
#include "domain/tool_calls/tool.hpp"
#include "utils/tokenizers/deepseek_tokenizer.hpp"
#include "utils/tokenizers/llama_tokenizer.hpp"
#include "utils/tokenizers/tokenizer.hpp"

using namespace tt::domain;
using namespace tt::domain::llm;
using namespace tt::domain::tool_calls;
using namespace tt::utils::tokenizers;

namespace {

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

  std::string buildToolSection(const std::vector<Tool>& tools) const override {
    std::ostringstream out;

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

// Parameterized test fixture for different tokenizers
struct TokenizerTestParam {
  std::string path;
  std::function<std::unique_ptr<Tokenizer>(const std::string&)> factory;
  const TokenizerTemplateConfig* config;
  std::string name;
};

class ChatTemplateToolsTest
    : public ::testing::TestWithParam<TokenizerTestParam> {
 protected:
  void SetUp() override {
    const auto& param = GetParam();
    if (!std::filesystem::exists(param.path)) {
      GTEST_SKIP() << "Tokenizer not found: " << param.path;
    }
    tokenizer = param.factory(param.path);
    config = param.config;
  }

  std::unique_ptr<Tokenizer> tokenizer;
  const TokenizerTemplateConfig* config;
};

TEST_P(ChatTemplateToolsTest, ChatTemplateWithoutTools) {
  std::vector<ChatMessage> messages;
  ChatMessage msg;
  msg.role = "user";
  msg.content = "What's the weather like?";
  messages.push_back(msg);

  std::string result =
      tokenizer->applyChatTemplate(messages, true, std::nullopt);

  // Should not contain tool-related markers when no tools provided
  EXPECT_TRUE(result.find("tools") == std::string::npos ||
              result.find(config->toolCallsBegin()) == std::string::npos);

  // Should contain the user message
  EXPECT_NE(result.find(msg.content), std::string::npos);
}

TEST_P(ChatTemplateToolsTest, SingleToolTemplate) {
  std::vector<ChatMessage> messages = {createUserMessage("Get weather for SF")};

  std::vector<Tool> tools = {createWeatherTool()};

  std::string actual = tokenizer->applyChatTemplate(messages, true, tools);

  std::ostringstream expected;
  expected << config->bos();

  if (std::string(config->name()) == "Llama") {
    expected << config->buildToolSection(tools);
    expected << "Get weather for SF" << config->endOfSentence();
    expected << config->assistantTag();
  } else {
    expected << config->buildToolSection(tools);
    expected << config->userTag() << "Get weather for SF";
    expected << config->assistantTag();
  }

  EXPECT_EQ(actual, expected.str());
}

TEST_P(ChatTemplateToolsTest, MultipleToolsTemplate) {
  std::vector<ChatMessage> messages = {
      createUserMessage("Check weather and time")};

  std::vector<Tool> tools = {createWeatherTool(), createTimeTool()};

  std::string actual = tokenizer->applyChatTemplate(messages, true, tools);

  std::ostringstream expected;
  expected << config->bos();

  if (std::string(config->name()) == "Llama") {
    expected << config->buildToolSection(tools);
    expected << "Check weather and time" << config->endOfSentence();
    expected << config->assistantTag();
  } else {
    expected << config->buildToolSection(tools);
    expected << config->userTag() << "Check weather and time";
    expected << config->assistantTag();
  }

  EXPECT_EQ(actual, expected.str());
}

TEST_P(ChatTemplateToolsTest, ConversationHistoryTemplate) {
  std::vector<ChatMessage> messages = {
      createUserMessage("Check SF weather"),
      createAssistantMessage("I'll check for you."),
      createUserMessage("Also check LA")};

  std::vector<Tool> tools = {createWeatherTool()};

  std::string actual = tokenizer->applyChatTemplate(messages, true, tools);

  std::ostringstream expected;
  expected << config->bos();

  if (std::string(config->name()) == "Llama") {
    expected << config->buildToolSection(tools);
    expected << "Check SF weather" << config->endOfSentence();
    expected << config->buildAssistantMessage("I'll check for you.");
    expected << config->buildUserMessage("Also check LA");
    expected << config->assistantTag();
  } else {
    expected << config->buildToolSection(tools);
    expected << config->buildUserMessage("Check SF weather");
    expected << config->buildAssistantMessage("I'll check for you.");
    expected << config->buildUserMessage("Also check LA");
    expected << config->assistantTag();
  }

  EXPECT_EQ(actual, expected.str());
}

TEST_P(ChatTemplateToolsTest, EmptyToolsVector) {
  std::vector<ChatMessage> messages;
  ChatMessage msg;
  msg.role = "user";
  msg.content = "Hello";
  messages.push_back(msg);

  std::vector<Tool> emptyTools;
  std::string result =
      tokenizer->applyChatTemplate(messages, true, emptyTools);

  EXPECT_FALSE(result.empty());
}

TEST_P(ChatTemplateToolsTest, ToolOutputs) {
  ChatMessage assistantMsg = createAssistantWithToolCall(
      "", "call_123", "get_weather", "{\"location\":\"San Francisco\"}");
  ChatMessage toolMsg = createToolOutputMessage(
      "call_123", "{\"temperature\":72,\"conditions\":\"sunny\"}");

  std::vector<ChatMessage> messages = {
      createUserMessage("What's the weather in SF?"), assistantMsg, toolMsg};

  std::vector<Tool> tools = {createWeatherTool()};

  std::string actual = tokenizer->applyChatTemplate(messages, true, tools);

  std::ostringstream expected;
  expected << config->bos();

  if (std::string(config->name()) == "Llama") {
    expected << config->buildToolSection(tools);
    expected << "What's the weather in SF?" << config->endOfSentence();
    expected << config->buildAssistantWithToolCall(assistantMsg);
    expected << config->toolOutputsBegin();
    expected << config->buildToolOutput(toolMsg);
    expected << config->toolOutputsEnd();
    expected << config->assistantTag();
  } else {
    expected << config->buildToolSection(tools);
    expected << config->userTag() << "What's the weather in SF?";
    expected << config->buildAssistantWithToolCall(assistantMsg);
    expected << config->toolOutputsBegin();
    expected << config->buildToolOutput(toolMsg);
    expected << config->toolOutputsEnd();
    expected << config->assistantTag();
  }

  EXPECT_EQ(actual, expected.str());
}

TEST_P(ChatTemplateToolsTest, MultipleToolOutputs) {
  // Llama only supports single tool calls, so skip this test
  if (std::string(config->name()) == "Llama") {
    GTEST_SKIP() << "Llama only supports single tool-calls";
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

  std::string actual = tokenizer->applyChatTemplate(messages, true, tools);

  // Verify key components
  EXPECT_NE(actual.find("Get weather for SF and LA"), std::string::npos);
  EXPECT_NE(actual.find("get_weather"), std::string::npos);
  EXPECT_NE(actual.find("\"temp\":72"), std::string::npos);
  EXPECT_NE(actual.find("\"temp\":85"), std::string::npos);
}

// Standalone test for tool structure validation (not tokenizer-dependent)
TEST(ToolStructureValidationTest, MatchesOpenAISpecification) {
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

  EXPECT_TRUE(toolJson.isMember("type"));
  EXPECT_EQ(toolJson["type"].asString(), "function");
  EXPECT_TRUE(toolJson.isMember("function"));
  EXPECT_TRUE(toolJson["function"].isMember("name"));
  EXPECT_TRUE(toolJson["function"].isMember("description"));
  EXPECT_TRUE(toolJson["function"].isMember("parameters"));

  auto& funcParams = toolJson["function"]["parameters"];
  EXPECT_TRUE(funcParams.isMember("type"));
  EXPECT_TRUE(funcParams.isMember("properties"));
  EXPECT_TRUE(funcParams.isMember("required"));
}

// Instantiate tests for available tokenizers
INSTANTIATE_TEST_SUITE_P(
    Tokenizers, ChatTemplateToolsTest,
    ::testing::Values(
        TokenizerTestParam{std::string(TOKENIZER_DIR) +
                               "/deepseek-ai/DeepSeek-R1-0528/tokenizer.json",
                           [](const std::string& p) {
                             return std::make_unique<DeepseekTokenizer>(p);
                           },
                           getDeepSeekConfig(), "DeepSeek"},
        TokenizerTestParam{
            std::string(TOKENIZER_DIR) +
                "/meta-llama/Llama-3.1-8B-Instruct/tokenizer.json",
            [](const std::string& p) {
              return std::make_unique<LlamaTokenizer>(p);
            },
            getLlamaConfig(), "Llama"}),
    [](const ::testing::TestParamInfo<TokenizerTestParam>& info) {
      return info.param.name;
    });

}  // namespace
