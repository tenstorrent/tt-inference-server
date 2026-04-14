// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <gtest/gtest.h>

#include "config/settings.hpp"
#include "domain/chat_message.hpp"
#include "domain/tool.hpp"
#include "utils/tokenizer.hpp"

using namespace tt::utils;
using namespace tt::domain;

class ToolCallingTest : public ::testing::Test {
 protected:
  std::unique_ptr<Tokenizer> tok;

  void SetUp() override {
    std::string path =
        tt::config::tokenizerPath(tt::config::ModelType::DEEPSEEK_R1_0528);
    if (path.empty()) {
      GTEST_SKIP() << "DeepSeek tokenizer files not found";
    }
    tok = createTokenizer(tt::config::ModelType::DEEPSEEK_R1_0528, path);
    if (!tok->isLoaded()) {
      FAIL() << "Failed to load DeepSeek tokenizer from: " << path;
    }
  }

  Tokenizer& tokenizer() { return *tok; }

  // Helper to create a test tool
  Tool createTool(const std::string& name, const std::string& description) {
    Tool tool;
    tool.type = "function";
    tool.function.name = name;
    tool.function.description = description;

    // Simple parameters schema
    Json::Value params;
    params["type"] = "object";
    params["properties"]["location"]["type"] = "string";
    params["required"].append("location");
    tool.function.parameters = params;

    return tool;
  }
};

TEST_F(ToolCallingTest, ApplyChatTemplateWithSingleTool) {
  std::vector<ChatMessage> messages = {
      {"user", "What's the weather in SF?"},
  };

  std::vector<Tool> tools = {
      createTool("get_weather", "Get current weather for a location")};

  std::string result = tokenizer().applyChatTemplate(messages, true, tools);

  // Verify the result contains key parts of the tool template
  EXPECT_NE(result.find("## Tools"), std::string::npos)
      << "Should contain Tools section header";
  EXPECT_NE(result.find("### get_weather"), std::string::npos)
      << "Should contain tool name";
  EXPECT_NE(result.find("Get current weather for a location"),
            std::string::npos)
      << "Should contain tool description";
  EXPECT_NE(result.find("Parameters:"), std::string::npos)
      << "Should contain Parameters label";
  EXPECT_NE(result.find(R"("type":"object")"), std::string::npos)
      << "Should contain parameters JSON";
  EXPECT_NE(result.find("IMPORTANT: ALWAYS adhere to this exact format"),
            std::string::npos)
      << "Should contain format instructions";
  EXPECT_NE(result.find("<｜tool▁calls▁begin｜>"), std::string::npos)
      << "Should contain tool_calls_begin token";
  EXPECT_NE(result.find("<｜User｜>What's the weather in SF?"),
            std::string::npos)
      << "Should contain user message";
  EXPECT_NE(result.find("<｜Assistant｜>"), std::string::npos)
      << "Should end with Assistant tag";
}

TEST_F(ToolCallingTest, ApplyChatTemplateWithMultipleTools) {
  std::vector<ChatMessage> messages = {
      {"user", "Search for info"},
  };

  std::vector<Tool> tools = {
      createTool("get_weather", "Get weather"),
      createTool("search_web", "Search the web"),
  };

  std::string result = tokenizer().applyChatTemplate(messages, true, tools);

  // Verify both tools are included
  EXPECT_NE(result.find("### get_weather"), std::string::npos)
      << "Should contain first tool";
  EXPECT_NE(result.find("### search_web"), std::string::npos)
      << "Should contain second tool";
}

TEST_F(ToolCallingTest, ApplyChatTemplateWithoutTools) {
  std::vector<ChatMessage> messages = {
      {"user", "Hello"},
  };

  std::string result =
      tokenizer().applyChatTemplate(messages, true, std::nullopt);

  // Verify no tool template is included
  EXPECT_EQ(result.find("## Tools"), std::string::npos)
      << "Should not contain Tools section when no tools provided";
  EXPECT_NE(result.find("<｜User｜>Hello"), std::string::npos)
      << "Should still contain user message";
}

TEST_F(ToolCallingTest, ApplyChatTemplateWithToolCalls) {
  std::vector<ChatMessage> messages = {
      {"user", "What's the weather in SF?"},
  };

  // Create assistant message with tool call
  ChatMessage assistantMsg;
  assistantMsg.role = "assistant";
  assistantMsg.content = "";

  ToolCall tc;
  tc.id = "call_123";
  tc.type = "function";
  tc.function.name = "get_weather";
  tc.function.arguments = R"({"location":"San Francisco"})";

  assistantMsg.tool_calls = std::vector<ToolCall>{tc};
  messages.push_back(assistantMsg);

  std::vector<Tool> tools = {createTool("get_weather", "Get weather")};

  std::string result = tokenizer().applyChatTemplate(messages, false, tools);

  // Verify tool call formatting
  EXPECT_NE(result.find("<｜tool▁calls▁begin｜>"), std::string::npos)
      << "Should contain tool_calls_begin";
  EXPECT_NE(result.find("<｜tool▁call▁begin｜>"), std::string::npos)
      << "Should contain tool_call_begin";
  EXPECT_NE(result.find("function<｜tool▁sep｜>get_weather"), std::string::npos)
      << "Should contain function name with separator";
  EXPECT_NE(result.find(R"({"location":"San Francisco"})"), std::string::npos)
      << "Should contain tool arguments";
  EXPECT_NE(result.find("<｜tool▁call▁end｜>"), std::string::npos)
      << "Should contain tool_call_end";
  EXPECT_NE(result.find("<｜tool▁calls▁end｜>"), std::string::npos)
      << "Should contain tool_calls_end";
  EXPECT_NE(result.find("<｜end▁of▁sentence｜>"), std::string::npos)
      << "Should contain end_of_sentence after tool calls";
}

TEST_F(ToolCallingTest, ApplyChatTemplateWithToolResults) {
  std::vector<ChatMessage> messages = {
      {"user", "What's the weather in SF?"},
  };

  // Assistant with tool call
  ChatMessage assistantMsg;
  assistantMsg.role = "assistant";
  assistantMsg.content = "";

  ToolCall tc;
  tc.id = "call_123";
  tc.type = "function";
  tc.function.name = "get_weather";
  tc.function.arguments = R"({"location":"San Francisco"})";
  assistantMsg.tool_calls = std::vector<ToolCall>{tc};
  messages.push_back(assistantMsg);

  // Tool result
  ChatMessage toolMsg;
  toolMsg.role = "tool";
  toolMsg.tool_call_id = "call_123";
  toolMsg.content = "72°F and sunny";
  messages.push_back(toolMsg);

  std::vector<Tool> tools = {createTool("get_weather", "Get weather")};

  std::string result = tokenizer().applyChatTemplate(messages, true, tools);

  // Verify tool output formatting
  EXPECT_NE(result.find("<｜tool▁outputs▁begin｜>"), std::string::npos)
      << "Should contain tool_outputs_begin";
  EXPECT_NE(result.find("<｜tool▁output▁begin｜>"), std::string::npos)
      << "Should contain tool_output_begin";
  EXPECT_NE(result.find("72°F and sunny"), std::string::npos)
      << "Should contain tool output content";
  EXPECT_NE(result.find("<｜tool▁output▁end｜>"), std::string::npos)
      << "Should contain tool_output_end";
  EXPECT_NE(result.find("<｜tool▁outputs▁end｜>"), std::string::npos)
      << "Should contain tool_outputs_end";
}

TEST_F(ToolCallingTest, CompletToolConversation) {
  std::vector<ChatMessage> messages;

  // User asks
  messages.push_back({"user", "What's the weather in SF?"});

  // Assistant makes tool call
  ChatMessage assistantCall;
  assistantCall.role = "assistant";
  assistantCall.content = "";
  ToolCall tc;
  tc.id = "call_123";
  tc.type = "function";
  tc.function.name = "get_weather";
  tc.function.arguments = R"({"location":"San Francisco"})";
  assistantCall.tool_calls = std::vector<ToolCall>{tc};
  messages.push_back(assistantCall);

  // Tool result
  ChatMessage toolResult;
  toolResult.role = "tool";
  toolResult.tool_call_id = "call_123";
  toolResult.content = "72°F and sunny";
  messages.push_back(toolResult);

  // User asks follow-up
  messages.push_back({"user", "How about NYC?"});

  std::vector<Tool> tools = {createTool("get_weather", "Get weather")};

  std::string result = tokenizer().applyChatTemplate(messages, true, tools);

  // Verify the complete flow
  EXPECT_NE(result.find("## Tools"), std::string::npos);
  EXPECT_NE(result.find("<｜User｜>What's the weather in SF?"),
            std::string::npos);
  EXPECT_NE(result.find("<｜tool▁calls▁begin｜>"), std::string::npos);
  EXPECT_NE(result.find("<｜tool▁outputs▁begin｜>"), std::string::npos);
  EXPECT_NE(result.find("72°F and sunny"), std::string::npos);
  EXPECT_NE(result.find("<｜tool▁outputs▁end｜>"), std::string::npos);
  EXPECT_NE(result.find("<｜User｜>How about NYC?"), std::string::npos);
  EXPECT_NE(result.find("<｜Assistant｜>"), std::string::npos);

  // Print for manual inspection
  std::cout << "\n=== Complete Tool Conversation Template ===\n"
            << result << "\n===========================================\n";
}
