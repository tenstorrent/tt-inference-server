// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Tool calling integration tests: end-to-end verification of the tokenizer
// and tool_call_parser pipeline.
//
// These tests use MockToolCallRunner to stream predefined token sequences
// that represent raw model output for tool calls. This exercises the full
// pipeline:
//   1. MockToolCallRunner streams DeepSeek-format tool call tokens
//   2. LLMService's StreamDecoder decodes tokens to text
//   3. Tool call parser processes special tokens and text
//   4. SSE response contains properly formatted tool_calls deltas
//
// This is NOT a unit test - it tests the actual integration of tokenizer
// decoding and tool call parsing as they work together in production.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <string>

#include "support/http_client.hpp"
#include "support/http_response.hpp"
#include "support/mock_tool_call_runner.hpp"
#include "support/test_server.hpp"
#include "support/test_worker_main.hpp"
#include "support/tool_call_request.hpp"
#include "support/tool_call_stream.hpp"
#include "utils/logger.hpp"

namespace {

void configureEnv() {
  setenv("LLM_DEVICE_BACKEND", "mock", 1);
  setenv("LLM_MODE", "regular", 1);
  setenv("DEVICE_IDS", "(0)", 1);
  setenv("MAX_NUM_SESSIONS", "4", 1);
}

}  // namespace

class ToolCallingIntegrationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    tt::utils::ZeroOverheadLogger::initialize();
    server = tt::test::TestServer::start();
  }
  static void TearDownTestSuite() { server.reset(); }

  static std::future<std::string> asyncRequest(const std::string& body) {
    return std::async(std::launch::async, [body] {
      return tt::test::sendAndReceive(server->host(), server->port(), body);
    });
  }

  static std::future<std::string> asyncRequest(
      const tt::test::ToolCallRequest& req) {
    return asyncRequest(req.toJson());
  }

  static std::unique_ptr<tt::test::TestServer> server;
};

std::unique_ptr<tt::test::TestServer> ToolCallingIntegrationTest::server;

// ---------------------------------------------------------------------------
// Debug helpers
// ---------------------------------------------------------------------------

void printDebugHeader(const std::string& testName) {
  std::cout << "\n"
            << "============================================================\n"
            << "DEBUG: " << testName << "\n"
            << "============================================================\n";
}

void printRequestBody(const tt::test::ToolCallRequest& request) {
  std::cout << "\n--- 1. REQUEST BODY ---\n" << request.toJson() << "\n";
}

void printTokenSequence(tt::test::MockToolCallRunner& runner) {
  std::cout << "\n--- 3. MODEL RUNNER OUTPUT (token by token) ---\n";
  runner.debugPrint();
}

void printParsedResponse(const tt::test::ToolCallStream& stream) {
  std::cout << "\n--- 4. FINAL PARSED MESSAGE ---\n";
  std::cout << "  ended_with_done: "
            << (stream.endedWithDone() ? "true" : "false") << "\n";
  std::cout << "  finish_reason: " << stream.finishReason().value_or("(none)")
            << "\n";
  std::cout << "  content: \"" << stream.content() << "\"\n";
  std::cout << "  tool_call_count: " << stream.toolCallCount() << "\n";
  for (size_t i = 0; i < stream.toolCallCount(); ++i) {
    const auto& tc = stream.toolCall(i);
    std::cout << "  tool_call[" << i << "]:\n";
    std::cout << "    id: " << tc.id << "\n";
    std::cout << "    type: " << tc.type << "\n";
    std::cout << "    function.name: " << tc.functionName << "\n";
    std::cout << "    function.arguments: " << tc.arguments << "\n";
  }
  std::cout << "  chunk_count: " << stream.chunkCount() << "\n";
}

void printRawResponse(const std::string& rawResponse) {
  std::cout << "\n--- RAW SSE RESPONSE ---\n";
  // Print first 2000 chars to avoid overwhelming output
  if (rawResponse.size() > 2000) {
    std::cout << rawResponse.substr(0, 2000) << "\n... (truncated)\n";
  } else {
    std::cout << rawResponse << "\n";
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Test single tool call: verify tokenizer decodes and parser emits tool_calls
TEST_F(ToolCallingIntegrationTest, SingleToolCall_ParsedCorrectly) {
  printDebugHeader("SingleToolCall_ParsedCorrectly");

  tt::test::MockToolCallRunner runner(server->resultQueue());
  runner.queueToolCall("get_weather", R"({"location":"San Francisco"})");

  auto request = tt::test::ToolCallRequest()
                     .user("What's the weather in San Francisco?")
                     .tool("get_weather", "Get weather for a location",
                           {{"location", "string"}})
                     .toolChoice("auto")
                     .maxTokens(128)
                     .stream();

  // 1. Print request body
  printRequestBody(request);

  // Send request - server will log prompt when processing
  auto responseFuture = asyncRequest(request);

  // Wait for server to process request (prompt is logged at this point)
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  // 3. Print token sequence (what we're about to stream)
  std::cout << "\n--- 2. PROMPT (see server log above) ---\n";
  printTokenSequence(runner);

  // Stream the tool call tokens
  runner.streamTo(seq->taskId);

  // Parse and verify response
  const std::string rawResponse = responseFuture.get();
  printRawResponse(rawResponse);

  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200);
  EXPECT_NE(response.header("content-type").find("text/event-stream"),
            std::string::npos);

  const auto stream = tt::test::ToolCallStream::parse(response);

  // 4. Print parsed response
  printParsedResponse(stream);

  EXPECT_TRUE(stream.endedWithDone());

  // Verify tool call was parsed correctly
  ASSERT_TRUE(stream.hasToolCalls()) << "Expected tool_calls in response";
  EXPECT_EQ(stream.toolCallCount(), 1u);
  EXPECT_EQ(stream.toolCallFunctionName(0), "get_weather");

  // Verify arguments contain the expected JSON
  const auto& args = stream.toolCallArguments(0);
  EXPECT_FALSE(args.empty());
  EXPECT_NE(args.find("San Francisco"), std::string::npos)
      << "Arguments should contain 'San Francisco', got: " << args;

  // Verify finish_reason is tool_calls
  EXPECT_EQ(stream.finishReason(), "tool_calls");
}

// Test multiple tool calls in single response
TEST_F(ToolCallingIntegrationTest, MultipleToolCalls_AllParsedCorrectly) {
  printDebugHeader("MultipleToolCalls_AllParsedCorrectly");

  tt::test::MockToolCallRunner runner(server->resultQueue());
  runner.queueMultiToolCall({
      {"get_weather", R"({"location":"SF"})"},
      {"get_weather", R"({"location":"NYC"})"},
  });

  auto request =
      tt::test::ToolCallRequest()
          .user("What's the weather in SF and NYC?")
          .tool("get_weather", "Get weather", {{"location", "string"}})
          .toolChoice("auto")
          .maxTokens(256)
          .stream();

  printRequestBody(request);

  auto responseFuture = asyncRequest(request);
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  std::cout << "\n--- 2. PROMPT (see server log above) ---\n";
  printTokenSequence(runner);

  runner.streamTo(seq->taskId);

  const std::string rawResponse = responseFuture.get();
  printRawResponse(rawResponse);

  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200);

  const auto stream = tt::test::ToolCallStream::parse(response);
  printParsedResponse(stream);

  EXPECT_TRUE(stream.endedWithDone());

  // Verify both tool calls were parsed
  ASSERT_EQ(stream.toolCallCount(), 2u) << "Expected 2 tool calls";
  EXPECT_EQ(stream.toolCallFunctionName(0), "get_weather");
  EXPECT_EQ(stream.toolCallFunctionName(1), "get_weather");

  // Verify arguments
  EXPECT_NE(stream.toolCallArguments(0).find("SF"), std::string::npos);
  EXPECT_NE(stream.toolCallArguments(1).find("NYC"), std::string::npos);

  EXPECT_EQ(stream.finishReason(), "tool_calls");
}

// Test text before tool call (assistant thinks then calls tool)
TEST_F(ToolCallingIntegrationTest, TextBeforeToolCall_BothParsed) {
  printDebugHeader("TextBeforeToolCall_BothParsed");

  tt::test::MockToolCallRunner runner(server->resultQueue());
  runner.queueTextThenToolCall("Let me check the weather for you.\n",
                               "get_weather", R"({"location":"Boston"})");

  auto request =
      tt::test::ToolCallRequest()
          .user("What's the weather in Boston?")
          .tool("get_weather", "Get weather", {{"location", "string"}})
          .toolChoice("auto")
          .maxTokens(128)
          .stream();

  printRequestBody(request);

  auto responseFuture = asyncRequest(request);
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  std::cout << "\n--- 2. PROMPT (see server log above) ---\n";
  printTokenSequence(runner);

  runner.streamTo(seq->taskId);

  const std::string rawResponse = responseFuture.get();
  printRawResponse(rawResponse);

  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200);

  const auto stream = tt::test::ToolCallStream::parse(response);
  printParsedResponse(stream);

  EXPECT_TRUE(stream.endedWithDone());

  // Should have both content and tool call
  EXPECT_FALSE(stream.content().empty()) << "Expected content before tool call";
  EXPECT_NE(stream.content().find("check the weather"), std::string::npos);

  ASSERT_TRUE(stream.hasToolCalls());
  EXPECT_EQ(stream.toolCallFunctionName(0), "get_weather");
  EXPECT_NE(stream.toolCallArguments(0).find("Boston"), std::string::npos);
}

// Test tool call with complex JSON arguments
TEST_F(ToolCallingIntegrationTest, ComplexJsonArguments_ParsedCorrectly) {
  printDebugHeader("ComplexJsonArguments_ParsedCorrectly");

  tt::test::MockToolCallRunner runner(server->resultQueue());
  runner.queueToolCall(
      "search_files",
      R"({"path":"/src","pattern":"*.cpp","recursive":true,"max_results":10})");

  auto request = tt::test::ToolCallRequest()
                     .user("Find all cpp files in src")
                     .tool("search_files", "Search for files",
                           {{"path", "string"},
                            {"pattern", "string"},
                            {"recursive", "boolean"},
                            {"max_results", "integer"}})
                     .toolChoice("auto")
                     .maxTokens(128)
                     .stream();

  printRequestBody(request);

  auto responseFuture = asyncRequest(request);
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  std::cout << "\n--- 2. PROMPT (see server log above) ---\n";
  printTokenSequence(runner);

  runner.streamTo(seq->taskId);

  const std::string rawResponse = responseFuture.get();
  printRawResponse(rawResponse);

  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200);

  const auto stream = tt::test::ToolCallStream::parse(response);
  printParsedResponse(stream);

  EXPECT_TRUE(stream.endedWithDone());

  ASSERT_TRUE(stream.hasToolCalls());
  EXPECT_EQ(stream.toolCallFunctionName(0), "search_files");

  // Verify complex JSON is preserved
  const auto& args = stream.toolCallArguments(0);
  EXPECT_NE(args.find("/src"), std::string::npos);
  EXPECT_NE(args.find("*.cpp"), std::string::npos);
  EXPECT_NE(args.find("true"), std::string::npos);
  EXPECT_NE(args.find("10"), std::string::npos);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  if (argc >= 3 && std::strcmp(argv[1], "--worker") == 0) {
    return tt::test::runWorkerSubprocess(std::atoi(argv[2]));
  }

  configureEnv();
  tt::utils::ZeroOverheadLogger::initialize();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();

  // Bypass atexit to avoid OpenSSL teardown crashes (see main_integration_test)
  std::_Exit(result);
}
