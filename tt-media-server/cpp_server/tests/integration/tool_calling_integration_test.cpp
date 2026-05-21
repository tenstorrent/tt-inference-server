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
// Tests
// ---------------------------------------------------------------------------

// Test single tool call: verify tokenizer decodes and parser emits tool_calls
TEST_F(ToolCallingIntegrationTest, SingleToolCall_ParsedCorrectly) {
  tt::test::MockToolCallRunner runner(server->resultQueue());
  runner.queueToolCall("get_weather", R"({"location":"San Francisco"})");

  auto request = tt::test::ToolCallRequest()
                     .user("What's the weather in San Francisco?")
                     .tool("get_weather", "Get weather for a location",
                           {{"location", "string"}})
                     .toolChoice("auto")
                     .maxTokens(128)
                     .stream();

  // Send request - server will log prompt when processing
  auto responseFuture = asyncRequest(request);

  // Wait for server to process request (prompt is logged at this point)
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  // Stream the tool call tokens
  runner.streamTo(seq->taskId);

  // Parse and verify response
  const std::string rawResponse = responseFuture.get();

  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200);
  EXPECT_NE(response.header("content-type").find("text/event-stream"),
            std::string::npos);

  const auto stream = tt::test::ToolCallStream::parse(response);

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

  auto responseFuture = asyncRequest(request);
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  runner.streamTo(seq->taskId);

  const std::string rawResponse = responseFuture.get();

  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200);

  const auto stream = tt::test::ToolCallStream::parse(response);

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

  auto responseFuture = asyncRequest(request);
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  runner.streamTo(seq->taskId);

  const std::string rawResponse = responseFuture.get();

  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200);

  const auto stream = tt::test::ToolCallStream::parse(response);

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

  auto responseFuture = asyncRequest(request);
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  runner.streamTo(seq->taskId);

  const std::string rawResponse = responseFuture.get();

  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200);

  const auto stream = tt::test::ToolCallStream::parse(response);

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

// Test with realistic agentic tools (edit, exec, read, write, web_search)
TEST_F(ToolCallingIntegrationTest, RealisticAgenticTools_EditFile) {
  tt::test::MockToolCallRunner runner(server->resultQueue());
  runner.queueToolCall(
      "edit",
      R"json({"path":"src/main.cpp","edits":[{"oldText":"int main()","newText":"int main(int argc, char** argv)"}]})json");

  // clang-format off
  auto request =
      tt::test::ToolCallRequest()
          .user("Add argc and argv parameters to the main function in "
                "src/main.cpp")
          .toolFromJson(R"json({
            "type": "function",
            "function": {
              "name": "edit",
              "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file.",
              "parameters": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                  "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative or absolute)"
                  },
                  "edits": {
                    "type": "array",
                    "description": "One or more targeted replacements.",
                    "items": {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "oldText": {
                          "type": "string",
                          "description": "Exact text to replace. Must be unique in the file."
                        },
                        "newText": {
                          "type": "string",
                          "description": "Replacement text."
                        }
                      },
                      "required": ["oldText", "newText"]
                    }
                  }
                },
                "required": ["path", "edits"]
              }
            }
          })json")
          .toolFromJson(R"json({
            "type": "function",
            "function": {
              "name": "read",
              "description": "Read the contents of a file. Supports text files and images.",
              "parameters": {
                "type": "object",
                "properties": {
                  "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                  },
                  "offset": {
                    "type": "number",
                    "description": "Line number to start reading from (1-indexed)"
                  },
                  "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to read"
                  }
                },
                "required": ["path"]
              }
            }
          })json")
          .toolFromJson(R"json({
            "type": "function",
            "function": {
              "name": "exec",
              "description": "Execute shell commands with background continuation.",
              "parameters": {
                "type": "object",
                "properties": {
                  "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                  },
                  "workdir": {
                    "type": "string",
                    "description": "Working directory"
                  },
                  "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds"
                  },
                  "background": {
                    "type": "boolean",
                    "description": "Run in background immediately"
                  }
                },
                "required": ["command"]
              }
            }
          })json")
          .toolFromJson(R"json({
            "type": "function",
            "function": {
              "name": "write",
              "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
              "parameters": {
                "type": "object",
                "properties": {
                  "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                  },
                  "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                  }
                },
                "required": ["path", "content"]
              }
            }
          })json")
          .toolFromJson(R"json({
            "type": "function",
            "function": {
              "name": "web_search",
              "description": "Search the web. Returns provider-normalized results for current information lookup.",
              "parameters": {
                "type": "object",
                "properties": {
                  "query": {
                    "type": "string",
                    "description": "Search query string"
                  },
                  "count": {
                    "type": "number",
                    "description": "Number of results to return",
                    "minimum": 1,
                    "maximum": 10
                  },
                  "freshness": {
                    "type": "string",
                    "description": "Filter by time: day, week, month, or year"
                  }
                },
                "required": ["query"]
              }
            }
          })json")
          .toolChoice("auto")
          .maxTokens(256)
          .stream();
  // clang-format on

  auto responseFuture = asyncRequest(request);
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  runner.streamTo(seq->taskId);

  const std::string rawResponse = responseFuture.get();

  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200);

  const auto stream = tt::test::ToolCallStream::parse(response);

  EXPECT_TRUE(stream.endedWithDone());

  ASSERT_TRUE(stream.hasToolCalls());
  EXPECT_EQ(stream.toolCallFunctionName(0), "edit");

  // Verify the edit arguments contain expected structure
  const auto& args = stream.toolCallArguments(0);
  EXPECT_NE(args.find("src/main.cpp"), std::string::npos);
  EXPECT_NE(args.find("oldText"), std::string::npos);
  EXPECT_NE(args.find("newText"), std::string::npos);
  EXPECT_NE(args.find("int main"), std::string::npos);
}

// Test exec tool call
TEST_F(ToolCallingIntegrationTest, RealisticAgenticTools_ExecCommand) {
  tt::test::MockToolCallRunner runner(server->resultQueue());
  runner.queueToolCall(
      "exec", R"json({"command":"make -j8 && ./run_tests","timeout":300})json");

  // clang-format off
  auto request =
      tt::test::ToolCallRequest()
          .user("Build the project and run tests")
          .toolFromJson(R"json({
            "type": "function",
            "function": {
              "name": "exec",
              "description": "Execute shell commands with background continuation.",
              "parameters": {
                "type": "object",
                "properties": {
                  "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                  },
                  "workdir": {
                    "type": "string",
                    "description": "Working directory"
                  },
                  "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds"
                  },
                  "background": {
                    "type": "boolean",
                    "description": "Run in background immediately"
                  }
                },
                "required": ["command"]
              }
            }
          })json")
          .toolChoice("auto")
          .maxTokens(128)
          .stream();
  // clang-format on

  auto responseFuture = asyncRequest(request);
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  runner.streamTo(seq->taskId);

  const std::string rawResponse = responseFuture.get();

  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200);

  const auto stream = tt::test::ToolCallStream::parse(response);

  EXPECT_TRUE(stream.endedWithDone());

  ASSERT_TRUE(stream.hasToolCalls());
  EXPECT_EQ(stream.toolCallFunctionName(0), "exec");

  const auto& args = stream.toolCallArguments(0);
  EXPECT_NE(args.find("make"), std::string::npos);
  EXPECT_NE(args.find("run_tests"), std::string::npos);
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
