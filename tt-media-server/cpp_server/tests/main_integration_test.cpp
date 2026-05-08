// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Main integration test: gray-box round-trip verification of LLMController.
//
// Each test fires an HTTP request, then inspects what the controller pushed
// to the IPC task queue (the boundary between cpp_server and the worker).
// The fixture also mocks the worker by pushing tokens to the result queue,
// so HTTP responses complete and can be asserted on.
//
// Test infrastructure lives under tests/support/:
//   - TestServer          : brings the full server stack up in-process
//   - sendAndReceive      : blocking HTTP POST helper
//   - ChatRequest         : fluent /v1/chat/completions body builder
//   - runWorkerSubprocess : --worker re-exec entry point

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <string>

#include "ipc/result_queue.hpp"
#include "support/chat_request.hpp"
#include "support/http_client.hpp"
#include "support/test_server.hpp"
#include "support/test_worker_main.hpp"
#include "utils/logger.hpp"

namespace {

void configureEnv() {
  setenv("LLM_DEVICE_BACKEND", "mock", 1);
  setenv("LLM_MODE", "regular", 1);
  setenv("DEVICE_IDS", "(0)", 1);
  setenv("MAX_NUM_SESSIONS", "4", 1);
}

}  // namespace

class MainIntegrationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    tt::utils::ZeroOverheadLogger::initialize();
    server_ = tt::test::TestServer::start();
  }
  static void TearDownTestSuite() { server_.reset(); }

  // Fire request in background. Returns future for the raw HTTP response.
  static std::future<std::string> asyncRequest(const std::string& body) {
    return std::async(std::launch::async, [body] {
      return tt::test::sendAndReceive(server_->host(), server_->port(), body);
    });
  }
  static std::future<std::string> asyncRequest(
      const tt::test::ChatRequest& req) {
    return asyncRequest(req.toJson());
  }

  // Mock the worker producing one output token + final marker.
  static void mockWorkerResponse(uint32_t taskId, uint64_t tokenId = 42) {
    tt::ipc::SharedToken tok{};
    tok.task_id = taskId;
    tok.token_id = tokenId;
    server_->resultQueue().push(tok);

    tt::ipc::SharedToken fin{};
    fin.task_id = taskId;
    fin.token_id = 0;
    fin.flags = tt::ipc::SharedToken::FLAG_FINAL;
    server_->resultQueue().push(fin);
  }

  static std::unique_ptr<tt::test::TestServer> server_;
};

std::unique_ptr<tt::test::TestServer> MainIntegrationTest::server_;

using tt::test::ChatRequest;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(MainIntegrationTest, SingleRequest_TaskQueueAndResponse) {
  auto responseFuture = asyncRequest(ChatRequest().user("hello").maxTokens(1));

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_GT(seq->getNumPromptTokens(), 0u);
  EXPECT_FALSE(seq->isContinuation());

  mockWorkerResponse(seq->taskId);

  const auto response = responseFuture.get();
  EXPECT_NE(response.find("200"), std::string::npos);
  EXPECT_NE(response.find("choices"), std::string::npos);
}

TEST_F(MainIntegrationTest, MultiTurn_AllRequestsAfterFirstAreContinuations) {
  // Each turn appends a new user message to the running conversation. Turn
  // N+1's controller-side prefix lookup hash matches turn N's history, so
  // every turn after the first must be flagged as a continuation.
  ChatRequest convo;
  const std::vector<std::string> userMessages = {
      "hello",
      "how are you",
      "tell me a joke",
      "thanks",
  };

  for (size_t i = 0; i < userMessages.size(); ++i) {
    convo.user(userMessages[i]).maxTokens(1);
    auto future = asyncRequest(convo);

    auto seq = server_->taskQueue().receive();
    ASSERT_NE(seq, nullptr) << "turn " << i;
    EXPECT_EQ(seq->isContinuation(), i > 0) << "turn " << i;

    mockWorkerResponse(seq->taskId);
    future.get();

    convo.assistant("ok");  // simulate the assistant reply for the next turn
  }
}

TEST_F(MainIntegrationTest, StreamingRequest_AlsoPushesToTaskQueue) {
  // stream=true goes through a different controller path but must still
  // push a Sequence to the task queue.
  auto future = asyncRequest(ChatRequest().user("hello").maxTokens(1).stream());

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_GT(seq->getNumPromptTokens(), 0u);
  EXPECT_FALSE(seq->isContinuation());

  mockWorkerResponse(seq->taskId);
  future.get();
}

TEST_F(MainIntegrationTest, SamplingParams_MaxTokensAndTemperature) {
  auto future =
      asyncRequest(ChatRequest().user("hello").maxTokens(42).temperature(0.7));

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  const auto& params = seq->getSamplingParams();
  EXPECT_EQ(params.max_tokens, 42);
  EXPECT_NEAR(params.temperature, 0.7f, 1e-4f);

  mockWorkerResponse(seq->taskId);
  future.get();
}

TEST_F(MainIntegrationTest, DisaggregatedFlag_IsFalse_InRegularMode) {
  // LLM_MODE=regular: every request is served locally, never disaggregated.
  auto future = asyncRequest(ChatRequest().user("hello").maxTokens(1));

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_FALSE(seq->isDisaggregated());

  mockWorkerResponse(seq->taskId);
  future.get();
}

TEST_F(MainIntegrationTest, SystemMessage_DoesNotTriggerContinuation) {
  // A system + user message is a first turn even though there are two messages.
  auto future = asyncRequest(
      ChatRequest().system("you are helpful").user("hello").maxTokens(1));

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_FALSE(seq->isContinuation());

  mockWorkerResponse(seq->taskId);
  future.get();
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
  return RUN_ALL_TESTS();
}
