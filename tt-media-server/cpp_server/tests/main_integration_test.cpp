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

#include "domain/manage_memory.hpp"
#include "ipc/result_queue.hpp"
#include "support/chat_request.hpp"
#include "support/http_client.hpp"
#include "support/http_response.hpp"
#include "support/test_server.hpp"
#include "support/test_worker_main.hpp"
#include "support/worker_response.hpp"
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

  // Mock the worker producing one output token + a clean final marker.
  // Tests that need a custom token stream use tt::test::WorkerResponse
  // directly.
  static void mockWorkerResponse(uint32_t taskId, uint64_t tokenId = 42) {
    tt::test::WorkerResponse(taskId).token(tokenId).finalize().sendTo(
        server_->resultQueue());
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

  const auto response = tt::test::HttpResponse::parse(responseFuture.get());
  EXPECT_EQ(response.statusCode(), 200);
  EXPECT_NE(response.header("content-type").find("application/json"),
            std::string::npos);
  const auto body = response.json();
  ASSERT_TRUE(body.isMember("choices"));
  EXPECT_EQ(body["choices"].size(), 1u);
}

TEST_F(MainIntegrationTest, WorkerResponseBuilder_MultipleTokensThenFinalize) {
  // Demonstrate the WorkerResponse builder + HttpResponse parser working
  // together: ask for up to 3 tokens, push exactly 3 specific token_ids
  // followed by a clean FINAL terminator, then assert on the parsed JSON.
  auto responseFuture =
      asyncRequest(ChatRequest().user("hello").maxTokens(3));

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  tt::test::WorkerResponse(seq->taskId)
      .tokens({101, 202, 303})
      .finalize()
      .sendTo(server_->resultQueue());

  const auto response = tt::test::HttpResponse::parse(responseFuture.get());
  EXPECT_EQ(response.statusCode(), 200);

  const auto body = response.json();
  ASSERT_TRUE(body.isMember("choices"));
  EXPECT_EQ(body["choices"].size(), 1u);
  ASSERT_TRUE(body.isMember("usage"));
  // The controller's completion_tokens accounting is implementation-defined
  // around the FINAL marker — assert positive and bounded by what we pushed.
  const int completionTokens = body["usage"]["completion_tokens"].asInt();
  EXPECT_GT(completionTokens, 0);
  EXPECT_LE(completionTokens, 3);
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

TEST_F(MainIntegrationTest, MemoryAllocate_RequestPushedAndResponseAccepted) {
  // Disable the auto-responder so we can inspect what SessionManager pushed
  // to the memory request queue, and inject the SUCCESS response ourselves.
  // Restored at end of test so the next test sees default behavior.
  server_->setMemoryAutoRespond(false);

  auto future = asyncRequest(ChatRequest().user("hello").maxTokens(1));

  // SessionManager should push exactly one ALLOCATE for the new session.
  tt::domain::ManageMemoryTask memReq{};
  server_->memoryRequestQueue().receive(memReq);
  EXPECT_EQ(memReq.action, tt::domain::MemoryManagementAction::ALLOCATE);
  EXPECT_GT(memReq.taskId, 0u);

  // Mock the memory manager: SUCCESS unblocks SessionManager's allocation.
  tt::domain::ManageMemoryResult memRes{};
  memRes.taskId = memReq.taskId;
  memRes.status = tt::domain::ManageMemoryStatus::SUCCESS;
  memRes.slotId = 0;
  server_->memoryResultQueue().push(memRes);

  // Once we've answered the ALLOCATE, SessionManager unblocks and the
  // controller pushes the Sequence onto the task queue. Without our
  // response, this receive() would hang.
  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  mockWorkerResponse(seq->taskId);
  future.get();

  server_->setMemoryAutoRespond(true);
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
  const int result = RUN_ALL_TESTS();

  // Bypass atexit / static destructors to dodge a segfault during OpenSSL's
  // OPENSSL_cleanup() — Drogon and the embedded Python interpreter both pull
  // in libssl, and their interleaved teardown corrupts OpenSSL's ex_data hash
  // table. The crash is unrelated to the tests; it would also turn a clean
  // exit-0 into exit-139 and fail CI.
  //
  // Safe to skip:
  //   - Drogon listener already quit in TearDownTestSuite.
  //   - Worker subprocess sets PR_SET_PDEATHSIG=SIGTERM so the kernel reaps it.
  //   - Boost.Interprocess shm in /dev/shm leaks but is reused on next run.
  std::_Exit(result);
}
