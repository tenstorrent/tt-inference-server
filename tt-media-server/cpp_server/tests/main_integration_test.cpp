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
#include "support/chat_completion_stream.hpp"
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

// Canonical happy-path walkthrough of /v1/chat/completions, in streaming mode
// (the primary use case for this server). Covers a fresh request end-to-end
// and then a continuation that reuses the same session via prefix-cache.
// Each subsequent test peels off a slice of this lifecycle and asserts on
// just that slice — start here when reading the suite.
//
//   1. HTTP POST arrives at the controller
//   2. SessionManager pushes an ALLOCATE onto the memory request queue
//   3. We mock the memory manager: push SUCCESS to the result queue
//   4. Controller pushes a Sequence onto the task queue
//   5. We mock the worker: push one token + FINAL via WorkerResponse
//   6. SSE stream terminates; assert on the events delivered
//   7. A follow-up request with the same opener HITs the prefix-cache:
//      no new ALLOCATE, the Sequence is flagged as a continuation, and
//      only the delta prompt (the new user turn) is sent to the task
//      queue — not the full prior conversation.
TEST_F(MainIntegrationTest, HappyPath_RequestToMemoryToTaskToResponse) {
  server_->setMemoryAutoRespond(false);

  // 1. Fire the streaming request.
  auto responseFuture =
      asyncRequest(ChatRequest().user("hello").maxTokens(1).stream());

  // 2. Receive and assert on the ALLOCATE.
  tt::domain::ManageMemoryTask memReq{};
  server_->memoryRequestQueue().receive(memReq);
  EXPECT_EQ(memReq.action, tt::domain::MemoryManagementAction::ALLOCATE);
  EXPECT_GT(memReq.taskId, 0u);

  // 3. Mock the memory manager.
  tt::domain::ManageMemoryResult memRes{};
  memRes.taskId = memReq.taskId;
  memRes.status = tt::domain::ManageMemoryStatus::SUCCESS;
  memRes.slotId = 0;
  server_->memoryResultQueue().push(memRes);

  // 4. Receive and assert on the Sequence.
  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_GT(seq->getNumPromptTokens(), 0u);
  EXPECT_FALSE(seq->isContinuation());

  // 5. Mock the worker: one token, then FINAL.
  tt::test::WorkerResponse(seq->taskId)
      .token(42)
      .finalize()
      .sendTo(server_->resultQueue());

  // 6. Assert on the SSE stream.
  const auto response = tt::test::HttpResponse::parse(responseFuture.get());
  EXPECT_EQ(response.statusCode(), 200);
  EXPECT_NE(response.header("content-type").find("text/event-stream"),
            std::string::npos);

  const auto stream = tt::test::ChatCompletionStream::parse(response);
  EXPECT_TRUE(stream.endedWithDone());
  EXPECT_EQ(stream.initialRole(), "assistant");
  EXPECT_FALSE(stream.contentDeltas().empty())
      << "expected at least one content delta";

  // 7. Follow-up with the same opener but a long claimed assistant turn
  //    in between. If the controller sent the full conversation to the
  //    worker (no cache hit), the prompt would include all those tokens.
  //    With the cache HIT, only the delta — the trailing user turn —
  //    is sent; the prompt token count stays close to a single-turn one.
  const std::string longPriorAssistant =
      "this is intentionally a long assistant turn so that if the controller "
      "sent the full conversation history to the worker the prompt would "
      "balloon to many more tokens than the small delta of the last user "
      "turn alone";
  auto followUpFuture = asyncRequest(ChatRequest()
                                         .user("hello")
                                         .assistant(longPriorAssistant)
                                         .user("y")
                                         .maxTokens(1)
                                         .stream());
  auto followUpSeq = server_->taskQueue().receive();
  ASSERT_NE(followUpSeq, nullptr);
  EXPECT_TRUE(followUpSeq->isContinuation())
      << "follow-up should HIT the seed session";
  // "y" tokenises to a single token. With the cache hit, the delta prompt
  // is exactly that token wrapped in the chat-template markers — 4 tokens
  // for the DeepSeek tokenizer (BOS + user-marker + "y" + assistant-marker).
  // If the full conversation had been sent, the long prior assistant turn
  // would have pushed this well into the dozens.
  EXPECT_EQ(followUpSeq->getNumPromptTokens(), 4u);
  tt::test::WorkerResponse(followUpSeq->taskId)
      .token(43)
      .finalize()
      .sendTo(server_->resultQueue());
  followUpFuture.get();

  EXPECT_TRUE(server_->memoryRequestQueue().empty())
      << "follow-up should HIT the cache; no new ALLOCATE expected";

  server_->setMemoryAutoRespond(true);
}

TEST_F(MainIntegrationTest, WorkerResponseBuilder_MultipleTokensThenFinalize) {
  // Push 3 specific tokens via WorkerResponse and assert the SSE stream
  // delivered them as separate content deltas, terminated by [DONE].
  auto responseFuture =
      asyncRequest(ChatRequest().user("hello").maxTokens(3).stream());

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  tt::test::WorkerResponse(seq->taskId)
      .tokens({101, 202, 303})
      .finalize()
      .sendTo(server_->resultQueue());

  const auto response = tt::test::HttpResponse::parse(responseFuture.get());
  EXPECT_EQ(response.statusCode(), 200);

  const auto stream = tt::test::ChatCompletionStream::parse(response);
  EXPECT_TRUE(stream.endedWithDone());
  // Controller's exact accounting around FINAL is implementation-defined;
  // we expect roughly one content delta per token we pushed.
  EXPECT_GT(stream.contentDeltas().size(), 0u);
  EXPECT_LE(stream.contentDeltas().size(), 3u);
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
    convo.user(userMessages[i]).maxTokens(1).stream();
    auto future = asyncRequest(convo);

    auto seq = server_->taskQueue().receive();
    ASSERT_NE(seq, nullptr) << "turn " << i;
    EXPECT_EQ(seq->isContinuation(), i > 0) << "turn " << i;

    mockWorkerResponse(seq->taskId);
    future.get();

    convo.assistant("ok");  // simulate the assistant reply for the next turn
  }
}

TEST_F(MainIntegrationTest, NonStreamingRequest_ReturnsBufferedJson) {
  // Most tests use streaming; this one verifies the non-streaming code path
  // still returns a single buffered JSON document with the assistant message.
  auto responseFuture = asyncRequest(ChatRequest().user("hello").maxTokens(1));

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_GT(seq->getNumPromptTokens(), 0u);

  mockWorkerResponse(seq->taskId);

  const auto response = tt::test::HttpResponse::parse(responseFuture.get());
  EXPECT_EQ(response.statusCode(), 200);
  EXPECT_NE(response.header("content-type").find("application/json"),
            std::string::npos);
  const auto body = response.json();
  ASSERT_EQ(body["choices"].size(), 1u);
  EXPECT_EQ(body["choices"][0]["message"]["role"].asString(), "assistant");
  EXPECT_FALSE(body["choices"][0]["message"]["content"].asString().empty());
}

TEST_F(MainIntegrationTest, SamplingParams_MaxTokensAndTemperature) {
  auto future = asyncRequest(
      ChatRequest().user("hello").maxTokens(42).temperature(0.7).stream());

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
  auto future = asyncRequest(ChatRequest().user("hello").maxTokens(1).stream());

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_FALSE(seq->isDisaggregated());

  mockWorkerResponse(seq->taskId);
  future.get();
}

TEST_F(MainIntegrationTest, TwoFirstTurns_EachAllocatesDistinctSlot) {
  // Two identical first-turn requests. Same content, same registration hash
  // — but neither has a prior turn, so both bypass prefix-cache routing and
  // each hits the Layer-2 ALLOCATE path. Verify they're independent: two
  // distinct ALLOCATE requests, two distinct mocked slots, both flowing
  // back to the Sequences pushed onto the task queue.
  server_->setMemoryAutoRespond(false);

  auto future1 =
      asyncRequest(ChatRequest().user("hello").maxTokens(1).stream());
  auto future2 =
      asyncRequest(ChatRequest().user("hello").maxTokens(1).stream());

  // Drain both ALLOCATEs before responding to either, so the test can prove
  // they ran concurrently rather than serialised behind one another.
  tt::domain::ManageMemoryTask allocReq1{}, allocReq2{};
  server_->memoryRequestQueue().receive(allocReq1);
  server_->memoryRequestQueue().receive(allocReq2);
  EXPECT_EQ(allocReq1.action, tt::domain::MemoryManagementAction::ALLOCATE);
  EXPECT_EQ(allocReq2.action, tt::domain::MemoryManagementAction::ALLOCATE);
  EXPECT_NE(allocReq1.taskId, allocReq2.taskId)
      << "two independent allocations should have distinct memory taskIds";

  // Hand out distinct slots for the two sessions.
  auto pushSuccess = [&](uint32_t allocTaskId, uint32_t slotId) {
    tt::domain::ManageMemoryResult res{};
    res.taskId = allocTaskId;
    res.status = tt::domain::ManageMemoryStatus::SUCCESS;
    res.slotId = slotId;
    server_->memoryResultQueue().push(res);
  };
  pushSuccess(allocReq1.taskId, 7);
  pushSuccess(allocReq2.taskId, 11);

  // Both sessions now allocate; controller pushes both Sequences.
  auto seq1 = server_->taskQueue().receive();
  auto seq2 = server_->taskQueue().receive();
  ASSERT_NE(seq1, nullptr);
  ASSERT_NE(seq2, nullptr);

  // Mocked slot IDs propagated through SessionManager into the Sequences.
  // Order of receive() vs the order of pushSuccess() isn't guaranteed, so
  // assert on the unordered pair.
  const uint32_t s1 = seq1->getKVCacheSlot();
  const uint32_t s2 = seq2->getKVCacheSlot();
  EXPECT_NE(s1, s2);
  EXPECT_TRUE((s1 == 7u && s2 == 11u) || (s1 == 11u && s2 == 7u))
      << "expected slots {7, 11}, got {" << s1 << ", " << s2 << "}";

  mockWorkerResponse(seq1->taskId);
  mockWorkerResponse(seq2->taskId);
  future1.get();
  future2.get();

  server_->setMemoryAutoRespond(true);
}

TEST_F(MainIntegrationTest, FirstRequestWithHistory_IsNotAContinuation) {
  // A first request that arrives with a multi-turn history baked into
  // `messages` — for example a client replaying a saved conversation
  // against a fresh server. hasPriorTurn=true triggers a prefix-cache
  // lookup on the prior-turn prefix, but the lookup hash isn't registered
  // yet (this is the first request mentioning this conversation), so the
  // controller must fall through to ALLOCATE a new session — not flag
  // the Sequence as a continuation.
  //
  // The first user message is unique to this test on purpose: the prefix
  // hash is computed from messages[0..n-2] (everything except the trailing
  // [assistant, user] pair), and the rest of the suite uses "hello" — we
  // need a string no other test has registered.
  auto future = asyncRequest(ChatRequest()
                                 .user("history-test-unique-first-turn")
                                 .assistant("hi back")
                                 .user("how are you")
                                 .maxTokens(1)
                                 .stream());

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_FALSE(seq->isContinuation())
      << "first request to a fresh server is never a continuation, even "
         "when the request body contains prior assistant turns";

  mockWorkerResponse(seq->taskId);
  future.get();
}

TEST_F(MainIntegrationTest,
       ConcurrentRequests_SameOpenerDifferentRest_DistinctSlots) {
  // Two phases, both using the same opener:
  //
  //   Seed phase: two concurrent single-turn requests. They each ALLOCATE
  //               a session under hash([user: opener]) at a distinct slot,
  //               so afterwards that hash has two candidates available.
  //
  //   Main phase: two concurrent follow-ups with shape
  //               [user: opener, assistant: ..., user: ...] but with
  //               different assistant/user content per request. Both
  //               compute the same prefix-cache lookup hash; each must
  //               acquire a *distinct* seeded session and reuse its slot
  //               (continuation=true).
  //
  // Pins the no-shared-slot guarantee under concurrent load: even when
  // multiple requests compute identical lookup hashes — and even when
  // multiple seeded candidates are available — every request ends up
  // with its own slot. No two requests ever share a slot.
  server_->setMemoryAutoRespond(false);
  const std::string opener = "concurrent-different-rest-opener";
  constexpr uint32_t kSeedSlotA = 7;
  constexpr uint32_t kSeedSlotB = 8;

  auto pushAllocSuccess = [&](uint32_t allocTaskId, uint32_t slotId) {
    tt::domain::ManageMemoryResult res{};
    res.taskId = allocTaskId;
    res.status = tt::domain::ManageMemoryStatus::SUCCESS;
    res.slotId = slotId;
    server_->memoryResultQueue().push(res);
  };

  // --- Seed phase ----------------------------------------------------------
  {
    auto seedF1 =
        asyncRequest(ChatRequest().user(opener).maxTokens(1).stream());
    auto seedF2 =
        asyncRequest(ChatRequest().user(opener).maxTokens(1).stream());

    tt::domain::ManageMemoryTask seedAlloc1{}, seedAlloc2{};
    server_->memoryRequestQueue().receive(seedAlloc1);
    server_->memoryRequestQueue().receive(seedAlloc2);
    EXPECT_EQ(seedAlloc1.action, tt::domain::MemoryManagementAction::ALLOCATE);
    EXPECT_EQ(seedAlloc2.action, tt::domain::MemoryManagementAction::ALLOCATE);
    EXPECT_NE(seedAlloc1.taskId, seedAlloc2.taskId);

    pushAllocSuccess(seedAlloc1.taskId, kSeedSlotA);
    pushAllocSuccess(seedAlloc2.taskId, kSeedSlotB);

    auto seedSeq1 = server_->taskQueue().receive();
    auto seedSeq2 = server_->taskQueue().receive();
    ASSERT_NE(seedSeq1, nullptr);
    ASSERT_NE(seedSeq2, nullptr);
    EXPECT_FALSE(seedSeq1->isContinuation());
    EXPECT_FALSE(seedSeq2->isContinuation());

    mockWorkerResponse(seedSeq1->taskId);
    mockWorkerResponse(seedSeq2->taskId);
    seedF1.get();
    seedF2.get();
  }

  // --- Main phase ----------------------------------------------------------
  auto future1 = asyncRequest(ChatRequest()
                                  .user(opener)
                                  .assistant("thread A's reply")
                                  .user("thread A's followup")
                                  .maxTokens(1)
                                  .stream());
  auto future2 = asyncRequest(ChatRequest()
                                  .user(opener)
                                  .assistant("thread B's reply")
                                  .user("thread B's followup")
                                  .maxTokens(1)
                                  .stream());

  auto seq1 = server_->taskQueue().receive();
  auto seq2 = server_->taskQueue().receive();
  ASSERT_NE(seq1, nullptr);
  ASSERT_NE(seq2, nullptr);
  EXPECT_TRUE(seq1->isContinuation())
      << "follow-up should HIT one of the seeded sessions";
  EXPECT_TRUE(seq2->isContinuation())
      << "follow-up should HIT one of the seeded sessions";

  // Each follow-up reuses a distinct seeded slot — no two follow-ups share.
  const uint32_t s1 = seq1->getKVCacheSlot();
  const uint32_t s2 = seq2->getKVCacheSlot();
  EXPECT_NE(s1, s2);
  EXPECT_TRUE((s1 == kSeedSlotA && s2 == kSeedSlotB) ||
              (s1 == kSeedSlotB && s2 == kSeedSlotA))
      << "expected the seeded slots {" << kSeedSlotA << ", " << kSeedSlotB
      << "}, got {" << s1 << ", " << s2 << "}";

  mockWorkerResponse(seq1->taskId);
  mockWorkerResponse(seq2->taskId);
  future1.get();
  future2.get();

  server_->setMemoryAutoRespond(true);
}

TEST_F(MainIntegrationTest, SystemMessage_DoesNotTriggerContinuation) {
  // A system + user message is a first turn even though there are two messages.
  auto future = asyncRequest(ChatRequest()
                                 .system("you are helpful")
                                 .user("hello")
                                 .maxTokens(1)
                                 .stream());

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
