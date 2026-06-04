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
#include "ipc/interface/result_queue.hpp"
#include "support/chat_completion_stream.hpp"
#include "support/chat_request.hpp"
#include "support/http_client.hpp"
#include "support/http_response.hpp"
#include "support/multiturn_prefix_cache.hpp"
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
  setenv("KV_CACHE_FIRST_BLOCK_SIZE", "32", 1);
  setenv("KV_CACHE_BLOCK_SIZE", "32", 1);
  // Disable the prefix-cache hit threshold by default: most tests assert that
  // ANY prefix match reuses the session (the pre-threshold contract). With the
  // production default (80%), a legitimate follow-up can be rejected when the
  // seed session has grown past the matched prefix — e.g. it gets re-registered
  // with the generated tokens, so a 1-block opener match becomes <80% of a
  // 2-block session and falls through to ALLOCATE, hanging tests that have the
  // memory auto-responder turned off. PrefixCacheHitThreshold_* opts back into
  // 80% explicitly to exercise the rejection path.
  setenv("PREFIX_CACHE_HIT_THRESHOLD", "0", 1);
}

}  // namespace

class MainIntegrationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    tt::utils::ZeroOverheadLogger::initialize();
    server = tt::test::TestServer::start();
  }
  static void TearDownTestSuite() { server.reset(); }

  // Fire request in background. Returns future for the raw HTTP response.
  static std::future<std::string> asyncRequest(const std::string& body) {
    return std::async(std::launch::async, [body] {
      return tt::test::sendAndReceive(server->host(), server->port(), body);
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
        server->resultQueue());
  }

  static std::unique_ptr<tt::test::TestServer> server;
};

std::unique_ptr<tt::test::TestServer> MainIntegrationTest::server;

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
  server->setMemoryAutoRespond(false);

  // 1. Fire the streaming request. The opener must be long enough to form at
  //    least one block (32 tokens with test config) so the follow-up can hit
  //    the prefix cache.
  const std::string opener =
      "hello this is a longer initial message that needs to have enough words "
      "to produce at least thirty two tokens after tokenization so that the "
      "prefix cache can form a block and the follow-up request can match it";
  auto responseFuture =
      asyncRequest(ChatRequest().user(opener).maxTokens(1).stream());

  // 2. Receive and assert on the ALLOCATE.
  tt::domain::ManageMemoryTask memReq{};
  server->memoryRequestQueue().receive(memReq);
  EXPECT_EQ(memReq.action, tt::domain::MemoryManagementAction::ALLOCATE);
  EXPECT_GT(memReq.taskId, 0u);

  // 3. Mock the memory manager.
  tt::domain::ManageMemoryResult memRes{};
  memRes.taskId = memReq.taskId;
  memRes.status = tt::domain::ManageMemoryStatus::SUCCESS;
  memRes.slotId = 0;
  server->memoryResultQueue().push(memRes);

  // 4. Receive and assert on the Sequence.
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_GT(seq->getNumPromptTokens(), 0u);
  EXPECT_FALSE(seq->isContinuation());

  // 5. Mock the worker: one token, then FINAL.
  tt::test::WorkerResponse(seq->taskId)
      .token(42)
      .finalize()
      .sendTo(server->resultQueue());

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

  // 7. Follow-up with the same opener but an assistant turn and a new user
  //    turn appended. The block-based prefix cache matches the first block(s)
  //    from the opener, and only the delta (assistant + new user) is sent.
  //    With block size 32, the matched prefix is trimmed, leaving
  //    (full_prompt - matched_tokens) tokens to prefill.
  const std::string longPriorAssistant =
      "this is the assistant response that was generated for the initial turn";
  auto followUpFuture = asyncRequest(ChatRequest()
                                         .user(opener)
                                         .assistant(longPriorAssistant)
                                         .user("y")
                                         .maxTokens(1)
                                         .stream());
  auto followUpSeq = server->taskQueue().receive();
  ASSERT_NE(followUpSeq, nullptr);
  EXPECT_TRUE(followUpSeq->isContinuation())
      << "follow-up should HIT the seed session";
  // Block-based prefix caching: block(s) from the opener are matched and
  // trimmed. The remaining tokens (full prompt minus matched) are sent to the
  // worker. The key verification is that it's a continuation (cache hit) and
  // fewer tokens than the full prompt are sent.
  EXPECT_TRUE(followUpSeq->getNumPromptTokens() > 0)
      << "continuation should still send some tokens";
  // Verify kv_position_id matches (matched_tokens - 1) for the 0-indexed KV
  // position. With block_size=32, first block matched = 32 tokens, so
  // kv_position_id = 31 (0-indexed position of last matched token).
  ASSERT_TRUE(followUpSeq->getKVPositionId().has_value())
      << "continuation should have kv_position_id set";
  EXPECT_EQ(*followUpSeq->getKVPositionId(), 31u)
      << "kv_position_id should be one less than matched tokens (0-indexed)";
  tt::test::WorkerResponse(followUpSeq->taskId)
      .token(43)
      .finalize()
      .sendTo(server->resultQueue());
  followUpFuture.get();

  EXPECT_TRUE(server->memoryRequestQueue().empty())
      << "follow-up should HIT the cache; no new ALLOCATE expected";

  server->setMemoryAutoRespond(true);
}

TEST_F(MainIntegrationTest, WorkerResponseBuilder_MultipleTokensThenFinalize) {
  // Push 3 specific tokens via WorkerResponse and assert the SSE stream
  // delivered them as separate content deltas, terminated by [DONE].
  auto responseFuture =
      asyncRequest(ChatRequest().user("hello").maxTokens(3).stream());

  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  tt::test::WorkerResponse(seq->taskId)
      .tokens({101, 202, 303})
      .finalize()
      .sendTo(server->resultQueue());

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
  // The first message must be long enough to form at least one block (32
  // tokens with test config) so subsequent turns can hit the prefix cache.
  ChatRequest convo;
  const std::vector<std::string> userMessages = {
      "multi-turn-test-unique-opener with enough words to produce at least "
      "more than we expect to have which is "
      "thirty two tokens after tokenization so that the prefix cache can be "
      "sure that all works fine and that new blocks are being created "
      "register a block for subsequent turns to match against list of the "
      "blocks in the session manager and reuse those blocks ",
      "how are you today? Is everything good? Did you sleep well? Hope you had "
      "some rest. ",
      "tell me a joke that involves elephants, dogs, cats and mouses. Make the "
      "joke about mouse that tricks everyone else",
      "thanks",
  };

  for (size_t i = 0; i < userMessages.size(); ++i) {
    convo.user(userMessages[i]).maxTokens(1).stream();
    auto future = asyncRequest(convo);

    auto seq = server->taskQueue().receive();
    ASSERT_NE(seq, nullptr) << "turn " << i;

    // first turn is 72 tokens
    if (i == 0) {
      EXPECT_EQ(seq->getTokenIds().size(), 72u) << "turn 0 tokenIds size";
    } else if (i == 1) {
      // 2nd turn is 96 tokens -> 63 from cache, 33 new
      EXPECT_EQ(seq->getTokenIds().size(), 33u) << "turn 1 tokenIds size";
      ASSERT_TRUE(seq->getKVPositionId().has_value()) << "turn 1 kvPositionId";
      EXPECT_EQ(*seq->getKVPositionId(), 63u) << "turn 1 kvPositionId value";
    } else if (i == 2) {
      // 3rd turn is 123 tokens -> 95 from cache, 28 new
      EXPECT_EQ(seq->getTokenIds().size(), 28u) << "turn 2 tokenIds size";
      ASSERT_TRUE(seq->getKVPositionId().has_value()) << "turn 2 kvPositionId";
      EXPECT_EQ(*seq->getKVPositionId(), 95u) << "turn 2 kvPositionId value";
    }
    // after first turn it should be a continuation
    EXPECT_EQ(seq->isContinuation(), i > 0) << "turn " << i;

    mockWorkerResponse(seq->taskId);
    future.get();

    convo.assistant("ok");  // simulate the assistant reply for the next turn
  }
}

TEST_F(MainIntegrationTest, MultiTurn_MatchedTokensEqualPriorPromptBlocks) {
  // Walk a 4-turn conversation and verify the EXACT number of prefix-cache
  // matched tokens reported on every continuation.
  //
  // The contract: when turn N+1 arrives, the seed session registered by turn N
  // covers [turn N's full prompt + the token(s) it generated]. Turn N+1's
  // prompt shares turn N's full prompt as a prefix, then diverges — the seed
  // continues with the generated token, the new prompt continues with the
  // fed-back assistant reply. Prefix matching is block-aligned, so the matched
  // prefix is exactly the COMPLETE blocks of turn N's full prompt:
  //
  //     matched_tokens(turn N+1) == block_size * floor(full_prompt(N) /
  //     block_size)
  //
  // We don't hardcode any tokenizer output: each turn's full prompt length is
  // reconstructed from the server's own reported numbers
  //     full_prompt = delta_tokens_sent_to_worker + kv_position_id
  // (the worker is sent the prompt minus the first matched_tokens-1 tokens).
  // In general kv_position_id == matched_tokens-1 + accumulated_think_tokens.
  // Think-filtering IS active in this harness (it defaults to DeepSeek-R1,
  // whose think ids 128798/128799 are live), but this conversation is plain
  // text and contains no <think>/</think> marker tokens, so
  // accumulated_think_tokens is 0 and kv_position_id == matched_tokens-1. The
  // assertions are therefore exact yet independent of the active tokenizer.
  // This is the regression guard for the multiturn matched-token accounting:
  // the pre-fix bug registered corrupt blocks past the matched prefix, so the
  // next turn matched fewer blocks than the prior prompt held.
  constexpr uint32_t kBlock =
      32;  // KV_CACHE_(FIRST_)BLOCK_SIZE in configureEnv

  // Each user turn is long enough to add at least one full 32-token block, so
  // the matched prefix strictly grows turn over turn.
  const std::vector<std::string> userMessages = {
      "opening message for the matched token accounting test with plenty of "
      "words so that this first turn alone tokenizes to well over thirty two "
      "tokens and therefore forms at least one complete prefix cache block",
      "second user turn that again carries more than enough words to push the "
      "running conversation across another full block boundary for the cache",
      "third user turn continuing the conversation with yet more words so the "
      "prefix keeps growing by at least one more complete block this turn too",
      "fourth and final user turn with a comfortable number of additional "
      "words "
      "to guarantee the matched prefix advances by another whole block again",
  };
  // A distinctive multi-word assistant reply fed back as history each turn. Its
  // first token differs from the mock's generated token (42), so the seed and
  // the next prompt diverge exactly at the end of the prior full prompt.
  const std::string assistantReply =
      "acknowledged and here is a sufficiently long assistant reply that is "
      "fed "
      "back into the conversation history on the following turn";

  ChatRequest convo;
  size_t priorFullPrompt = 0;  // full prompt token count of the previous turn
  uint32_t prevMatched = 0;    // matched tokens on the previous continuation

  for (size_t turn = 0; turn < userMessages.size(); ++turn) {
    convo.user(userMessages[turn]).maxTokens(1).stream();
    auto future = asyncRequest(convo);

    auto seq = server->taskQueue().receive();
    ASSERT_NE(seq, nullptr) << "turn " << turn;

    if (turn == 0) {
      // First turn: fresh allocation, nothing to match.
      EXPECT_FALSE(seq->isContinuation()) << "turn 0 must allocate";
      EXPECT_FALSE(seq->getKVPositionId().has_value()) << "turn 0";
      priorFullPrompt = seq->getNumPromptTokens();
      ASSERT_GE(priorFullPrompt, kBlock)
          << "opener must form at least one block";
    } else {
      // Every later turn must HIT the same seed session.
      ASSERT_TRUE(seq->isContinuation())
          << "turn " << turn << " must hit the prefix cache";
      ASSERT_TRUE(seq->getKVPositionId().has_value()) << "turn " << turn;

      // No think-marker tokens in this plain-text conversation, so
      // accumulated_think_tokens == 0 and matched == kv_position_id + 1.
      const uint32_t matched = *seq->getKVPositionId() + 1;
      const uint32_t expected =
          kBlock * static_cast<uint32_t>(priorFullPrompt / kBlock);

      EXPECT_EQ(matched, expected)
          << "turn " << turn << ": matched tokens must equal the complete "
          << "blocks of the prior full prompt (" << priorFullPrompt
          << " tokens)";
      EXPECT_EQ(matched % kBlock, 0u)
          << "turn " << turn << ": matched prefix must be block-aligned";
      EXPECT_GT(matched, prevMatched)
          << "turn " << turn << ": matched prefix must grow as the "
          << "conversation grows (it must not reset or alternate)";

      // Reconstruct this turn's full prompt from the worker-facing delta plus
      // the matched offset, for the next iteration's expectation.
      priorFullPrompt = seq->getNumPromptTokens() + *seq->getKVPositionId();
      prevMatched = matched;
    }

    mockWorkerResponse(seq->taskId);
    future.get();
    convo.assistant(assistantReply);  // history for the next turn
  }
}

TEST_F(MainIntegrationTest,
       MultiTurn_MatchedPrefixGrowsWithThinkMarkersInHistory) {
  // Reasoning-model multiturn: the bug we fixed only bites when the prompt's
  // conversation HISTORY carries <think>/</think> marker tokens (so the raw
  // token positions diverge from the non-think block count the matched prefix
  // is measured in). A real Kimi deployment produces that via its chat template
  // prefilling <think> on every assistant turn; here we reproduce the identical
  // on-the-wire condition with the working DeepSeek tokenizer by feeding back
  // assistant replies that literally contain the <think>…</think> tags — those
  // tag strings tokenize to the reasoning markers (128798/128799), which the
  // prefix-cache hasher must filter. (A Kimi-tokenizer variant can't run in
  // this in-process harness: cpp_server has no Kimi/tiktoken encoder — in
  // production the Dynamo frontend tokenizes Kimi and sends token IDs — so a
  // text-based Kimi request tokenizes to nothing here.)
  //
  // Pre-fix, the matched prefix plateaued (corrupt blocks past the matched
  // prefix); the shared helper asserts it instead advances by a full block
  // every turn.
  const std::vector<std::string> userMessages = {
      "opening reasoning turn for the think-marker multiturn prefix cache test "
      "with plenty of words so this first message tokenizes to well over "
      "thirty "
      "two tokens and forms at least one complete prefix cache block by itself",
      "second user turn that again carries more than enough words to push the "
      "running conversation across another full block boundary for the cache",
      "third user turn continuing the conversation with yet more words so the "
      "matched prefix keeps growing by at least one more complete block again",
      "fourth and final user turn with a comfortable number of additional "
      "words "
      "to guarantee the matched prefix advances by another whole block as well",
  };
  // Assistant reply fed back as history each turn, carrying reasoning markers
  // so the prompt history contains think tokens that must be filtered from
  // blocks.
  const std::string assistantReply =
      "<think> this is the hidden reasoning trace that is wrapped in think "
      "markers and must be excluded from the prefix cache block hashes "
      "</think> "
      "and here is the visible assistant answer fed back into the next turn";

  // Run at the PRODUCTION threshold (80%), not the suite's disabled default.
  // This is what makes the test discriminating: the bug registered corrupt,
  // duplicated blocks past the matched prefix, inflating the session block
  // count so matched/session fell below 80% and the next turn was REJECTED
  // (the alternating HIT/MISS we saw live). The fix keeps the session block
  // count exact, so match% stays ~100% and every turn HITs. At threshold 0 the
  // bug is masked, so 80% is required to guard it.
  setenv("PREFIX_CACHE_HIT_THRESHOLD", "80", 1);
  tt::test::verifyMultiTurnPrefixGrowth(*server, userMessages, assistantReply,
                                        /*blockSize=*/32);
  setenv("PREFIX_CACHE_HIT_THRESHOLD", "0", 1);  // restore suite default
}

TEST_F(MainIntegrationTest, SlotCopy_TriggeredWhenSessionInFlight) {
  // Verifies that a slot copy is triggered when a new request shares a prefix
  // with a session that is currently in-flight (busy generating tokens).
  //
  // Flow:
  //   1. Send request A (long opener) → allocates slot 0, registers session
  //   2. Send request B (continuation of A) → acquires A's session (in-flight)
  //   3. While B is in-flight, send request C with same prefix but different
  //      suffix → prefix cache finds the session but it's busy → ALLOCATE
  //      with slotIdToCopyFrom = slot of request A
  //   4. Mock the allocate response and verify the sequence is a continuation
  setenv("MIN_TOKENS_TO_COPY", "32", 1);
  server->setMemoryAutoRespond(false);

  const std::string opener =
      "slot-copy-test-unique-opener with enough words to produce at least "
      "more than we expect to have which is thirty two tokens after "
      "tokenization so that the prefix cache can form a block and the "
      "follow-up request can match it and reuse the session properly "
      "and here are even more words to extend the shared prefix past "
      "a second block boundary so that the slot copy test exercises "
      "multi-block matching behavior with a longer common prefix region";

  // --- Request A: seed the session ---
  auto futureA = asyncRequest(ChatRequest().user(opener).maxTokens(1).stream());

  tt::domain::ManageMemoryTask memReqA{};
  server->memoryRequestQueue().receive(memReqA);
  EXPECT_EQ(memReqA.action, tt::domain::MemoryManagementAction::ALLOCATE);

  tt::domain::ManageMemoryResult memResA{};
  memResA.taskId = memReqA.taskId;
  memResA.status = tt::domain::ManageMemoryStatus::SUCCESS;
  memResA.slotId = 0;
  server->memoryResultQueue().push(memResA);

  auto seqA = server->taskQueue().receive();
  ASSERT_NE(seqA, nullptr);
  EXPECT_FALSE(seqA->isContinuation());
  const uint32_t slotA = seqA->getKVCacheSlot();

  // Complete request A so the session is registered with its blocks.
  tt::test::WorkerResponse(seqA->taskId)
      .token(42)
      .finalize()
      .sendTo(server->resultQueue());
  futureA.get();

  // --- Request B: continuation that keeps the session in-flight ---
  auto futureB = asyncRequest(ChatRequest()
                                  .user(opener)
                                  .assistant("ok")
                                  .user("thirty two tokens after tokenization "
                                        "so that the prefix cache can be sure "
                                        "that all works fine and that new "
                                        "blocks are being created ")
                                  .maxTokens(1000)
                                  .stream());

  auto seqB = server->taskQueue().receive();
  ASSERT_NE(seqB, nullptr);
  EXPECT_TRUE(seqB->isContinuation()) << "request B should hit the cache";

  // DON'T complete request B yet — keep the session in-flight.

  // --- Request C: same prefix, different suffix → triggers slot copy ---
  auto futureC = asyncRequest(
      ChatRequest()
          .user("slot-copy-test-unique-opener with enough words to produce at "
                "least more than we expect to have which is thirty two tokens "
                "after tokenization so that the prefix cache can form a block "
                "and the follow-up request can match it and reuse the session "
                "properly and here are even more words to extend the shared "
                "prefix past a second block boundary so that the slot copy "
                "test exercises multi-block matching behavior with a longer "
                "common prefix region but this request diverges here with "
                "different content and we need even more words to push this "
                "request past multiple block boundaries so that the follow-up "
                "request D can match more than one block from this session "
                "and confirm reuse works")
          .maxTokens(1)
          .stream());

  // The session is in-flight (B holds it), so C falls through to ALLOCATE.
  // The memory request should have slotIdToCopyFrom pointing to slot A.
  tt::domain::ManageMemoryTask memReqC{};
  server->memoryRequestQueue().receive(memReqC);
  EXPECT_EQ(memReqC.action, tt::domain::MemoryManagementAction::ALLOCATE)
      << "request C should ALLOCATE (session is in-flight)";
  ASSERT_TRUE(memReqC.slotIdToCopyFrom.has_value())
      << "ALLOCATE should request a slot copy from the in-flight session";
  EXPECT_EQ(*memReqC.slotIdToCopyFrom, slotA)
      << "slotIdToCopyFrom should be the slot of request A";

  // Mock the allocate response for C: assign slot 1.
  tt::domain::ManageMemoryResult memResC{};
  memResC.taskId = memReqC.taskId;
  memResC.status = tt::domain::ManageMemoryStatus::SUCCESS;
  memResC.slotId = 1;
  server->memoryResultQueue().push(memResC);

  // The sequence for C should be flagged as a continuation (slot copy sets it).
  auto seqC = server->taskQueue().receive();
  ASSERT_NE(seqC, nullptr);
  EXPECT_TRUE(seqC->isContinuation())
      << "request C should be a continuation (slot copy)";
  EXPECT_EQ(seqC->getTokenIds().size(), 66u) << "request C tokenIds size";
  ASSERT_TRUE(seqC->getKVPositionId().has_value())
      << "slot copy should set kv_position_id";
  EXPECT_EQ(*seqC->getKVPositionId(), 63u) << "request C kvPositionId value";

  // Complete request C.
  tt::test::WorkerResponse(seqC->taskId)
      .token(99)
      .finalize()
      .sendTo(server->resultQueue());
  futureC.get();

  // Now complete request B.
  tt::test::WorkerResponse(seqB->taskId)
      .token(50)
      .finalize()
      .sendTo(server->resultQueue());
  futureB.get();

  // --- Request D: follow-up to C's session → confirms slot 1 is reused ---
  // Both sessions (slot 0 and slot 1) are now free. The session manager must
  // pick slot 1 (C's session) as the better fit because D's prompt shares
  // more blocks with C's registered session than with A/B's.
  auto futureD = asyncRequest(
      ChatRequest()
          .user("slot-copy-test-unique-opener with enough words to produce at "
                "least more than we expect to have which is thirty two tokens "
                "after tokenization so that the prefix cache can form a block "
                "and the follow-up request can match it and reuse the session "
                "properly and here are even more words to extend the shared "
                "prefix past a second block boundary so that the slot copy "
                "test exercises multi-block matching behavior with a longer "
                "common prefix region but this request diverges here with "
                "different content and we need even more words to push this "
                "request past multiple block boundaries so that the follow-up "
                "request D can match more than one block from this session "
                "and confirm reuse works")
          .assistant("ok")
          .user("this is the follow-up to request C")
          .maxTokens(1)
          .stream());

  auto seqD = server->taskQueue().receive();
  ASSERT_NE(seqD, nullptr);
  EXPECT_TRUE(seqD->isContinuation())
      << "request D should hit C's session (slot 1)";
  EXPECT_EQ(seqD->getKVCacheSlot(), 1u)
      << "request D should reuse slot 1 from request C";
  EXPECT_EQ(seqD->getTokenIds().size(), 14u) << "request D tokenIds size";
  ASSERT_TRUE(seqD->getKVPositionId().has_value())
      << "request D should have kv_position_id set";
  EXPECT_EQ(*seqD->getKVPositionId(), 127u) << "request D kvPositionId value";

  tt::test::WorkerResponse(seqD->taskId)
      .token(101)
      .finalize()
      .sendTo(server->resultQueue());
  futureD.get();

  server->setMemoryAutoRespond(true);
  unsetenv("MIN_TOKENS_TO_COPY");
}

TEST_F(MainIntegrationTest, NonStreamingRequest_ReturnsBufferedJson) {
  // Most tests use streaming; this one verifies the non-streaming code path
  // still returns a single buffered JSON document with the assistant message.
  auto responseFuture = asyncRequest(ChatRequest().user("hello").maxTokens(1));

  auto seq = server->taskQueue().receive();
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

  auto seq = server->taskQueue().receive();
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

  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_FALSE(seq->isDisaggregated());

  mockWorkerResponse(seq->taskId);
  future.get();
}

TEST_F(MainIntegrationTest, TwoFirstTurns_EachAllocatesDistinctSlot) {
  // Two identical first-turn requests. Same content, same registration hash
  // — but when all candidate sessions are in-flight, both fall through to
  // the ALLOCATE path. Verify they're independent: two distinct ALLOCATE
  // requests, two distinct mocked slots, both flowing back to the Sequences
  // pushed onto the task queue.
  //
  // Uses a unique opener to avoid acquiring sessions registered by earlier
  // tests in this suite.
  server->setMemoryAutoRespond(false);

  auto future1 = asyncRequest(
      ChatRequest().user("two-first-turns-test").maxTokens(1).stream());
  auto future2 = asyncRequest(
      ChatRequest().user("two-first-turns-test").maxTokens(1).stream());

  // Drain both ALLOCATEs before responding to either, so the test can prove
  // they ran concurrently rather than serialised behind one another.
  tt::domain::ManageMemoryTask allocReq1{}, allocReq2{};
  server->memoryRequestQueue().receive(allocReq1);
  server->memoryRequestQueue().receive(allocReq2);
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
    server->memoryResultQueue().push(res);
  };
  pushSuccess(allocReq1.taskId, 7);
  pushSuccess(allocReq2.taskId, 11);

  // Both sessions now allocate; controller pushes both Sequences.
  auto seq1 = server->taskQueue().receive();
  auto seq2 = server->taskQueue().receive();
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

  server->setMemoryAutoRespond(true);
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

  auto seq = server->taskQueue().receive();
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
  server->setMemoryAutoRespond(false);
  // The opener must be long enough to form at least one block (32 tokens with
  // test config) so the follow-up requests can hit the prefix cache.
  const std::string opener =
      "concurrent-different-rest-opener with enough words to produce at least "
      "thirty two tokens after tokenization so that the prefix cache can "
      "register blocks for the follow-up requests to match against";
  constexpr uint32_t kSeedSlotA = 7;
  constexpr uint32_t kSeedSlotB = 8;

  auto pushAllocSuccess = [&](uint32_t allocTaskId, uint32_t slotId) {
    tt::domain::ManageMemoryResult res{};
    res.taskId = allocTaskId;
    res.status = tt::domain::ManageMemoryStatus::SUCCESS;
    res.slotId = slotId;
    server->memoryResultQueue().push(res);
  };

  // --- Seed phase ----------------------------------------------------------
  {
    auto seedF1 =
        asyncRequest(ChatRequest().user(opener).maxTokens(1).stream());
    auto seedF2 =
        asyncRequest(ChatRequest().user(opener).maxTokens(1).stream());

    tt::domain::ManageMemoryTask seedAlloc1{}, seedAlloc2{};
    server->memoryRequestQueue().receive(seedAlloc1);
    server->memoryRequestQueue().receive(seedAlloc2);
    EXPECT_EQ(seedAlloc1.action, tt::domain::MemoryManagementAction::ALLOCATE);
    EXPECT_EQ(seedAlloc2.action, tt::domain::MemoryManagementAction::ALLOCATE);
    EXPECT_NE(seedAlloc1.taskId, seedAlloc2.taskId);

    pushAllocSuccess(seedAlloc1.taskId, kSeedSlotA);
    pushAllocSuccess(seedAlloc2.taskId, kSeedSlotB);

    auto seedSeq1 = server->taskQueue().receive();
    auto seedSeq2 = server->taskQueue().receive();
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

  auto seq1 = server->taskQueue().receive();
  auto seq2 = server->taskQueue().receive();
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

  server->setMemoryAutoRespond(true);
}

TEST_F(MainIntegrationTest, SystemMessage_DoesNotTriggerContinuation) {
  // A system + user message is a first turn even though there are two messages.
  auto future = asyncRequest(ChatRequest()
                                 .system("you are helpful")
                                 .user("hello")
                                 .maxTokens(1)
                                 .stream());

  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_FALSE(seq->isContinuation());

  mockWorkerResponse(seq->taskId);
  future.get();
}

TEST_F(MainIntegrationTest, PrefixCacheHitThreshold_RejectsLowMatchPercentage) {
  // Build a session with many blocks (long conversation), then send a short
  // request that only matches the first block. With PREFIX_CACHE_HIT_THRESHOLD
  // at 80% (default), the session should be rejected and a new one allocated.
  //
  // Setup: opener (1 block) -> many assistant/user turns (many more blocks)
  // Test request: same opener only -> matches 1 block out of many -> rejected
  //
  // Steps 1-2 BUILD the long session and must run with the threshold disabled
  // (the suite default, see configureEnv): each growth turn only matches the
  // leading block(s) of the session-so-far, which is below 80%, so an active
  // threshold would reject these legitimate continuations and the session
  // would never grow. The threshold is enabled only for step 3 — the actual
  // rejection assertion — and restored to "0" at the end.
  server->setMemoryAutoRespond(false);

  // Step 1: Create a long conversation to build up many blocks.
  // The opener must be long enough to form at least one block (32 tokens).
  const std::string opener =
      "prefix-threshold-test-unique-opener-v2 with enough tokens to form "
      "exactly one block when tokenized this needs at least thirty two tokens "
      "after tokenization so we add more words here";

  // First turn: establishes the session with opener
  auto future1 = asyncRequest(ChatRequest().user(opener).maxTokens(1).stream());

  tt::domain::ManageMemoryTask memReq1{};
  server->memoryRequestQueue().receive(memReq1);
  EXPECT_EQ(memReq1.action, tt::domain::MemoryManagementAction::ALLOCATE);

  tt::domain::ManageMemoryResult memRes1{};
  memRes1.taskId = memReq1.taskId;
  memRes1.status = tt::domain::ManageMemoryStatus::SUCCESS;
  memRes1.slotId = 0;
  server->memoryResultQueue().push(memRes1);

  auto seq1 = server->taskQueue().receive();
  ASSERT_NE(seq1, nullptr);
  EXPECT_FALSE(seq1->isContinuation())
      << "First turn should not be continuation";

  tt::test::WorkerResponse(seq1->taskId)
      .token(100)
      .finalize()
      .sendTo(server->resultQueue());
  future1.get();

  // Step 2: Add several more turns to grow the session's block count.
  // Each turn adds more blocks, making the session much longer than just
  // opener.
  ChatRequest convo;
  convo.user(opener).assistant("ok");

  for (int i = 0; i < 5; ++i) {
    std::string longMessage =
        "additional turn " + std::to_string(i) +
        " with enough words to add more blocks to this session making it "
        "much longer than the original opener so that a short request "
        "matching only the opener will have a low match percentage";
    convo.user(longMessage).maxTokens(1).stream();

    auto future = asyncRequest(convo);
    auto seq = server->taskQueue().receive();
    ASSERT_NE(seq, nullptr);
    EXPECT_TRUE(seq->isContinuation())
        << "Turn " << i << " should be continuation";

    tt::test::WorkerResponse(seq->taskId)
        .token(101 + i)
        .finalize()
        .sendTo(server->resultQueue());
    future.get();

    convo.assistant("got it");
  }

  // Step 3: Enable the production 80% threshold, then send a NEW request with
  // only the opener (no history). This matches only the first block of the
  // multi-block session, so the ~10-20% match should be rejected.
  setenv("PREFIX_CACHE_HIT_THRESHOLD", "80", 1);
  auto future2 = asyncRequest(ChatRequest().user(opener).maxTokens(1).stream());

  // Should allocate a NEW session because match percentage is below threshold.
  // An ALLOCATE request indicates the prefix cache candidate was rejected.
  tt::domain::ManageMemoryTask memReq2{};
  server->memoryRequestQueue().receive(memReq2);
  EXPECT_EQ(memReq2.action, tt::domain::MemoryManagementAction::ALLOCATE)
      << "Low match percentage should trigger new ALLOCATE, not reuse session";

  tt::domain::ManageMemoryResult memRes2{};
  memRes2.taskId = memReq2.taskId;
  memRes2.status = tt::domain::ManageMemoryStatus::SUCCESS;
  memRes2.slotId = 1;  // Different slot
  server->memoryResultQueue().push(memRes2);

  auto seq2 = server->taskQueue().receive();
  ASSERT_NE(seq2, nullptr);
  EXPECT_FALSE(seq2->isContinuation())
      << "Short request should NOT be continuation due to threshold rejection";

  tt::test::WorkerResponse(seq2->taskId)
      .token(200)
      .finalize()
      .sendTo(server->resultQueue());
  future2.get();

  server->setMemoryAutoRespond(true);
  setenv("PREFIX_CACHE_HIT_THRESHOLD", "0", 1);
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
