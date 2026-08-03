// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/session_manager.hpp"

#include <gtest/gtest.h>
#include <trantor/net/EventLoop.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "../integration_test_helpers.hpp"
#include "domain/session.hpp"
#include "utils/conversation_hasher.hpp"

namespace {

using tt::test::acquireInFlight;
using tt::test::bootstrapSessionWithResponseId;
using tt::test::callGetSlot;
using tt::test::createTestSession;
using tt::test::makeSequentialPrompt;
using tt::test::releaseSlot;
using tt::test::runConcurrently;
using tt::test::TrantorLoopFixture;

// Aliases for backward compatibility with existing tests.
using LoopFixture = TrantorLoopFixture;

inline std::string createSessionWithSlot(tt::services::SessionManager& manager,
                                         trantor::EventLoop* loop,
                                         uint32_t slotId) {
  return createTestSession(manager, loop, slotId);
}

inline std::string createSessionWithSlot(
    tt::services::SessionManager& manager, trantor::EventLoop* loop,
    uint32_t slotId, const std::vector<tt::utils::BlockHashInfo>& blockInfos) {
  return createTestSession(manager, loop, slotId, blockInfos);
}

// ---------------------------------------------------------------------------
// tryMarkInFlight
// ---------------------------------------------------------------------------

TEST(SessionManager, TryMarkInFlight_MarksSessionAndReturnsSlot) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 11u);
  ASSERT_FALSE(sessionId.empty());

  std::function<void()> cancelFn;
  auto result = manager.tryMarkInFlight(sessionId, cancelFn);
  EXPECT_EQ(result.outcome, tt::domain::MarkInFlightOutcome::Marked);
  EXPECT_EQ(result.slotId, 11u);

  manager.getSession(sessionId)->release();
}

TEST(SessionManager, TryMarkInFlight_BusyWhenAlreadyInFlight) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 12u);
  ASSERT_FALSE(sessionId.empty());

  std::function<void()> cancelFn;
  ASSERT_EQ(manager.tryMarkInFlight(sessionId, cancelFn).outcome,
            tt::domain::MarkInFlightOutcome::Marked);

  auto busy = manager.tryMarkInFlight(sessionId, cancelFn);
  EXPECT_EQ(busy.outcome, tt::domain::MarkInFlightOutcome::Busy);

  manager.getSession(sessionId)->release();
}

TEST(SessionManager, TryMarkInFlight_StaleWhenKeyHashMismatch) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 13u);
  ASSERT_FALSE(sessionId.empty());

  std::function<void()> cancelFn;
  auto result = manager.tryMarkInFlight(sessionId, cancelFn, 999u, nullptr);
  EXPECT_EQ(result.outcome, tt::domain::MarkInFlightOutcome::Stale);
}

TEST(SessionManager, GetAndSetSessionHash) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 14u);
  ASSERT_FALSE(sessionId.empty());

  EXPECT_TRUE(manager.getSessionHash(sessionId).has_value());
  EXPECT_TRUE(manager.setSessionHash(sessionId, 42u));
  EXPECT_EQ(manager.getSessionHash(sessionId), 42u);
}

TEST(SessionManager, SetSessionResponseId) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 15u);
  ASSERT_FALSE(sessionId.empty());

  ASSERT_TRUE(manager.setSessionResponseId(sessionId, "resp-lease"));
  auto session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  EXPECT_EQ(session->getResponseId(), "resp-lease");
}

// ---------------------------------------------------------------------------
// SessionManager lifecycle tests
// ---------------------------------------------------------------------------

TEST(SessionManagerLifecycle, CloseIdleSession_ReturnsSuccess) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 10u);
  ASSERT_FALSE(sessionId.empty());

  EXPECT_EQ(manager.closeSession(sessionId),
            tt::services::CloseSessionResult::SUCCESS);
  EXPECT_FALSE(manager.getSession(sessionId));
}

TEST(SessionManagerLifecycle, CloseNonExistentSession_ReturnsNotFound) {
  tt::services::SessionManager manager;

  EXPECT_EQ(manager.closeSession("no-such-id"),
            tt::services::CloseSessionResult::NOT_FOUND);
}

TEST(SessionManagerLifecycle, AcquireInFlight_ReturnsPreAssignedSlotId) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 7u);
  ASSERT_FALSE(sessionId.empty());

  EXPECT_EQ(acquireInFlight(manager, sessionId), 7u);
  auto session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  session->release();
}

TEST(SessionManagerLifecycle, AcquireInFlight_AlreadyInFlight_Throws) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 8u);
  ASSERT_FALSE(sessionId.empty());

  acquireInFlight(manager, sessionId);
  EXPECT_THROW(acquireInFlight(manager, sessionId),
               tt::services::SessionInFlightException);
  auto session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  session->release();
}

TEST(SessionManagerLifecycle, CloseWhileInFlight_RemovesSessionImmediately) {
  // closeSession must remove the session and trigger dealloc immediately,
  // even when the session is in-flight. Session clearInFlight called afterwards
  // would fail since session is already gone (we don't test that here since
  // we can't get the session pointer after it's closed).
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 9u);
  ASSERT_FALSE(sessionId.empty());

  acquireInFlight(manager, sessionId);
  manager.closeSession(sessionId);
  EXPECT_FALSE(manager.getSession(sessionId));
}

TEST(SessionManagerLifecycle, GetActiveSessionCount_ReflectsLifecycle) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  EXPECT_EQ(manager.getActiveSessionCount(), 0u);

  auto s1 = createSessionWithSlot(manager, lf.loop, 20u);
  EXPECT_EQ(manager.getActiveSessionCount(), 1u);

  auto s2 = createSessionWithSlot(manager, lf.loop, 21u);
  EXPECT_EQ(manager.getActiveSessionCount(), 2u);

  manager.closeSession(s1);
  EXPECT_EQ(manager.getActiveSessionCount(), 1u);

  manager.closeSession(s2);
  EXPECT_EQ(manager.getActiveSessionCount(), 0u);
}

TEST(SessionManagerLifecycle, AcquireAfterRelease_Succeeds) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 11u);
  ASSERT_FALSE(sessionId.empty());

  acquireInFlight(manager, sessionId);
  auto session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  session->release();

  EXPECT_NO_THROW(acquireInFlight(manager, sessionId));
  session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  session->release();
}

TEST(SessionManagerLifecycle, GetSession_ReturnsCorrectData) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 12u);
  ASSERT_FALSE(sessionId.empty());

  auto session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  EXPECT_EQ(session->getSessionId(), sessionId);
  EXPECT_EQ(session->getSlotId(), 12u);
}

TEST(SessionManagerLifecycle, AssignSlotId_UpdatesSession) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 30u);
  ASSERT_FALSE(sessionId.empty());

  EXPECT_TRUE(manager.assignSlotId(sessionId, 99u));

  auto session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  EXPECT_EQ(session->getSlotId(), 99u);
}

TEST(SessionManagerLifecycle, GetSlotIdBySessionId_ReturnsSlotId) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 13u);
  ASSERT_FALSE(sessionId.empty());

  EXPECT_EQ(manager.getSlotIdBySessionId(sessionId), 13u);
}

TEST(SessionManagerLifecycle,
     GetSlotIdBySessionId_NotFound_ReturnsInvalidSlot) {
  tt::services::SessionManager manager;

  EXPECT_EQ(manager.getSlotIdBySessionId("no-such-id"),
            tt::domain::INVALID_SLOT_ID);
}

// ---------------------------------------------------------------------------
// SessionManager close-while-in-flight tests
// ---------------------------------------------------------------------------

TEST(SessionManagerClose, CloseInFlight_RemovesSessionImmediately) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 42u);
  ASSERT_FALSE(sessionId.empty());

  acquireInFlight(manager, sessionId);

  EXPECT_EQ(manager.closeSession(sessionId),
            tt::services::CloseSessionResult::SUCCESS);
  EXPECT_FALSE(manager.getSession(sessionId));
}

TEST(SessionManagerClose, CloseInFlight_FiresCancelFn_AtomicWithAcquire) {
  // Cancel and in-flight state are set atomically by acquireInFlight.
  // closeSession must fire the cancel function immediately.
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 45u);
  ASSERT_FALSE(sessionId.empty());

  std::atomic<bool> cancelCalled{false};
  manager.acquireInFlight(sessionId,
                          [&cancelCalled]() { cancelCalled = true; });

  manager.closeSession(sessionId);

  EXPECT_TRUE(cancelCalled.load());
  EXPECT_FALSE(manager.getSession(sessionId));
}

TEST(SessionManagerClose, CloseIdle_NoCancelFired) {
  // Idle sessions have no in-flight request; closeSession must not fire any
  // cancel (there is none registered).
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 46u);
  ASSERT_FALSE(sessionId.empty());

  // Close without ever calling acquireInFlight — no cancel should be needed.
  EXPECT_EQ(manager.closeSession(sessionId),
            tt::services::CloseSessionResult::SUCCESS);
  EXPECT_FALSE(manager.getSession(sessionId));
}

TEST(SessionManagerClose, ReleaseInFlight_AfterClose_IsNoOp) {
  // Simulates the SSE writer attempting to clear in-flight after the session
  // was already removed by a concurrent closeSession. Since the session is
  // gone, we can't call clearInFlight on it (it would be a dangling pointer).
  // This test now just verifies that closing an in-flight session removes it.
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 43u);
  ASSERT_FALSE(sessionId.empty());

  acquireInFlight(manager, sessionId);
  manager.closeSession(sessionId);

  // Session is gone - we can't call clearInFlight on it
  EXPECT_FALSE(manager.getSession(sessionId));
  EXPECT_EQ(manager.getActiveSessionCount(), 0u);
}

TEST(SessionManagerClose, CancelFn_ClearedOnNormalCompletion) {
  // If the request completes normally, session clearInFlight must clear the
  // in-flight state so a subsequent close does not fire stale cancel logic.
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 44u);
  ASSERT_FALSE(sessionId.empty());

  std::atomic<bool> cancelCalled{false};
  manager.acquireInFlight(sessionId,
                          [&cancelCalled]() { cancelCalled = true; });

  auto session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  session->release();               // normal completion clears in-flight state
  manager.closeSession(sessionId);  // should not fire cancel

  EXPECT_FALSE(cancelCalled.load());
}

TEST(SessionManagerClose, ReleaseInFlight_NormalCompletion_SessionStaysIdle) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 47u);
  ASSERT_FALSE(sessionId.empty());

  acquireInFlight(manager, sessionId);
  auto session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  session->release();

  // Session still present and acquirable again.
  EXPECT_TRUE(manager.getSession(sessionId));
  EXPECT_NO_THROW(acquireInFlight(manager, sessionId));
  session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  session->release();
}

// ---------------------------------------------------------------------------
// Concurrency tests
// ---------------------------------------------------------------------------

TEST(SessionManagerConcurrency, ConcurrentClose_OnlyOneSucceeds) {
  // Two threads race to close the same session. Exactly one must get SUCCESS;
  // the other must get NOT_FOUND. The session must be gone afterwards.
  constexpr int iterations = 200;
  for (int i = 0; i < iterations; ++i) {
    tt::services::SessionManager manager;
    LoopFixture lf;
    auto sessionId = createSessionWithSlot(manager, lf.loop, 100u);

    std::atomic<int> successCount{0};
    runConcurrently([&] {
      auto result = manager.closeSession(sessionId);
      if (result == tt::services::CloseSessionResult::SUCCESS) {
        successCount.fetch_add(1, std::memory_order_relaxed);
      }
    });

    EXPECT_EQ(successCount.load(), 1) << "iteration " << i;
    EXPECT_EQ(manager.getActiveSessionCount(), 0u) << "iteration " << i;
  }
}

TEST(SessionManagerConcurrency, ConcurrentAcquire_OnlyOneSucceeds) {
  // Two threads race to acquireInFlight the same session. Exactly one must
  // succeed; the other must throw SessionInFlightException.
  constexpr int iterations = 200;
  for (int i = 0; i < iterations; ++i) {
    tt::services::SessionManager manager;
    LoopFixture lf;
    auto sessionId = createSessionWithSlot(manager, lf.loop, 101u);

    std::atomic<int> acquireCount{0};
    runConcurrently([&] {
      try {
        manager.acquireInFlight(sessionId, nullptr);
        acquireCount.fetch_add(1, std::memory_order_relaxed);
      } catch (const tt::services::SessionInFlightException&) {
      }
    });

    EXPECT_EQ(acquireCount.load(), 1) << "iteration " << i;
    auto session = manager.getSession(sessionId);
    if (session) {
      session->release();
    }
  }
}

TEST(SessionManagerConcurrency,
     ConcurrentAcquireAndClose_CancelFiredAtMostOnce) {
  // One thread acquires in-flight while another closes. The cancel function
  // must fire at most once regardless of which wins the race. The session
  // must be absent and the count zero after both threads finish.
  constexpr int iterations = 200;
  for (int i = 0; i < iterations; ++i) {
    tt::services::SessionManager manager;
    LoopFixture lf;
    auto sessionId = createSessionWithSlot(manager, lf.loop, 102u);

    std::atomic<int> cancelCount{0};
    std::atomic<bool> ready{false};

    std::thread acquirer([&] {
      while (!ready.load(std::memory_order_acquire)) {
      }
      try {
        manager.acquireInFlight(sessionId,
                                [&cancelCount] { cancelCount.fetch_add(1); });
        auto session = manager.getSession(sessionId);
        if (session) {
          session->release();
        }
      } catch (const tt::services::SessionRateLimitException&) {
      }
    });

    std::thread closer([&] {
      while (!ready.load(std::memory_order_acquire)) {
      }
      manager.closeSession(sessionId);
    });

    ready.store(true, std::memory_order_release);
    acquirer.join();
    closer.join();

    EXPECT_LE(cancelCount.load(), 1) << "iteration " << i;
    EXPECT_EQ(manager.getActiveSessionCount(), 0u) << "iteration " << i;
  }
}

// ---------------------------------------------------------------------------
// getSlot() routing tests
//
// These cover the unified slot acquisition path used by LLMPipeline:
// response-id continuation, prefix-cache fallback, and in-flight / cancel
// semantics. Turn-1 sessions are bootstrapped with createTestSession +
// registerResponseId (IPC allocation is not available in these unit tests).
// ---------------------------------------------------------------------------

namespace {

// Three full prefix blocks with default KV cache block sizes (128 + 32 + 32).
std::vector<uint32_t> makeThreeBlockPrompt() {
  return makeSequentialPrompt(128 + 32 + 32);
}

std::vector<uint32_t> makeFourBlockPrompt() {
  auto prompt = makeThreeBlockPrompt();
  auto tail = makeSequentialPrompt(32, prompt.size());
  prompt.insert(prompt.end(), tail.begin(), tail.end());
  return prompt;
}

}  // namespace

TEST(SessionManagerGetSlot, ResponseIdHit_ReturnsSessionAndSlot) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId =
      bootstrapSessionWithResponseId(manager, lf.loop, 50u, "resp-1");
  ASSERT_FALSE(sessionId.empty());

  auto prompt = makeThreeBlockPrompt();
  tt::services::GetSlotOptions opts;
  opts.previousResponseId = "resp-1";

  auto outcome = callGetSlot(manager, lf.loop, prompt, opts);
  ASSERT_TRUE(outcome.result.has_value());
  EXPECT_EQ(outcome.result->sessionId, sessionId);
  EXPECT_EQ(outcome.result->slotId, 50u);
  EXPECT_FALSE(outcome.result->isNewSession);

  releaseSlot(manager, sessionId);
}

TEST(SessionManagerGetSlot,
     UnknownPreviousResponseId_FallsThroughToPrefixCache) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto prompt = makeThreeBlockPrompt();
  auto blocks = manager.computeBlockInfos(prompt);
  auto sessionId = createTestSession(manager, lf.loop, 51u, blocks);
  ASSERT_FALSE(sessionId.empty());

  tt::services::GetSlotOptions opts;
  opts.previousResponseId = "no-such-id";

  auto outcome = callGetSlot(manager, lf.loop, prompt, opts);
  ASSERT_TRUE(outcome.result.has_value());
  EXPECT_EQ(outcome.result->sessionId, sessionId);
  EXPECT_FALSE(outcome.result->isNewSession);

  releaseSlot(manager, sessionId);
}

TEST(SessionManagerGetSlot, NoPreviousResponseId_UsesPrefixCache) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto prompt = makeThreeBlockPrompt();
  auto blocks = manager.computeBlockInfos(prompt);
  auto sessionId = createTestSession(manager, lf.loop, 52u, blocks);
  ASSERT_FALSE(sessionId.empty());

  auto outcome = callGetSlot(manager, lf.loop, prompt, {});
  ASSERT_TRUE(outcome.result.has_value());
  EXPECT_EQ(outcome.result->sessionId, sessionId);
  EXPECT_FALSE(outcome.result->isNewSession);

  releaseSlot(manager, sessionId);
}

TEST(SessionManagerGetSlot, ReKey_MovesSessionToNewId) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId =
      bootstrapSessionWithResponseId(manager, lf.loop, 53u, "resp-1");
  ASSERT_FALSE(sessionId.empty());

  auto prompt = makeThreeBlockPrompt();
  tt::services::GetSlotOptions opts;
  opts.previousResponseId = "resp-1";
  opts.responseId = "resp-2";

  auto outcome = callGetSlot(manager, lf.loop, prompt, opts);
  ASSERT_TRUE(outcome.result.has_value());
  EXPECT_EQ(outcome.result->sessionId, sessionId);
  releaseSlot(manager, sessionId);

  // resp-1 is no longer in the response-id index; getSlot falls through to
  // prefix-cache and still finds the same session.
  tt::services::GetSlotOptions stale;
  stale.previousResponseId = "resp-1";
  auto staleOutcome = callGetSlot(manager, lf.loop, prompt, stale);
  ASSERT_TRUE(staleOutcome.result.has_value());
  EXPECT_EQ(staleOutcome.result->sessionId, sessionId);
  releaseSlot(manager, sessionId);

  tt::services::GetSlotOptions fresh;
  fresh.previousResponseId = "resp-2";
  auto freshOutcome = callGetSlot(manager, lf.loop, prompt, fresh);
  ASSERT_TRUE(freshOutcome.result.has_value());
  EXPECT_EQ(freshOutcome.result->sessionId, sessionId);

  releaseSlot(manager, sessionId);
}

TEST(SessionManagerGetSlot, SecondAcquireWhileInFlight_RateLimited) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId =
      bootstrapSessionWithResponseId(manager, lf.loop, 54u, "resp-1");
  ASSERT_FALSE(sessionId.empty());

  auto prompt = makeThreeBlockPrompt();
  tt::services::GetSlotOptions opts;
  opts.previousResponseId = "resp-1";

  auto first = callGetSlot(manager, lf.loop, prompt, opts);
  ASSERT_TRUE(first.result.has_value());

  auto second = callGetSlot(manager, lf.loop, prompt, opts);
  EXPECT_TRUE(second.rateLimited);

  releaseSlot(manager, sessionId);
}

TEST(SessionManagerGetSlot, AcquireAfterRelease_Succeeds) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId =
      bootstrapSessionWithResponseId(manager, lf.loop, 55u, "resp-1");
  ASSERT_FALSE(sessionId.empty());

  auto prompt = makeThreeBlockPrompt();
  tt::services::GetSlotOptions opts;
  opts.previousResponseId = "resp-1";

  auto first = callGetSlot(manager, lf.loop, prompt, opts);
  ASSERT_TRUE(first.result.has_value());
  releaseSlot(manager, sessionId);

  auto second = callGetSlot(manager, lf.loop, prompt, opts);
  ASSERT_TRUE(second.result.has_value());
  EXPECT_EQ(second.result->sessionId, sessionId);
  releaseSlot(manager, sessionId);
}

TEST(SessionManagerGetSlot, CloseSession_RemovesFromResponseIdIndex) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto prompt = makeThreeBlockPrompt();
  auto blocks = manager.computeBlockInfos(prompt);
  auto sessionId =
      bootstrapSessionWithResponseId(manager, lf.loop, 56u, "resp-1", blocks);
  ASSERT_FALSE(sessionId.empty());

  ASSERT_EQ(manager.closeSession(sessionId),
            tt::services::CloseSessionResult::SUCCESS);

  // A live session with the same prefix should be acquired via prefix-cache,
  // not the closed session via the stale response-id entry.
  auto replacementId = createTestSession(manager, lf.loop, 561u, blocks);
  tt::services::GetSlotOptions opts;
  opts.previousResponseId = "resp-1";
  auto outcome = callGetSlot(manager, lf.loop, prompt, opts);
  ASSERT_TRUE(outcome.result.has_value());
  EXPECT_EQ(outcome.result->sessionId, replacementId);
  releaseSlot(manager, replacementId);
}

TEST(SessionManagerGetSlot, CloseWhileAcquired_FiresCancelFn) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId =
      bootstrapSessionWithResponseId(manager, lf.loop, 57u, "resp-1");
  ASSERT_FALSE(sessionId.empty());

  auto prompt = makeThreeBlockPrompt();
  std::atomic<bool> cancelCalled{false};
  tt::services::GetSlotOptions opts;
  opts.previousResponseId = "resp-1";
  opts.cancelFn = [&cancelCalled]() { cancelCalled = true; };

  auto outcome = callGetSlot(manager, lf.loop, prompt, opts);
  ASSERT_TRUE(outcome.result.has_value());

  manager.closeSession(sessionId);
  EXPECT_TRUE(cancelCalled.load());
  EXPECT_FALSE(manager.getSession(sessionId));
}

TEST(SessionManagerGetSlot, TwoTurnContinuation_ReKeysAcrossIds) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = bootstrapSessionWithResponseId(manager, lf.loop, 58u, "r1");
  ASSERT_FALSE(sessionId.empty());

  auto turn1Prompt = makeThreeBlockPrompt();

  tt::services::GetSlotOptions turn2;
  turn2.previousResponseId = "r1";
  turn2.responseId = "r2";
  auto t2 = callGetSlot(manager, lf.loop, turn1Prompt, turn2);
  ASSERT_TRUE(t2.result.has_value());
  EXPECT_EQ(t2.result->sessionId, sessionId);
  releaseSlot(manager, sessionId);

  tt::services::GetSlotOptions turn3;
  turn3.previousResponseId = "r2";
  auto t3 = callGetSlot(manager, lf.loop, turn1Prompt, turn3);
  ASSERT_TRUE(t3.result.has_value());
  EXPECT_EQ(t3.result->sessionId, sessionId);
  releaseSlot(manager, sessionId);
}

TEST(SessionManagerGetSlot, PrefixCacheIndex_SurvivesResponseIdContinuation) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto turn1Prompt = makeThreeBlockPrompt();
  auto turn1Blocks = manager.computeBlockInfos(turn1Prompt);
  ASSERT_EQ(turn1Blocks.size(), 3u);

  auto sessionId =
      bootstrapSessionWithResponseId(manager, lf.loop, 60u, "r1", turn1Blocks);
  ASSERT_FALSE(sessionId.empty());

  auto [matchedTokens1, thinkTokens1] =
      manager.computeMatchedTokens(sessionId, turn1Blocks);
  EXPECT_GT(matchedTokens1, 0u);

  tt::services::GetSlotOptions turn2;
  turn2.previousResponseId = "r1";
  turn2.responseId = "r2";
  auto t2 = callGetSlot(manager, lf.loop, turn1Prompt, turn2);
  ASSERT_TRUE(t2.result.has_value());
  EXPECT_EQ(t2.result->sessionId, sessionId);
  EXPECT_EQ(t2.result->slotId, 60u);

  auto [matchedTokens2, thinkTokens2] =
      manager.computeMatchedTokens(sessionId, turn1Blocks);
  EXPECT_EQ(matchedTokens2, matchedTokens1);

  releaseSlot(manager, sessionId);

  auto turn2Prompt = makeFourBlockPrompt();
  auto turn2Blocks = manager.computeBlockInfos(turn2Prompt);
  ASSERT_EQ(turn2Blocks.size(), 4u);

  auto [matchedTokens3, thinkTokens3] =
      manager.computeMatchedTokens(sessionId, turn2Blocks);
  EXPECT_EQ(matchedTokens3, matchedTokens1)
      << "index still reflects turn-1 blocks before turn 3 acquire";

  tt::services::GetSlotOptions turn3;
  turn3.previousResponseId = "r2";
  auto t3 = callGetSlot(manager, lf.loop, turn2Prompt, turn3);
  ASSERT_TRUE(t3.result.has_value());
  EXPECT_EQ(t3.result->sessionId, sessionId);

  auto [matchedTokens4, thinkTokens4] =
      manager.computeMatchedTokens(sessionId, turn2Blocks);
  EXPECT_GT(matchedTokens4, matchedTokens1);

  auto [matchedTokens5, thinkTokens5] =
      manager.computeMatchedTokens(sessionId, turn1Blocks);
  EXPECT_EQ(matchedTokens5, matchedTokens1);

  releaseSlot(manager, sessionId);
}

// ---------------------------------------------------------------------------
// clearSessionBlockThinkTokens tests
// ---------------------------------------------------------------------------

TEST(SessionManagerClearThinkTokens, ResetsThinkTokensToZero) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  // Register a session with blocks that have non-zero think token counts.
  std::vector<tt::utils::BlockHashInfo> blocks = {
      {100, 5},   // key block with 5 think tokens
      {200, 12},  // remaining block 1 with 12 accumulated think tokens
      {300, 20},  // remaining block 2 with 20 accumulated think tokens
  };
  auto sessionId = createSessionWithSlot(manager, lf.loop, 70u, blocks);
  ASSERT_FALSE(sessionId.empty());

  // Verify think tokens are reported before clearing.
  auto [matchedBefore, thinkBefore] =
      manager.computeMatchedTokens(sessionId, blocks);
  EXPECT_GT(matchedBefore, 0u);
  EXPECT_EQ(thinkBefore, 20u)
      << "Think tokens should reflect the last matched block's accumulated "
         "count";

  // Clear think tokens for this session.
  manager.clearSessionBlockThinkTokens(sessionId);

  // After clearing, computeMatchedTokens should still match all blocks (hashes
  // are unchanged) but report 0 think tokens.
  auto [matchedAfter, thinkAfter] =
      manager.computeMatchedTokens(sessionId, blocks);
  EXPECT_EQ(matchedAfter, matchedBefore)
      << "Block matching should be unaffected (hashes unchanged)";
  EXPECT_EQ(thinkAfter, 0u)
      << "Think tokens should be 0 after clearSessionBlockThinkTokens";
}

TEST(SessionManagerClearThinkTokens, NoOpForUnknownSession) {
  tt::services::SessionManager manager;

  // Should not crash when called with a session that doesn't exist.
  EXPECT_NO_THROW(manager.clearSessionBlockThinkTokens("nonexistent-session"));
}

TEST(SessionManagerClearThinkTokens, NoOpForSessionWithNoBlocks) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  // Create a session without any block infos (hash will be 0).
  auto sessionId = createSessionWithSlot(manager, lf.loop, 71u);
  ASSERT_FALSE(sessionId.empty());

  // Should not crash when the session has no prefix index entry.
  EXPECT_NO_THROW(manager.clearSessionBlockThinkTokens(sessionId));
}

}  // namespace
