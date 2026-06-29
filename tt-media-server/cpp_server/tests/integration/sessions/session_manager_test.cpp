// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/session_manager.hpp"

#include <gtest/gtest.h>
#include <trantor/net/EventLoop.h>

#include <atomic>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "../integration_test_helpers.hpp"
#include "domain/session.hpp"
#include "utils/conversation_hasher.hpp"

namespace {

using tt::test::acquireInFlight;
using tt::test::createTestSession;
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
// Prefix-cache slot-copy tests
// ---------------------------------------------------------------------------

TEST(SessionManagerPrefixCache, SlotCopySkipsThresholdRejectedBusyCandidate) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto makeBlocks = [](size_t count) {
    std::vector<tt::utils::BlockHashInfo> blocks;
    blocks.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      blocks.push_back({static_cast<uint64_t>(1000 + i), 0});
    }
    return blocks;
  };

  // Issue #4401 shape: 35/1283 blocks is below the 40% hit threshold.
  auto rejectedBlocks = makeBlocks(1283);
  std::vector<tt::utils::BlockHashInfo> requestBlocks(
      rejectedBlocks.begin(), rejectedBlocks.begin() + 35);
  std::vector<tt::utils::BlockHashInfo> qualifyingBlocks(
      rejectedBlocks.begin(), rejectedBlocks.begin() + 34);

  auto rejectedSessionId =
      createSessionWithSlot(manager, lf.loop, 2u, rejectedBlocks);
  ASSERT_FALSE(rejectedSessionId.empty());

  manager.setResidentPrefixBlocks(rejectedSessionId, rejectedBlocks.size());
  acquireInFlight(manager, rejectedSessionId);

  const char* previousThreshold = std::getenv("PREFIX_CACHE_HIT_THRESHOLD");
  const std::string previousValue =
      previousThreshold == nullptr ? "" : previousThreshold;
  setenv("PREFIX_CACHE_HIT_THRESHOLD", "40", 1);

  auto acquired = manager.tryAcquireByPrefixHash(requestBlocks, nullptr);
  if (!acquired.has_value()) {
    ADD_FAILURE() << "busy prefix candidates should be returned";
  } else {
    EXPECT_FALSE(acquired->sessionFound);
    EXPECT_FALSE(manager.findASlotToCopyFrom(acquired->candidatesList))
        << "threshold-rejected candidates must not be copy sources";
  }

  auto qualifyingSessionId =
      createSessionWithSlot(manager, lf.loop, 3u, qualifyingBlocks);
  ASSERT_FALSE(qualifyingSessionId.empty());
  manager.setResidentPrefixBlocks(qualifyingSessionId, qualifyingBlocks.size());
  acquireInFlight(manager, qualifyingSessionId);

  acquired = manager.tryAcquireByPrefixHash(requestBlocks, nullptr);
  if (!acquired.has_value()) {
    ADD_FAILURE() << "valid busy candidates should be returned";
  } else {
    EXPECT_FALSE(acquired->sessionFound);
    auto copyCandidate = manager.findASlotToCopyFrom(acquired->candidatesList);
    if (!copyCandidate.has_value()) {
      ADD_FAILURE()
          << "the threshold-valid busy session should remain copy-eligible";
    } else {
      EXPECT_EQ(copyCandidate->sessionId, qualifyingSessionId)
          << "slot copy must skip the threshold-rejected session";
    }
  }

  if (previousThreshold == nullptr) {
    unsetenv("PREFIX_CACHE_HIT_THRESHOLD");
  } else {
    setenv("PREFIX_CACHE_HIT_THRESHOLD", previousValue.c_str(), 1);
  }
  manager.getSession(rejectedSessionId)->release();
  manager.getSession(qualifyingSessionId)->release();
}

// ---------------------------------------------------------------------------
// Response-id continuation tests
//
// These cover the OpenAI Responses API routing path: initResponseId stores
// a session under a response id for the first time, registerResponseId re-keys
// from one id to another, and tryAcquireByResponseId resolves that id back to
// the session/slot. The prefix delta is derived from block matching
// (computeMatchedTokens), not stored in the response-id index.
// ---------------------------------------------------------------------------

TEST(SessionManagerResponseId, RegisterThenAcquire_ReturnsSessionAndSlot) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 50u);
  ASSERT_FALSE(sessionId.empty());

  manager.initResponseId(sessionId, "resp-1");

  auto acquired = manager.tryAcquireByResponseId("resp-1", nullptr);
  ASSERT_TRUE(acquired.has_value());
  EXPECT_EQ(acquired->sessionId, sessionId);
  EXPECT_EQ(acquired->slotId, 50u);

  auto session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  session->release();
}

TEST(SessionManagerResponseId, AcquireUnknownId_ReturnsNullopt) {
  tt::services::SessionManager manager;
  EXPECT_FALSE(
      manager.tryAcquireByResponseId("no-such-id", nullptr).has_value());
}

TEST(SessionManagerResponseId, AcquireEmptyId_ReturnsNullopt) {
  tt::services::SessionManager manager;
  EXPECT_FALSE(manager.tryAcquireByResponseId("", nullptr).has_value());
}

TEST(SessionManagerResponseId, RegisterEmptyId_IsNoOp) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 51u);
  ASSERT_FALSE(sessionId.empty());

  manager.initResponseId(sessionId, "");

  auto session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  EXPECT_TRUE(session->getResponseId().empty());
}

TEST(SessionManagerResponseId, ReKey_MovesSessionToNewId) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 52u);
  ASSERT_FALSE(sessionId.empty());

  manager.initResponseId(sessionId, "resp-1");
  manager.registerResponseId("resp-1", "resp-2");

  // The previous turn's id no longer resolves once re-keyed.
  EXPECT_FALSE(manager.tryAcquireByResponseId("resp-1", nullptr).has_value());

  // The new id resolves.
  auto acquired = manager.tryAcquireByResponseId("resp-2", nullptr);
  ASSERT_TRUE(acquired.has_value());
  EXPECT_EQ(acquired->sessionId, sessionId);

  manager.getSession(sessionId)->release();
}

TEST(SessionManagerResponseId, AcquireMarksInFlight_SecondAcquireThrows) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 53u);
  ASSERT_FALSE(sessionId.empty());

  manager.initResponseId(sessionId, "resp-1");

  auto acquired = manager.tryAcquireByResponseId("resp-1", nullptr);
  ASSERT_TRUE(acquired.has_value());

  // The only session under this id is now in-flight → maps to HTTP 429.
  EXPECT_THROW(manager.tryAcquireByResponseId("resp-1", nullptr),
               tt::services::SessionInFlightException);

  manager.getSession(sessionId)->release();
}

TEST(SessionManagerResponseId, AcquireAfterRelease_Succeeds) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 54u);
  ASSERT_FALSE(sessionId.empty());

  manager.initResponseId(sessionId, "resp-1");

  auto first = manager.tryAcquireByResponseId("resp-1", nullptr);
  ASSERT_TRUE(first.has_value());
  manager.getSession(sessionId)->release();

  auto second = manager.tryAcquireByResponseId("resp-1", nullptr);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->sessionId, sessionId);
  manager.getSession(sessionId)->release();
}

TEST(SessionManagerResponseId, CloseSession_RemovesFromResponseIdIndex) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 56u);
  ASSERT_FALSE(sessionId.empty());

  manager.initResponseId(sessionId, "resp-1");
  ASSERT_EQ(manager.closeSession(sessionId),
            tt::services::CloseSessionResult::SUCCESS);

  // The index entry must be gone after the session is closed.
  EXPECT_FALSE(manager.tryAcquireByResponseId("resp-1", nullptr).has_value());
}

TEST(SessionManagerResponseId, CloseWhileAcquired_FiresCancelFn) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 57u);
  ASSERT_FALSE(sessionId.empty());

  manager.initResponseId(sessionId, "resp-1");

  std::atomic<bool> cancelCalled{false};
  auto acquired = manager.tryAcquireByResponseId(
      "resp-1", [&cancelCalled]() { cancelCalled = true; });
  ASSERT_TRUE(acquired.has_value());

  // The cancel fn registered atomically with the in-flight mark must fire.
  manager.closeSession(sessionId);
  EXPECT_TRUE(cancelCalled.load());
  EXPECT_FALSE(manager.getSession(sessionId));
}

TEST(SessionManagerResponseId, TwoTurnContinuation_ReKeysAcrossIds) {
  // Simulates the two-turn response-id flow: turn 1 registers the session
  // under id "r1"; turn 2 acquires by "r1", re-keys under "r2" for turn 3.
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 58u);
  ASSERT_FALSE(sessionId.empty());

  manager.initResponseId(sessionId, "r1");

  // Turn 2: arrives with previous_response_id="r1".
  auto t2 = manager.tryAcquireByResponseId("r1", nullptr);
  ASSERT_TRUE(t2.has_value());
  EXPECT_EQ(t2->sessionId, sessionId);
  // Re-key under turn 2's own id.
  manager.registerResponseId("r1", "r2");
  manager.getSession(sessionId)->release();

  // Turn 3: arrives with previous_response_id="r2".
  auto t3 = manager.tryAcquireByResponseId("r2", nullptr);
  ASSERT_TRUE(t3.has_value());
  EXPECT_EQ(t3->sessionId, sessionId);
  EXPECT_FALSE(manager.tryAcquireByResponseId("r1", nullptr).has_value());
  manager.getSession(sessionId)->release();
}

TEST(SessionManagerResponseId,
     PrefixCacheIndex_HitAndUpdated_ViaResponseIdPath) {
  // Verifies that the prefix cache index is populated when a session is
  // created with block infos, remains queryable after acquisition through the
  // response-id path, and reflects updated blocks after re-registration.
  tt::services::SessionManager manager;
  LoopFixture lf;

  // --- Turn 1: create session with 3 initial blocks ---
  std::vector<tt::utils::BlockHashInfo> turn1Blocks = {
      {100, 0},  // key block
      {200, 0},  // remaining block 1
      {300, 0},  // remaining block 2
  };
  auto sessionId = createSessionWithSlot(manager, lf.loop, 60u, turn1Blocks);
  ASSERT_FALSE(sessionId.empty());

  // Prefix index should reflect all 3 blocks for this session.
  auto [matchedTokens1, thinkTokens1] =
      manager.computeMatchedTokens(sessionId, turn1Blocks);
  EXPECT_GT(matchedTokens1, 0u)
      << "prefixCacheIndex should have been populated by createSession";

  // Register the session under response id "r1" and prefix hash.
  manager.initResponseId(sessionId, "r1");

  manager.registerPrefixHash(sessionId, turn1Blocks);
  // --- Turn 2: arrive via previous_response_id="r1" ---
  auto t2 = manager.tryAcquireByResponseId("r1", nullptr);
  ASSERT_TRUE(t2.has_value());
  EXPECT_EQ(t2->sessionId, sessionId);
  EXPECT_EQ(t2->slotId, 60u);

  // While acquired through the response-id path, the prefix cache index
  // should still be intact and report the same match.
  auto [matchedTokens2, thinkTokens2] =
      manager.computeMatchedTokens(sessionId, turn1Blocks);
  EXPECT_EQ(matchedTokens2, matchedTokens1)
      << "prefixCacheIndex should still be queryable after response-id acquire";

  // Simulate turn 2 producing more tokens: update the prefix hash with an
  // extended block sequence (original 3 blocks + 1 new block).
  std::vector<tt::utils::BlockHashInfo> turn2Blocks = {
      {100, 0},  // same key block
      {200, 0},  // same remaining block 1
      {300, 0},  // same remaining block 2
      {400, 0},  // new block from turn 2's output
  };
  manager.registerPrefixHash(sessionId, turn2Blocks);
  manager.registerResponseId("r1", "r2");
  manager.getSession(sessionId)->release();

  // The prefix index should now match all 4 blocks.
  auto [matchedTokens3, thinkTokens3] =
      manager.computeMatchedTokens(sessionId, turn2Blocks);
  EXPECT_GT(matchedTokens3, matchedTokens1)
      << "prefixCacheIndex should reflect the updated (longer) block sequence";

  // The original 3-block query should still match its 3 blocks (prefix).
  auto [matchedTokens4, thinkTokens4] =
      manager.computeMatchedTokens(sessionId, turn1Blocks);
  EXPECT_EQ(matchedTokens4, matchedTokens1)
      << "shorter prefix query should still match the original blocks";

  // --- Turn 3: arrive via previous_response_id="r2" ---
  auto t3 = manager.tryAcquireByResponseId("r2", nullptr);
  ASSERT_TRUE(t3.has_value());
  EXPECT_EQ(t3->sessionId, sessionId);

  // Prefix index should still be consistent after the second response-id hop.
  auto [matchedTokens5, thinkTokens5] =
      manager.computeMatchedTokens(sessionId, turn2Blocks);
  EXPECT_EQ(matchedTokens5, matchedTokens3)
      << "prefixCacheIndex should survive re-keying across response ids";

  manager.getSession(sessionId)->release();
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
