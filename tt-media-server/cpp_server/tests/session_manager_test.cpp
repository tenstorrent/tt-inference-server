// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/session_manager.hpp"

#include <gtest/gtest.h>
#include <trantor/net/EventLoop.h>

#include <atomic>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "domain/session.hpp"
#include "utils/conversation_hasher.hpp"

namespace {

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

uint32_t createSessionWithSlot(tt::services::SessionManager& manager,
                               uint32_t slotId) {
  manager.createSession(slotId);
  return slotId;
}

uint32_t createSessionWithSlot(
    tt::services::SessionManager& manager, uint32_t slotId,
    const std::vector<tt::utils::BlockHashInfo>& blockInfos) {
  manager.createSession(slotId, blockInfos);
  return slotId;
}

// Convenience: acquire with no cancel function.
void acquireInFlight(tt::services::SessionManager& manager, uint32_t slotId) {
  manager.acquireInFlight(slotId, nullptr);
}

// ---------------------------------------------------------------------------
// SessionManager lifecycle tests
// ---------------------------------------------------------------------------

TEST(SessionManagerLifecycle, CloseIdleSession_ReturnsSuccess) {
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 10u);

  EXPECT_EQ(manager.closeSession(slotId),
            tt::services::CloseSessionResult::SUCCESS);
  EXPECT_FALSE(manager.getSession(slotId));
}

TEST(SessionManagerLifecycle, CloseNonExistentSession_ReturnsNotFound) {
  tt::services::SessionManager manager;

  EXPECT_EQ(manager.closeSession(999u),
            tt::services::CloseSessionResult::NOT_FOUND);
}

TEST(SessionManagerLifecycle, AcquireInFlight_SessionExists) {
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 7u);

  acquireInFlight(manager, slotId);
  auto session = manager.getSession(slotId);
  ASSERT_TRUE(session);
  EXPECT_EQ(session->getSlotId(), 7u);
  session->clearInFlight();
}

TEST(SessionManagerLifecycle, AcquireInFlight_AlreadyInFlight_Throws) {
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 8u);

  acquireInFlight(manager, slotId);
  EXPECT_THROW(acquireInFlight(manager, slotId),
               tt::services::SessionInFlightException);
  auto session = manager.getSession(slotId);
  ASSERT_TRUE(session);
  session->clearInFlight();
}

TEST(SessionManagerLifecycle, CloseWhileInFlight_RemovesSessionImmediately) {
  // closeSession must remove the session and trigger dealloc immediately,
  // even when the session is in-flight. Session clearInFlight called afterwards
  // would fail since session is already gone (we don't test that here since
  // we can't get the session pointer after it's closed).
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 9u);

  acquireInFlight(manager, slotId);
  manager.closeSession(slotId);
  EXPECT_FALSE(manager.getSession(slotId));
}

TEST(SessionManagerLifecycle, GetActiveSessionCount_ReflectsLifecycle) {
  tt::services::SessionManager manager;

  EXPECT_EQ(manager.getActiveSessionCount(), 0u);

  auto s1 = createSessionWithSlot(manager, 20u);
  EXPECT_EQ(manager.getActiveSessionCount(), 1u);

  auto s2 = createSessionWithSlot(manager, 21u);
  EXPECT_EQ(manager.getActiveSessionCount(), 2u);

  manager.closeSession(s1);
  EXPECT_EQ(manager.getActiveSessionCount(), 1u);

  manager.closeSession(s2);
  EXPECT_EQ(manager.getActiveSessionCount(), 0u);
}

TEST(SessionManagerLifecycle, AcquireAfterRelease_Succeeds) {
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 11u);

  acquireInFlight(manager, slotId);
  auto session = manager.getSession(slotId);
  ASSERT_TRUE(session);
  session->clearInFlight();

  EXPECT_NO_THROW(acquireInFlight(manager, slotId));
  session = manager.getSession(slotId);
  ASSERT_TRUE(session);
  session->clearInFlight();
}

TEST(SessionManagerLifecycle, GetSession_ReturnsCorrectData) {
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 12u);

  auto session = manager.getSession(slotId);
  ASSERT_TRUE(session);
  EXPECT_EQ(session->getSlotId(), 12u);
}

// ---------------------------------------------------------------------------
// SessionManager close-while-in-flight tests
// ---------------------------------------------------------------------------

TEST(SessionManagerClose, CloseInFlight_RemovesSessionImmediately) {
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 42u);

  acquireInFlight(manager, slotId);

  EXPECT_EQ(manager.closeSession(slotId),
            tt::services::CloseSessionResult::SUCCESS);
  EXPECT_FALSE(manager.getSession(slotId));
}

TEST(SessionManagerClose, CloseInFlight_FiresCancelFn_AtomicWithAcquire) {
  // Cancel and in-flight state are set atomically by acquireInFlight.
  // closeSession must fire the cancel function immediately.
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 45u);

  std::atomic<bool> cancelCalled{false};
  manager.acquireInFlight(slotId, [&cancelCalled]() { cancelCalled = true; });

  manager.closeSession(slotId);

  EXPECT_TRUE(cancelCalled.load());
  EXPECT_FALSE(manager.getSession(slotId));
}

TEST(SessionManagerClose, CloseIdle_NoCancelFired) {
  // Idle sessions have no in-flight request; closeSession must not fire any
  // cancel (there is none registered).
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 46u);

  // Close without ever calling acquireInFlight — no cancel should be needed.
  EXPECT_EQ(manager.closeSession(slotId),
            tt::services::CloseSessionResult::SUCCESS);
  EXPECT_FALSE(manager.getSession(slotId));
}

TEST(SessionManagerClose, ReleaseInFlight_AfterClose_IsNoOp) {
  // Simulates the SSE writer attempting to clear in-flight after the session
  // was already removed by a concurrent closeSession. Since the session is
  // gone, we can't call clearInFlight on it (it would be a dangling pointer).
  // This test now just verifies that closing an in-flight session removes it.
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 43u);

  acquireInFlight(manager, slotId);
  manager.closeSession(slotId);

  // Session is gone - we can't call clearInFlight on it
  EXPECT_FALSE(manager.getSession(slotId));
  EXPECT_EQ(manager.getActiveSessionCount(), 0u);
}

TEST(SessionManagerClose, CancelFn_ClearedOnNormalCompletion) {
  // If the request completes normally, session clearInFlight must clear the
  // in-flight state so a subsequent close does not fire stale cancel logic.
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 44u);

  std::atomic<bool> cancelCalled{false};
  manager.acquireInFlight(slotId, [&cancelCalled]() { cancelCalled = true; });

  auto session = manager.getSession(slotId);
  ASSERT_TRUE(session);
  session->clearInFlight();         // normal completion clears in-flight state
  manager.closeSession(slotId);     // should not fire cancel

  EXPECT_FALSE(cancelCalled.load());
}

TEST(SessionManagerClose, ReleaseInFlight_NormalCompletion_SessionStaysIdle) {
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 47u);

  acquireInFlight(manager, slotId);
  auto session = manager.getSession(slotId);
  ASSERT_TRUE(session);
  session->clearInFlight();

  // Session still present and acquirable again.
  EXPECT_TRUE(manager.getSession(slotId));
  EXPECT_NO_THROW(acquireInFlight(manager, slotId));
  session = manager.getSession(slotId);
  ASSERT_TRUE(session);
  session->clearInFlight();
}

// ---------------------------------------------------------------------------
// Concurrency tests
// ---------------------------------------------------------------------------

// Runs a function from two threads simultaneously using a shared latch.
template <typename F>
void runConcurrently(F&& f) {
  std::atomic<bool> ready{false};
  auto t1 = std::thread([&] {
    while (!ready.load(std::memory_order_acquire)) {
    }
    f();
  });
  auto t2 = std::thread([&] {
    while (!ready.load(std::memory_order_acquire)) {
    }
    f();
  });
  ready.store(true, std::memory_order_release);
  t1.join();
  t2.join();
}

TEST(SessionManagerConcurrency, ConcurrentClose_OnlyOneSucceeds) {
  // Two threads race to close the same session. Exactly one must get SUCCESS;
  // the other must get NOT_FOUND. The session must be gone afterwards.
  constexpr int iterations = 200;
  for (int i = 0; i < iterations; ++i) {
    tt::services::SessionManager manager;
    auto slotId = createSessionWithSlot(manager, 100u);

    std::atomic<int> successCount{0};
    runConcurrently([&] {
      auto result = manager.closeSession(slotId);
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
    auto slotId = createSessionWithSlot(manager, 101u);

    std::atomic<int> acquireCount{0};
    runConcurrently([&] {
      try {
        manager.acquireInFlight(slotId, nullptr);
        acquireCount.fetch_add(1, std::memory_order_relaxed);
      } catch (const tt::services::SessionInFlightException&) {
      }
    });

    EXPECT_EQ(acquireCount.load(), 1) << "iteration " << i;
    auto session = manager.getSession(slotId);
    if (session) {
      session->clearInFlight();
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
    auto slotId = createSessionWithSlot(manager, 102u);

    std::atomic<int> cancelCount{0};
    std::atomic<bool> ready{false};

    std::thread acquirer([&] {
      while (!ready.load(std::memory_order_acquire)) {
      }
      try {
        manager.acquireInFlight(slotId,
                                [&cancelCount] { cancelCount.fetch_add(1); });
        auto session = manager.getSession(slotId);
        if (session) {
          session->clearInFlight();
        }
      } catch (const tt::services::SessionRateLimitException&) {
      }
    });

    std::thread closer([&] {
      while (!ready.load(std::memory_order_acquire)) {
      }
      manager.closeSession(slotId);
    });

    ready.store(true, std::memory_order_release);
    acquirer.join();
    closer.join();

    EXPECT_LE(cancelCount.load(), 1) << "iteration " << i;
    EXPECT_EQ(manager.getActiveSessionCount(), 0u) << "iteration " << i;
  }
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

  auto slotId = createSessionWithSlot(manager, 50u);

  manager.initResponseId(slotId, "resp-1");

  auto acquired = manager.tryAcquireByResponseId("resp-1", nullptr);
  ASSERT_TRUE(acquired.has_value());
  EXPECT_EQ(acquired->slotId, 50u);

  auto session = manager.getSession(slotId);
  ASSERT_TRUE(session);
  session->clearInFlight();
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

  auto slotId = createSessionWithSlot(manager, 51u);

  manager.initResponseId(slotId, "");

  auto session = manager.getSession(slotId);
  ASSERT_TRUE(session);
  EXPECT_TRUE(session->getResponseId().empty());
}

TEST(SessionManagerResponseId, ReKey_MovesSessionToNewId) {
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 52u);

  manager.initResponseId(slotId, "resp-1");
  manager.registerResponseId("resp-1", "resp-2");

  // The previous turn's id no longer resolves once re-keyed.
  EXPECT_FALSE(manager.tryAcquireByResponseId("resp-1", nullptr).has_value());

  // The new id resolves.
  auto acquired = manager.tryAcquireByResponseId("resp-2", nullptr);
  ASSERT_TRUE(acquired.has_value());
  EXPECT_EQ(acquired->slotId, slotId);

  manager.getSession(slotId)->clearInFlight();
}

TEST(SessionManagerResponseId, AcquireMarksInFlight_SecondAcquireThrows) {
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 53u);

  manager.initResponseId(slotId, "resp-1");

  auto acquired = manager.tryAcquireByResponseId("resp-1", nullptr);
  ASSERT_TRUE(acquired.has_value());

  // The only session under this id is now in-flight → maps to HTTP 429.
  EXPECT_THROW(manager.tryAcquireByResponseId("resp-1", nullptr),
               tt::services::SessionInFlightException);

  manager.getSession(slotId)->clearInFlight();
}

TEST(SessionManagerResponseId, AcquireAfterRelease_Succeeds) {
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 54u);

  manager.initResponseId(slotId, "resp-1");

  auto first = manager.tryAcquireByResponseId("resp-1", nullptr);
  ASSERT_TRUE(first.has_value());
  manager.getSession(slotId)->clearInFlight();

  auto second = manager.tryAcquireByResponseId("resp-1", nullptr);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->slotId, slotId);
  manager.getSession(slotId)->clearInFlight();
}

TEST(SessionManagerResponseId, CloseSession_RemovesFromResponseIdIndex) {
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 56u);

  manager.initResponseId(slotId, "resp-1");
  ASSERT_EQ(manager.closeSession(slotId),
            tt::services::CloseSessionResult::SUCCESS);

  // The index entry must be gone after the session is closed.
  EXPECT_FALSE(manager.tryAcquireByResponseId("resp-1", nullptr).has_value());
}

TEST(SessionManagerResponseId, CloseWhileAcquired_FiresCancelFn) {
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 57u);

  manager.initResponseId(slotId, "resp-1");

  std::atomic<bool> cancelCalled{false};
  auto acquired = manager.tryAcquireByResponseId(
      "resp-1", [&cancelCalled]() { cancelCalled = true; });
  ASSERT_TRUE(acquired.has_value());

  // The cancel fn registered atomically with the in-flight mark must fire.
  manager.closeSession(slotId);
  EXPECT_TRUE(cancelCalled.load());
  EXPECT_FALSE(manager.getSession(slotId));
}

TEST(SessionManagerResponseId, TwoTurnContinuation_ReKeysAcrossIds) {
  // Simulates the two-turn response-id flow: turn 1 registers the session
  // under id "r1"; turn 2 acquires by "r1", re-keys under "r2" for turn 3.
  tt::services::SessionManager manager;

  auto slotId = createSessionWithSlot(manager, 58u);

  manager.initResponseId(slotId, "r1");

  // Turn 2: arrives with previous_response_id="r1".
  auto t2 = manager.tryAcquireByResponseId("r1", nullptr);
  ASSERT_TRUE(t2.has_value());
  EXPECT_EQ(t2->slotId, slotId);
  // Re-key under turn 2's own id.
  manager.registerResponseId("r1", "r2");
  manager.getSession(slotId)->clearInFlight();

  // Turn 3: arrives with previous_response_id="r2".
  auto t3 = manager.tryAcquireByResponseId("r2", nullptr);
  ASSERT_TRUE(t3.has_value());
  EXPECT_EQ(t3->slotId, slotId);
  EXPECT_FALSE(manager.tryAcquireByResponseId("r1", nullptr).has_value());
  manager.getSession(slotId)->clearInFlight();
}

TEST(SessionManagerResponseId,
     PrefixCacheIndex_HitAndUpdated_ViaResponseIdPath) {
  // Verifies that the prefix cache index is populated when a session is
  // created with block infos, remains queryable after acquisition through the
  // response-id path, and reflects updated blocks after re-registration.
  tt::services::SessionManager manager;

  // --- Turn 1: create session with 3 initial blocks ---
  std::vector<tt::utils::BlockHashInfo> turn1Blocks = {
      {100, 0},  // key block
      {200, 0},  // remaining block 1
      {300, 0},  // remaining block 2
  };
  auto slotId = createSessionWithSlot(manager, 60u, turn1Blocks);

  // Prefix index should reflect all 3 blocks for this session.
  auto [matchedTokens1, thinkTokens1] =
      manager.computeMatchedTokens(slotId, turn1Blocks);
  EXPECT_GT(matchedTokens1, 0u)
      << "prefixCacheIndex should have been populated by createSession";

  // Register the session under response id "r1" and prefix hash.
  manager.initResponseId(slotId, "r1");

  manager.registerPrefixHash(slotId, turn1Blocks);
  // --- Turn 2: arrive via previous_response_id="r1" ---
  auto t2 = manager.tryAcquireByResponseId("r1", nullptr);
  ASSERT_TRUE(t2.has_value());
  EXPECT_EQ(t2->slotId, 60u);

  // While acquired through the response-id path, the prefix cache index
  // should still be intact and report the same match.
  auto [matchedTokens2, thinkTokens2] =
      manager.computeMatchedTokens(slotId, turn1Blocks);
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
  manager.registerPrefixHash(slotId, turn2Blocks);
  manager.registerResponseId("r1", "r2");
  manager.getSession(slotId)->clearInFlight();

  // The prefix index should now match all 4 blocks.
  auto [matchedTokens3, thinkTokens3] =
      manager.computeMatchedTokens(slotId, turn2Blocks);
  EXPECT_GT(matchedTokens3, matchedTokens1)
      << "prefixCacheIndex should reflect the updated (longer) block sequence";

  // The original 3-block query should still match its 3 blocks (prefix).
  auto [matchedTokens4, thinkTokens4] =
      manager.computeMatchedTokens(slotId, turn1Blocks);
  EXPECT_EQ(matchedTokens4, matchedTokens1)
      << "shorter prefix query should still match the original blocks";

  // --- Turn 3: arrive via previous_response_id="r2" ---
  auto t3 = manager.tryAcquireByResponseId("r2", nullptr);
  ASSERT_TRUE(t3.has_value());
  EXPECT_EQ(t3->slotId, slotId);

  // Prefix index should still be consistent after the second response-id hop.
  auto [matchedTokens5, thinkTokens5] =
      manager.computeMatchedTokens(slotId, turn2Blocks);
  EXPECT_EQ(matchedTokens5, matchedTokens3)
      << "prefixCacheIndex should survive re-keying across response ids";

  manager.getSession(slotId)->clearInFlight();
}

// ---------------------------------------------------------------------------
// clearSessionBlockThinkTokens tests
// ---------------------------------------------------------------------------

TEST(SessionManagerClearThinkTokens, ResetsThinkTokensToZero) {
  tt::services::SessionManager manager;

  // Register a session with blocks that have non-zero think token counts.
  std::vector<tt::utils::BlockHashInfo> blocks = {
      {100, 5},   // key block with 5 think tokens
      {200, 12},  // remaining block 1 with 12 accumulated think tokens
      {300, 20},  // remaining block 2 with 20 accumulated think tokens
  };
  auto slotId = createSessionWithSlot(manager, 70u, blocks);

  // Verify think tokens are reported before clearing.
  auto [matchedBefore, thinkBefore] =
      manager.computeMatchedTokens(slotId, blocks);
  EXPECT_GT(matchedBefore, 0u);
  EXPECT_EQ(thinkBefore, 20u)
      << "Think tokens should reflect the last matched block's accumulated "
         "count";

  // Clear think tokens for this session.
  manager.clearSessionBlockThinkTokens(slotId);

  // After clearing, computeMatchedTokens should still match all blocks (hashes
  // are unchanged) but report 0 think tokens.
  auto [matchedAfter, thinkAfter] =
      manager.computeMatchedTokens(slotId, blocks);
  EXPECT_EQ(matchedAfter, matchedBefore)
      << "Block matching should be unaffected (hashes unchanged)";
  EXPECT_EQ(thinkAfter, 0u)
      << "Think tokens should be 0 after clearSessionBlockThinkTokens";
}

TEST(SessionManagerClearThinkTokens, NoOpForUnknownSession) {
  tt::services::SessionManager manager;

  // Should not crash when called with a session that doesn't exist.
  EXPECT_NO_THROW(manager.clearSessionBlockThinkTokens(999u));
}

TEST(SessionManagerClearThinkTokens, NoOpForSessionWithNoBlocks) {
  tt::services::SessionManager manager;

  // Create a session without any block infos (hash will be 0).
  auto slotId = createSessionWithSlot(manager, 71u);

  // Should not crash when the session has no prefix index entry.
  EXPECT_NO_THROW(manager.clearSessionBlockThinkTokens(slotId));
}

}  // namespace
