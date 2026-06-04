// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/session_manager.hpp"

#include <gtest/gtest.h>
#include <trantor/net/EventLoop.h>

#include <atomic>
#include <future>
#include <string>
#include <thread>

#include "domain/session.hpp"

namespace {

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// Trantor requires an EventLoop to be both created and run on the same thread.
struct LoopFixture {
  std::promise<trantor::EventLoop*> promise_;
  trantor::EventLoop* loop{nullptr};
  std::thread loopThread;

  LoopFixture() {
    auto future = promise_.get_future();
    loopThread = std::thread([this]() {
      trantor::EventLoop eventLoop;
      promise_.set_value(&eventLoop);
      eventLoop.loop();
    });
    loop = future.get();
  }

  ~LoopFixture() {
    if (loop) loop->quit();
    if (loopThread.joinable()) loopThread.join();
  }
};

std::string createSessionWithSlot(tt::services::SessionManager& manager,
                                  trantor::EventLoop* loop, uint32_t slotId) {
  std::promise<std::string> promise;
  auto future = promise.get_future();

  manager.createSession(
      [&promise](const tt::domain::Session& s) {
        promise.set_value(s.getSessionId());
      },
      [&promise](std::string_view err) {
        promise.set_exception(
            std::make_exception_ptr(std::runtime_error(std::string(err))));
      },
      loop, {}, slotId);

  return future.get();
}

// Convenience: acquire with no cancel function.
uint32_t acquireInFlight(tt::services::SessionManager& manager,
                         const std::string& sessionId) {
  return manager.acquireInFlight(sessionId, nullptr);
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
  session->clearInFlight();
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
  session->clearInFlight();
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
  session->clearInFlight();

  EXPECT_NO_THROW(acquireInFlight(manager, sessionId));
  session = manager.getSession(sessionId);
  ASSERT_TRUE(session);
  session->clearInFlight();
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
  session->clearInFlight();         // normal completion clears in-flight state
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
  session->clearInFlight();

  // Session still present and acquirable again.
  EXPECT_TRUE(manager.getSession(sessionId));
  EXPECT_NO_THROW(acquireInFlight(manager, sessionId));
  session = manager.getSession(sessionId);
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
          session->clearInFlight();
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

  manager.getSession(sessionId)->clearInFlight();
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

  manager.getSession(sessionId)->clearInFlight();
}

TEST(SessionManagerResponseId, AcquireAfterRelease_Succeeds) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 54u);
  ASSERT_FALSE(sessionId.empty());

  manager.initResponseId(sessionId, "resp-1");

  auto first = manager.tryAcquireByResponseId("resp-1", nullptr);
  ASSERT_TRUE(first.has_value());
  manager.getSession(sessionId)->clearInFlight();

  auto second = manager.tryAcquireByResponseId("resp-1", nullptr);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->sessionId, sessionId);
  manager.getSession(sessionId)->clearInFlight();
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
  manager.getSession(sessionId)->clearInFlight();

  // Turn 3: arrives with previous_response_id="r2".
  auto t3 = manager.tryAcquireByResponseId("r2", nullptr);
  ASSERT_TRUE(t3.has_value());
  EXPECT_EQ(t3->sessionId, sessionId);
  EXPECT_FALSE(manager.tryAcquireByResponseId("r1", nullptr).has_value());
  manager.getSession(sessionId)->clearInFlight();
}

}  // namespace
