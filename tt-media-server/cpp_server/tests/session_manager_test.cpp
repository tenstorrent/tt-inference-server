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
#include "domain/slot_types.hpp"

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
      loop, slotId);

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
  EXPECT_FALSE(manager.getSession(sessionId).has_value());
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
  manager.releaseInFlight(sessionId);
}

TEST(SessionManagerLifecycle, AcquireInFlight_AlreadyInFlight_Throws) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 8u);
  ASSERT_FALSE(sessionId.empty());

  acquireInFlight(manager, sessionId);
  EXPECT_THROW(acquireInFlight(manager, sessionId),
               tt::services::SessionInFlightException);
  manager.releaseInFlight(sessionId);
}

TEST(SessionManagerLifecycle, CloseWhileInFlight_RemovesSessionImmediately) {
  // closeSession must remove the session and trigger dealloc immediately,
  // even when the session is in-flight. releaseInFlight called afterwards
  // should be a safe no-op.
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 9u);
  ASSERT_FALSE(sessionId.empty());

  acquireInFlight(manager, sessionId);
  manager.closeSession(sessionId);
  EXPECT_FALSE(manager.getSession(sessionId).has_value());

  // releaseInFlight called by the SSE writer after the request drains must
  // not crash or assert, even though the session is already gone.
  EXPECT_NO_THROW(manager.releaseInFlight(sessionId));
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
  manager.releaseInFlight(sessionId);

  EXPECT_NO_THROW(acquireInFlight(manager, sessionId));
  manager.releaseInFlight(sessionId);
}

TEST(SessionManagerLifecycle, GetSession_ReturnsCorrectData) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 12u);
  ASSERT_FALSE(sessionId.empty());

  auto session = manager.getSession(sessionId);
  ASSERT_TRUE(session.has_value());
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
  ASSERT_TRUE(session.has_value());
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
  EXPECT_FALSE(manager.getSession(sessionId).has_value());
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
  EXPECT_FALSE(manager.getSession(sessionId).has_value());
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
  EXPECT_FALSE(manager.getSession(sessionId).has_value());
}

TEST(SessionManagerClose, ReleaseInFlight_AfterClose_IsNoOp) {
  // Simulates the SSE writer calling releaseInFlight after the session was
  // already removed by a concurrent closeSession.
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 43u);
  ASSERT_FALSE(sessionId.empty());

  acquireInFlight(manager, sessionId);
  manager.closeSession(sessionId);

  EXPECT_NO_THROW(manager.releaseInFlight(sessionId));
  EXPECT_EQ(manager.getActiveSessionCount(), 0u);
}

TEST(SessionManagerClose, CancelFn_ClearedOnNormalCompletion) {
  // If the request completes normally, releaseInFlight must clear the cancel
  // function so a subsequent close does not fire stale cancel logic.
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 44u);
  ASSERT_FALSE(sessionId.empty());

  std::atomic<bool> cancelCalled{false};
  manager.acquireInFlight(sessionId,
                          [&cancelCalled]() { cancelCalled = true; });

  manager.releaseInFlight(sessionId);  // normal completion clears cancel fn
  manager.closeSession(sessionId);     // should not fire cancel

  EXPECT_FALSE(cancelCalled.load());
}

TEST(SessionManagerClose, ReleaseInFlight_NormalCompletion_SessionStaysIdle) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 47u);
  ASSERT_FALSE(sessionId.empty());

  acquireInFlight(manager, sessionId);
  manager.releaseInFlight(sessionId);

  // Session still present and acquirable again.
  EXPECT_TRUE(manager.getSession(sessionId).has_value());
  EXPECT_NO_THROW(acquireInFlight(manager, sessionId));
  manager.releaseInFlight(sessionId);
}

}  // namespace
