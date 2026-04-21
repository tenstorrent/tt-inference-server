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

TEST(SessionManagerLifecycle, AcquireSlot_ReturnsPreAssignedSlotId) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 7u);
  ASSERT_FALSE(sessionId.empty());

  EXPECT_EQ(manager.acquireSessionSlot(sessionId), 7u);
  manager.releaseInFlight(sessionId);
}

TEST(SessionManagerLifecycle, AcquireSlot_InFlightSession_Throws) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 8u);
  ASSERT_FALSE(sessionId.empty());

  manager.acquireSessionSlot(sessionId);  // IDLE -> IN_FLIGHT
  EXPECT_THROW(manager.acquireSessionSlot(sessionId),
               tt::services::SessionInFlightException);
  manager.releaseInFlight(sessionId);
}

TEST(SessionManagerLifecycle, DeferredClose_FinalizedOnReleaseInFlight) {
  // CLOSE_REQUESTED path: the session must stay in the map until
  // releaseInFlight completes the CLOSE_REQUESTED -> CLOSING -> finalized
  // transition.
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 9u);
  ASSERT_FALSE(sessionId.empty());

  manager.acquireSessionSlot(sessionId);  // IDLE -> IN_FLIGHT
  manager.closeSession(sessionId);        // IN_FLIGHT -> CLOSE_REQUESTED
  EXPECT_TRUE(manager.getSession(sessionId).has_value());  // still present

  manager.releaseInFlight(
      sessionId);  // CLOSE_REQUESTED -> CLOSING -> finalized
  EXPECT_FALSE(manager.getSession(sessionId).has_value());  // now gone
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

  manager.acquireSessionSlot(sessionId);  // IDLE -> IN_FLIGHT
  manager.releaseInFlight(sessionId);     // IN_FLIGHT -> IDLE

  EXPECT_NO_THROW(manager.acquireSessionSlot(sessionId));
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
// SessionManager abort callback tests
// ---------------------------------------------------------------------------

TEST(SessionManagerAbort, AbortCallbackInvokedOnCloseWhileInFlight) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 42u);
  ASSERT_FALSE(sessionId.empty());

  manager.acquireSessionSlot(sessionId);  // IDLE -> IN_FLIGHT

  std::atomic<bool> abortCalled{false};
  manager.setSessionAbortCallback(sessionId,
                                  [&abortCalled]() { abortCalled = true; });

  auto result = manager.closeSession(sessionId);

  EXPECT_EQ(result, tt::services::CloseSessionResult::SUCCESS);
  EXPECT_TRUE(abortCalled.load());
}

TEST(SessionManagerAbort, CloseSessionReturnsSuccessRegardlessOfInFlight) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 43u);
  ASSERT_FALSE(sessionId.empty());

  manager.acquireSessionSlot(sessionId);  // IDLE -> IN_FLIGHT
  // No abort callback registered (e.g. non-streaming request)

  auto result = manager.closeSession(sessionId);
  EXPECT_EQ(result, tt::services::CloseSessionResult::SUCCESS);
}

TEST(SessionManagerAbort, AbortCallbackClearedAfterRequestCompletesNormally) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  auto sessionId = createSessionWithSlot(manager, lf.loop, 44u);
  ASSERT_FALSE(sessionId.empty());

  manager.acquireSessionSlot(sessionId);  // IDLE -> IN_FLIGHT

  std::atomic<bool> abortCalled{false};
  manager.setSessionAbortCallback(sessionId,
                                  [&abortCalled]() { abortCalled = true; });

  manager.releaseInFlight(sessionId);  // IN_FLIGHT -> IDLE, clears callback

  manager.closeSession(sessionId);

  EXPECT_FALSE(abortCalled.load());
}

}  // namespace
