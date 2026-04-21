// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/session.hpp"

#include <gtest/gtest.h>
#include <trantor/net/EventLoop.h>

#include <atomic>
#include <future>
#include <string>
#include <thread>

#include "services/session_manager.hpp"

namespace {

// ---------------------------------------------------------------------------
// Session state machine transition tests
// ---------------------------------------------------------------------------

TEST(SessionState, InitialStateIsIdle) {
  tt::domain::Session s;
  EXPECT_TRUE(s.isIdle());
  EXPECT_FALSE(s.isInFlight());
  EXPECT_FALSE(s.isCloseRequested());
  EXPECT_FALSE(s.isClosing());
}

TEST(SessionState, MarkInFlightFromIdle) {
  tt::domain::Session s;
  EXPECT_TRUE(s.markInFlight());
  EXPECT_TRUE(s.isInFlight());
  EXPECT_FALSE(s.isIdle());
}

TEST(SessionState, MarkInFlightFromNonIdleReturnsFalseAndPreservesState) {
  tt::domain::Session s;
  s.markInFlight();
  EXPECT_FALSE(s.markInFlight());  // already IN_FLIGHT
  EXPECT_TRUE(s.isInFlight());
}

TEST(SessionState, ClearInFlightFromInFlightTransitionsToIdle) {
  tt::domain::Session s;
  s.markInFlight();
  EXPECT_TRUE(s.clearInFlight());
  EXPECT_TRUE(s.isIdle());
}

TEST(SessionState, ClearInFlightFromCloseRequestedTransitionsToClosing) {
  tt::domain::Session s;
  s.markInFlight();
  s.markCloseRequested();
  EXPECT_TRUE(s.clearInFlight());
  EXPECT_TRUE(s.isClosing());
}

TEST(SessionState, ClearInFlightFromIdleReturnsFalse) {
  tt::domain::Session s;
  EXPECT_FALSE(s.clearInFlight());
  EXPECT_TRUE(s.isIdle());  // state unchanged
}

TEST(SessionState, MarkCloseRequestedFromInFlight) {
  tt::domain::Session s;
  s.markInFlight();
  EXPECT_TRUE(s.markCloseRequested());
  EXPECT_TRUE(s.isCloseRequested());
}

TEST(SessionState, MarkCloseRequestedFromIdleReturnsFalse) {
  tt::domain::Session s;
  EXPECT_FALSE(s.markCloseRequested());
  EXPECT_TRUE(s.isIdle());  // state unchanged
}

TEST(SessionState, MarkCloseRequestedFromCloseRequestedReturnsFalse) {
  tt::domain::Session s;
  s.markInFlight();
  s.markCloseRequested();
  EXPECT_FALSE(s.markCloseRequested());
  EXPECT_TRUE(s.isCloseRequested());  // state unchanged
}

// ---------------------------------------------------------------------------
// SessionManager abort callback tests
// ---------------------------------------------------------------------------

// Helper: creates a trantor event loop on a background thread (Trantor
// requires the loop to be created and run on the same thread).
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
