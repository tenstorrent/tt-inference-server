// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/session_manager.hpp"

#include <gtest/gtest.h>
#include <trantor/net/EventLoop.h>

#include <array>
#include <atomic>
#include <chrono>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "../integration_test_helpers.hpp"
#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "domain/session.hpp"
#include "ipc/boost/boost_memory_queue.hpp"
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

// Auto-ack ALLOCATE requests so getSlot() can create sessions without a real
// worker. Must be constructed after SessionManager (which creates the queues).
class MemoryAutoResponder {
 public:
  explicit MemoryAutoResponder(uint32_t firstSlot = 200) : nextSlot(firstSlot) {
    requestQueue_ = tt::ipc::boost::MemoryRequestQueue::openExisting(
        tt::config::ttMemoryRequestQueueName());
    resultQueue_ = tt::ipc::boost::MemoryResultQueue::openExisting(
        tt::config::ttMemoryResultQueueName());
    thread_ = std::thread([this] { run(); });
  }

  ~MemoryAutoResponder() {
    stop_.store(true);
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  MemoryAutoResponder(const MemoryAutoResponder&) = delete;
  MemoryAutoResponder& operator=(const MemoryAutoResponder&) = delete;

 private:
  void run() {
    tt::domain::ManageMemoryTask request{};
    while (!stop_.load()) {
      if (!requestQueue_->tryPop(request)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      if (request.action == tt::domain::MemoryManagementAction::ALLOCATE) {
        tt::domain::ManageMemoryResult response{};
        response.taskId = request.taskId;
        response.status = tt::domain::ManageMemoryStatus::SUCCESS;
        response.slotId = nextSlot.fetch_add(1);
        resultQueue_->push(response);
      }
    }
  }

  std::unique_ptr<tt::ipc::boost::MemoryRequestQueue> requestQueue_;
  std::unique_ptr<tt::ipc::boost::MemoryResultQueue> resultQueue_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::atomic<uint32_t> nextSlot;
};

struct GetSlotTestContext {
  tt::services::SessionManager manager;
  MemoryAutoResponder memoryResponder;
  LoopFixture lf;
};

tt::services::SlotAcquireResult runGetSlot(
    tt::services::SessionManager& manager, trantor::EventLoop* loop,
    std::span<const uint32_t> tokens, tt::services::GetSlotOptions opts = {}) {
  std::promise<tt::services::SlotAcquireResult> promise;
  auto future = promise.get_future();
  manager.getSlot(
      tokens, std::move(opts), loop,
      [&promise](tt::services::SlotAcquireResult result) {
        promise.set_value(std::move(result));
      },
      [&promise](const std::string& err) {
        promise.set_exception(std::make_exception_ptr(std::runtime_error(err)));
      });
  return future.get();
}

tt::services::SlotAcquireResult runGetSlotWithBlocks(
    tt::services::SessionManager& manager, trantor::EventLoop* loop,
    std::vector<tt::utils::BlockHashInfo> blocks,
    tt::services::GetSlotOptions opts = {}) {
  opts.precomputedBlocks = std::move(blocks);
  return runGetSlot(manager, loop, {}, std::move(opts));
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
// Response-id continuation tests (via getSlot)
// ---------------------------------------------------------------------------

constexpr std::array<uint32_t, 5> kResponseIdTokens = {10, 11, 12, 13, 14};

TEST(SessionManagerGetSlot, ResponseId_RegisterThenAcquire_ReturnsSession) {
  GetSlotTestContext ctx;

  tt::services::GetSlotOptions turn1Opts;
  turn1Opts.responseId = "resp-1";
  auto turn1 =
      runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens, turn1Opts);
  EXPECT_TRUE(turn1.isNewSession);
  ctx.manager.getSession(turn1.sessionId)->release();

  tt::services::GetSlotOptions turn2Opts;
  turn2Opts.previousResponseId = "resp-1";
  auto turn2 =
      runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens, turn2Opts);
  ASSERT_FALSE(turn2.isNewSession);
  EXPECT_EQ(turn2.sessionId, turn1.sessionId);
  EXPECT_EQ(turn2.slotId, turn1.slotId);
  ctx.manager.getSession(turn2.sessionId)->release();
}

TEST(SessionManagerGetSlot, ResponseId_UnknownPreviousId_AllocatesNewSession) {
  GetSlotTestContext ctx;

  auto result = runGetSlot(
      ctx.manager, ctx.lf.loop, kResponseIdTokens,
      tt::services::GetSlotOptions{.previousResponseId = "no-such-id"});
  EXPECT_TRUE(result.isNewSession);
  ctx.manager.getSession(result.sessionId)->release();
}

TEST(SessionManagerGetSlot, ResponseId_EmptyPreviousId_AllocatesNewSession) {
  GetSlotTestContext ctx;

  auto result =
      runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                 tt::services::GetSlotOptions{.previousResponseId = ""});
  EXPECT_TRUE(result.isNewSession);
  ctx.manager.getSession(result.sessionId)->release();
}

TEST(SessionManagerGetSlot, ResponseId_EmptyResponseId_IsNoOp) {
  GetSlotTestContext ctx;

  auto result = runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                           tt::services::GetSlotOptions{.responseId = ""});
  auto session = ctx.manager.getSession(result.sessionId);
  ASSERT_TRUE(session);
  EXPECT_TRUE(session->getResponseId().empty());
  session->release();
}

TEST(SessionManagerGetSlot, ResponseId_ReKey_MovesSessionToNewId) {
  GetSlotTestContext ctx;

  auto turn1 = runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                          tt::services::GetSlotOptions{.responseId = "resp-1"});
  ctx.manager.getSession(turn1.sessionId)->release();

  auto turn2 =
      runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                 tt::services::GetSlotOptions{.previousResponseId = "resp-1",
                                              .responseId = "resp-2"});
  EXPECT_EQ(turn2.sessionId, turn1.sessionId);
  ctx.manager.getSession(turn2.sessionId)->release();

  auto miss =
      runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                 tt::services::GetSlotOptions{.previousResponseId = "resp-1"});
  EXPECT_TRUE(miss.isNewSession);
  EXPECT_NE(miss.sessionId, turn1.sessionId);
  ctx.manager.getSession(miss.sessionId)->release();

  auto hit =
      runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                 tt::services::GetSlotOptions{.previousResponseId = "resp-2"});
  EXPECT_FALSE(hit.isNewSession);
  EXPECT_EQ(hit.sessionId, turn1.sessionId);
  ctx.manager.getSession(hit.sessionId)->release();
}

TEST(SessionManagerGetSlot, ResponseId_SecondAcquireWhileInFlight_Throws) {
  GetSlotTestContext ctx;

  auto turn1 = runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                          tt::services::GetSlotOptions{.responseId = "resp-1"});

  EXPECT_THROW(
      runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                 tt::services::GetSlotOptions{.previousResponseId = "resp-1"}),
      tt::services::SessionInFlightException);

  ctx.manager.getSession(turn1.sessionId)->release();
}

TEST(SessionManagerGetSlot, ResponseId_AcquireAfterRelease_Succeeds) {
  GetSlotTestContext ctx;

  auto turn1 = runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                          tt::services::GetSlotOptions{.responseId = "resp-1"});
  ctx.manager.getSession(turn1.sessionId)->release();

  auto turn2 =
      runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                 tt::services::GetSlotOptions{.previousResponseId = "resp-1"});
  ASSERT_FALSE(turn2.isNewSession);
  EXPECT_EQ(turn2.sessionId, turn1.sessionId);
  ctx.manager.getSession(turn2.sessionId)->release();
}

TEST(SessionManagerGetSlot, ResponseId_CloseSession_RemovesFromIndex) {
  GetSlotTestContext ctx;

  auto turn1 = runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                          tt::services::GetSlotOptions{.responseId = "resp-1"});
  ctx.manager.getSession(turn1.sessionId)->release();
  ASSERT_EQ(ctx.manager.closeSession(turn1.sessionId),
            tt::services::CloseSessionResult::SUCCESS);

  auto turn2 =
      runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                 tt::services::GetSlotOptions{.previousResponseId = "resp-1"});
  EXPECT_TRUE(turn2.isNewSession);
  ctx.manager.getSession(turn2.sessionId)->release();
}

TEST(SessionManagerGetSlot, ResponseId_CloseWhileAcquired_FiresCancelFn) {
  GetSlotTestContext ctx;

  std::atomic<bool> cancelCalled{false};
  tt::services::GetSlotOptions opts;
  opts.responseId = "resp-1";
  opts.cancelFn = [&cancelCalled]() { cancelCalled = true; };
  auto turn1 = runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens, opts);

  ctx.manager.closeSession(turn1.sessionId);
  EXPECT_TRUE(cancelCalled.load());
  EXPECT_FALSE(ctx.manager.getSession(turn1.sessionId));
}

TEST(SessionManagerGetSlot, ResponseId_TwoTurnContinuation_ReKeysAcrossIds) {
  GetSlotTestContext ctx;

  auto turn1 = runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                          tt::services::GetSlotOptions{.responseId = "r1"});
  ctx.manager.getSession(turn1.sessionId)->release();

  auto turn2 = runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                          tt::services::GetSlotOptions{
                              .previousResponseId = "r1", .responseId = "r2"});
  EXPECT_EQ(turn2.sessionId, turn1.sessionId);
  ctx.manager.getSession(turn2.sessionId)->release();

  auto turn3 =
      runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                 tt::services::GetSlotOptions{.previousResponseId = "r2"});
  EXPECT_EQ(turn3.sessionId, turn1.sessionId);

  auto miss =
      runGetSlot(ctx.manager, ctx.lf.loop, kResponseIdTokens,
                 tt::services::GetSlotOptions{.previousResponseId = "r1"});
  EXPECT_TRUE(miss.isNewSession);
  ctx.manager.getSession(turn3.sessionId)->release();
  ctx.manager.getSession(miss.sessionId)->release();
}

TEST(SessionManagerGetSlot,
     ResponseId_PrefixCacheIndex_HitAndUpdated_ViaResponseIdPath) {
  GetSlotTestContext ctx;

  std::vector<tt::utils::BlockHashInfo> turn1Blocks = {
      {100, 0},
      {200, 0},
      {300, 0},
  };

  auto turn1 =
      runGetSlotWithBlocks(ctx.manager, ctx.lf.loop, turn1Blocks,
                           tt::services::GetSlotOptions{.responseId = "r1"});
  EXPECT_TRUE(turn1.isNewSession);
  ctx.manager.getSession(turn1.sessionId)->release();

  auto turn2 =
      runGetSlotWithBlocks(ctx.manager, ctx.lf.loop, turn1Blocks,
                           tt::services::GetSlotOptions{
                               .previousResponseId = "r1", .responseId = "r2"});
  ASSERT_FALSE(turn2.isNewSession);
  EXPECT_EQ(turn2.sessionId, turn1.sessionId);
  EXPECT_GT(turn2.matchedTokens, 0u);

  std::vector<tt::utils::BlockHashInfo> turn2Blocks = {
      {100, 0},
      {200, 0},
      {300, 0},
      {400, 0},
  };
  ASSERT_TRUE(turn2.registerPrefixBlocks);
  turn2.registerPrefixBlocks(turn2Blocks);
  ctx.manager.getSession(turn2.sessionId)->release();

  auto turn3 = runGetSlotWithBlocks(
      ctx.manager, ctx.lf.loop, turn2Blocks,
      tt::services::GetSlotOptions{.previousResponseId = "r2"});
  ASSERT_FALSE(turn3.isNewSession);
  EXPECT_EQ(turn3.sessionId, turn1.sessionId);
  EXPECT_GT(turn3.matchedTokens, turn2.matchedTokens);
  ctx.manager.getSession(turn3.sessionId)->release();

  auto prefixHit =
      runGetSlotWithBlocks(ctx.manager, ctx.lf.loop, turn1Blocks, {});
  EXPECT_FALSE(prefixHit.isNewSession);
  EXPECT_EQ(prefixHit.sessionId, turn1.sessionId);
  ctx.manager.getSession(prefixHit.sessionId)->release();
}

// ---------------------------------------------------------------------------
// clearSessionBlockThinkTokens tests
// ---------------------------------------------------------------------------

TEST(SessionManagerClearThinkTokens, ResetsThinkTokensToZero) {
  tt::services::SessionManager manager;
  LoopFixture lf;

  std::vector<tt::utils::BlockHashInfo> blocks = {
      {100, 5},
      {200, 12},
      {300, 20},
  };
  createSessionWithSlot(manager, lf.loop, 70u, blocks);

  auto before = runGetSlotWithBlocks(manager, lf.loop, blocks, {});
  EXPECT_GT(before.matchedTokens, 0u);
  EXPECT_EQ(before.accumulatedThinkTokens, 20u);
  manager.getSession(before.sessionId)->release();

  manager.clearSessionBlockThinkTokens(before.sessionId);

  auto after = runGetSlotWithBlocks(manager, lf.loop, blocks, {});
  EXPECT_EQ(after.matchedTokens, before.matchedTokens);
  EXPECT_EQ(after.accumulatedThinkTokens, 0u);
  manager.getSession(after.sessionId)->release();
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
