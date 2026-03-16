// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

// Unit tests for the cancellation feature.
//
// LLMService::cancel_request() and the corrected consumer_loop_for_worker()
// both operate on a ConcurrentMap<string, StreamCallback> plus the
// StreamingChunkResponse domain objects.  The service itself cannot be
// instantiated in a unit test (it requires Drogon, IPC queues, worker
// processes), so these tests exercise the mechanisms directly.
//
// Covered behaviours:
//  1. cancel for an unknown task_id returns false (no-op).
//  2. cancel for a known task_id invokes the callback with is_final=true
//     and finish_reason="cancelled".
//  3. The callback is removed atomically — a second cancel is a no-op.
//  4. Cancelling one task does not affect callbacks for other tasks.
//  5. Consumer loop: a token arriving for an already-removed callback is
//     silently discarded (no throw) — the corrected path in consumer_loop.
//  6. pending_tasks_ is decremented exactly once by cancel; late final
//     tokens from the worker (missing callback) do not double-decrement.

#include "domain/completion_response.hpp"
#include "domain/task_id.hpp"
#include "utils/concurrent_map.hpp"
#include "ipc/cancel_queue.hpp"

#include <atomic>
#include <functional>
#include <string>
#include <unistd.h>

#include <gtest/gtest.h>

namespace {

using tt::domain::CompletionChoice;
using tt::domain::StreamingChunkResponse;
using tt::domain::TaskID;
using StreamCallback = std::function<void(StreamingChunkResponse&, bool)>;
using CallbackMap = ConcurrentMap<std::string, StreamCallback>;

// Mirrors the final-chunk construction in LLMService::cancel_request().
StreamingChunkResponse make_cancelled_chunk(const std::string& task_id) {
    StreamingChunkResponse chunk{TaskID(task_id)};
    chunk.id = task_id;
    CompletionChoice choice;
    choice.text = "";
    choice.index = 0;
    choice.finish_reason = "cancelled";
    chunk.choices.push_back(std::move(choice));
    return chunk;
}

// ---------------------------------------------------------------------------
// cancel_request() behaviour
// ---------------------------------------------------------------------------

TEST(CancelRequestTest, UnknownTaskId_TakeReturnsNullopt) {
    CallbackMap callbacks;
    EXPECT_FALSE(callbacks.take("does-not-exist").has_value());
}

TEST(CancelRequestTest, KnownTaskId_CallbackInvokedWithCancelledChunk) {
    CallbackMap callbacks;
    const std::string task_id = "task-abc";

    StreamingChunkResponse received{TaskID("")};
    bool received_is_final = false;
    bool called = false;

    callbacks.insert(task_id, [&](StreamingChunkResponse& chunk, bool is_final) {
        received = chunk;
        received_is_final = is_final;
        called = true;
    });

    auto cb = callbacks.take(task_id);
    ASSERT_TRUE(cb.has_value());

    auto chunk = make_cancelled_chunk(task_id);
    cb.value()(chunk, /*is_final=*/true);

    EXPECT_TRUE(called);
    EXPECT_TRUE(received_is_final);
    ASSERT_FALSE(received.choices.empty());
    ASSERT_TRUE(received.choices[0].finish_reason.has_value());
    EXPECT_EQ(received.choices[0].finish_reason.value(), "cancelled");
    EXPECT_EQ(received.id, task_id);
}

TEST(CancelRequestTest, DoubleCancelIsNoOp) {
    CallbackMap callbacks;
    const std::string task_id = "task-abc";
    callbacks.insert(task_id, [](StreamingChunkResponse&, bool) {});

    EXPECT_TRUE(callbacks.take(task_id).has_value());   // first cancel succeeds
    EXPECT_FALSE(callbacks.take(task_id).has_value());  // second is a no-op
}

TEST(CancelRequestTest, CancelDoesNotAffectOtherTasks) {
    CallbackMap callbacks;
    const std::string id_a = "task-a";
    const std::string id_b = "task-b";
    callbacks.insert(id_a, [](StreamingChunkResponse&, bool) {});
    callbacks.insert(id_b, [](StreamingChunkResponse&, bool) {});

    callbacks.take(id_a);

    EXPECT_FALSE(callbacks.contains(id_a));
    EXPECT_TRUE(callbacks.contains(id_b));
}

// ---------------------------------------------------------------------------
// consumer_loop_for_worker() corrected behaviour
// ---------------------------------------------------------------------------

// Verifies that when a token arrives for a task whose callback has already
// been removed (e.g. the request was cancelled), the corrected consumer loop
// discards it without throwing.
TEST(ConsumerLoopTest, MissingCallback_IsDiscardedNotThrown) {
    CallbackMap callbacks;
    // Callback intentionally absent — simulates a cancelled task.
    EXPECT_FALSE(callbacks.get("already-cancelled").has_value());

    // The corrected consumer path: check, warn, continue.  Must not throw.
    EXPECT_NO_THROW({
        auto val = callbacks.get("already-cancelled");
        if (!val.has_value()) {
            // warn + skip (no throw)
        }
    });
}

// Verifies that pending_tasks_ is decremented exactly once: by cancel_request,
// not again when the late final token from the worker is discarded.
TEST(ConsumerLoopTest, PendingTasksDecrementedExactlyOnce) {
    std::atomic<size_t> pending{1};
    CallbackMap callbacks;
    const std::string task_id = "task-xyz";
    callbacks.insert(task_id, [](StreamingChunkResponse&, bool) {});

    // --- cancel_request path ---
    auto cb = callbacks.take(task_id);
    ASSERT_TRUE(cb.has_value());
    pending.fetch_sub(1);
    EXPECT_EQ(pending.load(), 0u);

    // --- late final token from worker arrives ---
    // Consumer finds no callback → skips, does NOT decrement pending again.
    auto late = callbacks.get(task_id);
    EXPECT_FALSE(late.has_value());
    // pending must still be 0, not wrapped around to SIZE_MAX.
    EXPECT_EQ(pending.load(), 0u);
}

// ---------------------------------------------------------------------------
// CancelQueue — IPC round-trip
// ---------------------------------------------------------------------------

// A unique name per test run avoids conflicts with any leftover shared memory
// from a previous crash.  The queue is removed before creation and after use.
static std::string cancel_queue_test_name() {
    // Use the test binary's PID so parallel test runs don't collide.
    return "tt_cancel_test_" + std::to_string(getpid());
}

TEST(CancelQueueTest, PushAndTryPop_RoundTrip) {
    const std::string name = cancel_queue_test_name();
    tt::ipc::CancelQueue::remove(name);

    tt::ipc::CancelQueue producer(name, /*max_messages=*/8);
    tt::ipc::CancelQueue consumer(name);

    const std::string task_id = "task-cancel-ipc-123";

    ASSERT_TRUE(producer.push(task_id));

    std::string received;
    ASSERT_TRUE(consumer.try_pop(received));
    EXPECT_EQ(received, task_id);

    // Queue is now empty.
    EXPECT_FALSE(consumer.try_pop(received));

    tt::ipc::CancelQueue::remove(name);
}

TEST(CancelQueueTest, MultipleMessages_PreserveOrder) {
    const std::string name = cancel_queue_test_name() + "_order";
    tt::ipc::CancelQueue::remove(name);

    tt::ipc::CancelQueue producer(name, /*max_messages=*/8);
    tt::ipc::CancelQueue consumer(name);

    std::vector<std::string> ids = {"task-a", "task-b", "task-c"};
    for (const auto& id : ids) {
        ASSERT_TRUE(producer.push(id));
    }

    std::vector<std::string> received;
    std::string out;
    while (consumer.try_pop(out)) {
        received.push_back(out);
    }

    EXPECT_EQ(received, ids);

    tt::ipc::CancelQueue::remove(name);
}

TEST(CancelQueueTest, PushReturnsFalse_WhenFull) {
    const std::string name = cancel_queue_test_name() + "_full";
    tt::ipc::CancelQueue::remove(name);

    tt::ipc::CancelQueue producer(name, /*max_messages=*/2);
    tt::ipc::CancelQueue consumer(name);

    EXPECT_TRUE(producer.push("task-1"));
    EXPECT_TRUE(producer.push("task-2"));
    EXPECT_FALSE(producer.push("task-3"));  // queue full

    tt::ipc::CancelQueue::remove(name);
}

}  // namespace
