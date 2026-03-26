// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// Unit and integration tests for request cancellation:
//   1. Scheduler::abortRequest() — removes sequences, frees blocks, idempotent
//   2. LLMRunner cancel queue — abort stops token emission for that request
//   3. BoostIpcCancelQueue — IPC push/drain round-trip

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "config/runner_config.hpp"
#include "ipc/boost_ipc_cancel_queue.hpp"
#include "ipc/shared_memory.hpp"
#include "runners/llm_runner.hpp"
#include "runners/llm_runner/in_memory_task_queue.hpp"
#include "runners/llm_runner/prefill_first_scheduler.hpp"
#include "runners/llm_runner/sampling_params.hpp"
#include "runners/llm_runner/scheduler.hpp"
#include "runners/llm_runner/sequence.hpp"

namespace llm_engine {

using Config = tt::config::LLMConfig;

namespace {

// ──────────────────────────────────────────────────────────────────────────────
// Helpers shared across test suites
// ──────────────────────────────────────────────────────────────────────────────

std::shared_ptr<ITaskQueue> makeQueue() {
  return std::make_shared<InMemoryTaskQueue>();
}

Config makeConfig(int numBlocks = 32, int blockSize = 8,
                  int maxBatchedTokens = 512) {
  Config c;
  c.num_kvcache_blocks = numBlocks;
  c.kvcache_block_size = blockSize;
  c.max_num_batched_tokens = maxBatchedTokens;
  c.eos = 0;
  return c;
}

std::vector<int64_t> prompt(size_t len) {
  std::vector<int64_t> p;
  for (size_t i = 0; i < len; ++i) p.push_back(static_cast<int64_t>(i + 1));
  return p;
}

TaskID nextId() { return TaskID(TaskID::generate()); }

// Advance a prefill batch and return the batch (mutates the scheduler decode
// queue). Returns the scheduled batch or empty if nothing to schedule.
std::vector<Sequence*> runPrefill(Scheduler& sched) {
  auto [batch, is_prefill] = sched.schedule();
  if (!is_prefill || batch.empty()) return {};
  std::vector<int64_t> tokens(batch.size(), 1);
  sched.postprocess(batch, tokens);
  return batch;
}

// ──────────────────────────────────────────────────────────────────────────────
// Suite 1: Scheduler::abortRequest()
// ──────────────────────────────────────────────────────────────────────────────

// Abort a sequence that has not yet been scheduled (still in prefill queue).
// The scheduler should report it as gone and skip it during the next schedule.
TEST(SchedulerAbortTest, AbortWaiting_SequenceBecomesUnfindable) {
  Config cfg = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{cfg, queue.get(), 4};

  TaskID id = nextId();
  sched.addRequest(id, prompt(4), {.max_tokens = 10});

  ASSERT_NE(sched.findSequence(id), nullptr);
  sched.abortRequest(id);
  EXPECT_EQ(sched.findSequence(id), nullptr);
}

// Abort a sequence that is in the decode queue (post-prefill, running).
TEST(SchedulerAbortTest, AbortDecoding_SequenceBecomesUnfindable) {
  Config cfg = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{cfg, queue.get(), 4};

  TaskID id = nextId();
  sched.addRequest(id, prompt(4), {.max_tokens = 10});
  runPrefill(sched);  // moves sequence to decode queue

  ASSERT_NE(sched.findSequence(id), nullptr);
  sched.abortRequest(id);
  EXPECT_EQ(sched.findSequence(id), nullptr);
}

// After aborting a sequence, the scheduler is finished (no more work).
TEST(SchedulerAbortTest, AbortDecoding_SchedulerReportsFinished) {
  Config cfg = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{cfg, queue.get(), 4};

  TaskID id = nextId();
  sched.addRequest(id, prompt(4), {.max_tokens = 10});
  runPrefill(sched);

  sched.abortRequest(id);
  EXPECT_TRUE(sched.isFinished());
}

// Aborting one request must free KV blocks so a previously-blocked request
// can be scheduled.
TEST(SchedulerAbortTest, AbortFreesBlocks_AllowsNewRequest) {
  // 1 block of size 8: there is room for exactly one prompt's KV allocation.
  // seq1 occupies that block after prefill.
  // seq2 then cannot be prefilled (no free blocks).
  // Aborting seq1 frees the block so seq2 can prefill.
  Config cfg = makeConfig(/*numBlocks=*/1, /*blockSize=*/8);
  auto queue = makeQueue();
  PrefillFirstScheduler sched{cfg, queue.get(), 4};

  // seq1: 4 prompt tokens → 1 block needed.
  TaskID id1 = nextId();
  sched.addRequest(id1, prompt(4), {.max_tokens = 5});

  // Prefill seq1: it occupies the single block.
  auto [batch1, is_prefill1] = sched.schedule();
  ASSERT_TRUE(is_prefill1);
  ASSERT_EQ(batch1.size(), 1u);
  sched.postprocess(batch1,
                    {1});  // seq1 → decode queue, 5 tokens, still in block

  // Now add seq2. seq2 also needs 1 block but there are none free.
  TaskID id2 = nextId();
  sched.addRequest(id2, prompt(4), {.max_tokens = 5});

  // schedule() falls through to decode seq1 because seq2 cannot be prefilled.
  auto [batch_decode, is_prefill_decode] = sched.schedule();
  EXPECT_FALSE(is_prefill_decode) << "seq2 cannot prefill; decode seq1 instead";
  // seq2 must not be in the scheduled batch (it's still stuck in prefill
  // queue).
  bool seq2InBatch =
      std::any_of(batch_decode.begin(), batch_decode.end(),
                  [&](const Sequence* s) { return s->taskId == id2; });
  EXPECT_FALSE(seq2InBatch) << "seq2 must not appear in decode batch";

  // Abort seq1: releases the block.
  sched.abortRequest(id1);

  // seq2 must now be prefillable.
  auto [batch_after, is_prefill_after] = sched.schedule();
  EXPECT_TRUE(is_prefill_after) << "seq2 should now get a prefill batch";
  ASSERT_EQ(batch_after.size(), 1u);
  EXPECT_EQ(batch_after[0]->taskId, id2);
}

// Calling abortRequest twice for the same ID must not crash or corrupt state.
TEST(SchedulerAbortTest, AbortIsIdempotent) {
  Config cfg = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{cfg, queue.get(), 4};

  TaskID id = nextId();
  sched.addRequest(id, prompt(4), {.max_tokens = 10});

  EXPECT_NO_FATAL_FAILURE(sched.abortRequest(id));
  EXPECT_NO_FATAL_FAILURE(sched.abortRequest(id));
  EXPECT_EQ(sched.findSequence(id), nullptr);
}

// Calling abortRequest with an unknown task ID must be a silent no-op.
TEST(SchedulerAbortTest, AbortUnknownId_IsNoop) {
  Config cfg = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{cfg, queue.get(), 4};

  TaskID unknown = nextId();
  EXPECT_NO_FATAL_FAILURE(sched.abortRequest(unknown));
  EXPECT_TRUE(sched.isFinished());
}

// A sequence aborted while still in the prefill queue must be silently skipped
// (not scheduled) on the next schedule() call.
TEST(SchedulerAbortTest, AbortedInPrefillQueue_SkippedDuringSchedule) {
  Config cfg = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{cfg, queue.get(), 4};

  // Add two requests: abort the first one before it is ever scheduled.
  TaskID id1 = nextId();
  TaskID id2 = nextId();
  sched.addRequest(id1, prompt(4), {.max_tokens = 10});
  sched.addRequest(id2, prompt(4), {.max_tokens = 10});

  sched.abortRequest(id1);

  // The first scheduled batch must only contain id2.
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(batch.size(), 1u);
  EXPECT_EQ(batch[0]->taskId, id2);
}

// ──────────────────────────────────────────────────────────────────────────────
// Suite 2: LLMRunner + cancel queue
// ──────────────────────────────────────────────────────────────────────────────

Config makeEngineConfig() {
  Config c;
  c.num_kvcache_blocks = 128;
  c.kvcache_block_size = 8;
  c.max_num_batched_tokens = 512;
  c.eos = 0;
  return c;
}

// Abort a request before the engine starts. The aborted request must emit no
// tokens at all; the other request must complete normally.
TEST(LLMRunnerCancelTest, AbortBeforeStart_EmitsNoTokens) {
  Config config = makeEngineConfig();
  auto taskQueue = makeQueue();
  std::string resultName = "/test_cancel_preabort_" + std::to_string(getpid());
  tt::ipc::TokenRingBuffer<65536> resultQueue(resultName, true);

  tt::runners::LLMRunner engine{config, &resultQueue, taskQueue.get()};

  TaskID abortedId = TaskID(TaskID::generate());
  TaskID normalId = TaskID(TaskID::generate());

  engine.scheduler().addRequest(abortedId, prompt(4), {.max_tokens = 20});
  engine.scheduler().addRequest(normalId, prompt(4), {.max_tokens = 5});

  // Abort before start: no tokens should be generated for abortedId.
  engine.scheduler().abortRequest(abortedId);

  std::unordered_map<std::string, std::vector<int64_t>> received;
  std::unordered_map<std::string, bool> finalSeen;
  std::atomic<int> doneCount{0};

  std::thread consumer([&]() {
    tt::ipc::SharedToken token;
    while (doneCount.load() < 1) {
      if (resultQueue.pop(token)) {
        std::string tid(token.task_id);
        received[tid].push_back(static_cast<int64_t>(token.token_id));
        if (token.isFinal()) {
          finalSeen[tid] = true;
          doneCount.fetch_add(1);
        }
      } else {
        std::this_thread::yield();
      }
    }
    engine.stop();
  });

  engine.start();
  consumer.join();

  // abortedId: must have produced zero tokens.
  EXPECT_EQ(received.count(abortedId.id), 0u)
      << "aborted request should emit no tokens";

  // normalId: must have completed with FLAG_FINAL.
  ASSERT_TRUE(finalSeen.count(normalId.id) && finalSeen[normalId.id])
      << "normal request should complete";
  EXPECT_EQ(received[normalId.id].size(), 5u)
      << "normal request should produce exactly max_tokens tokens";

  resultQueue.shutdown();
}

// Abort a request mid-stream via the IPC cancel queue. The aborted request
// must stop emitting tokens after the abort is delivered; the other request
// must complete normally.
TEST(LLMRunnerCancelTest, AbortViaCancelQueue_StopsEmitting) {
  std::string cancelQName = "tt_cancel_runner_test_" + std::to_string(getpid());
  tt::ipc::BoostIpcCancelQueue::removeByName(cancelQName);

  Config config = makeEngineConfig();
  auto taskQueue = makeQueue();
  std::string resultName = "/test_cancel_midstream_" + std::to_string(getpid());
  tt::ipc::TokenRingBuffer<65536> resultQueue(resultName, true);

  // Create the cancel queue (simulates the main-process side).
  auto cancelQueue = std::make_shared<tt::ipc::BoostIpcCancelQueue>(
      cancelQName, /*capacity=*/64);

  tt::runners::LLMRunner engine{config, &resultQueue, taskQueue.get(),
                                cancelQueue.get()};

  TaskID abortedId = TaskID(TaskID::generate());
  TaskID normalId = TaskID(TaskID::generate());

  // normalId is short (5 tokens). abortedId is long (50 tokens).
  engine.scheduler().addRequest(normalId, prompt(4), {.max_tokens = 5});
  engine.scheduler().addRequest(abortedId, prompt(4), {.max_tokens = 50});

  std::unordered_map<std::string, std::vector<int64_t>> received;
  std::unordered_map<std::string, bool> finalSeen;
  std::atomic<bool> abortPushed{false};
  std::atomic<int> doneCount{0};

  std::thread consumer([&]() {
    tt::ipc::SharedToken token;
    while (doneCount.load() < 1) {
      if (resultQueue.pop(token)) {
        std::string tid(token.task_id);
        received[tid].push_back(static_cast<int64_t>(token.token_id));
        if (token.isFinal()) {
          finalSeen[tid] = true;
          doneCount.fetch_add(1);

          // Once normalId finishes, push cancel for abortedId.
          if (tid == normalId.id && !abortPushed.exchange(true)) {
            cancelQueue->push(abortedId);
            // Give the engine a moment to process the cancel, then stop.
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            engine.stop();
          }
        }
      } else {
        std::this_thread::yield();
      }
    }
  });

  engine.start();
  consumer.join();

  // normalId must have completed.
  ASSERT_TRUE(finalSeen.count(normalId.id) && finalSeen[normalId.id])
      << "normal request should complete with FLAG_FINAL";
  EXPECT_EQ(received[normalId.id].size(), 5u)
      << "normal request must produce exactly max_tokens tokens";

  // abortedId must never have received FLAG_FINAL.
  EXPECT_FALSE(finalSeen.count(abortedId.id) && finalSeen[abortedId.id])
      << "aborted request must not receive FLAG_FINAL";

  // abortedId must have received fewer tokens than its max_tokens.
  if (received.count(abortedId.id)) {
    EXPECT_LT(received[abortedId.id].size(), 50u)
        << "aborted request must not run to completion";
  }

  resultQueue.shutdown();
  tt::ipc::BoostIpcCancelQueue::removeByName(cancelQName);
}

// ──────────────────────────────────────────────────────────────────────────────
// Suite 3: BoostIpcCancelQueue — IPC push/drain round-trip
// ──────────────────────────────────────────────────────────────────────────────

class CancelQueueTest : public ::testing::Test {
 protected:
  static constexpr const char* Q_NAME = "tt_cancel_queue_unit_test";

  void SetUp() override { tt::ipc::BoostIpcCancelQueue::removeByName(Q_NAME); }
  void TearDown() override {
    tt::ipc::BoostIpcCancelQueue::removeByName(Q_NAME);
  }
};

// Pushing a TaskID and immediately draining it returns the same ID.
TEST_F(CancelQueueTest, PushAndDrain_RoundTrip) {
  tt::ipc::BoostIpcCancelQueue q{Q_NAME, 64};

  tt::domain::TaskID id(tt::domain::TaskID::generate());
  q.push(id);

  std::vector<tt::domain::TaskID> out;
  q.tryPopAll(out);

  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, id.id);
}

// Draining an empty queue returns an empty vector.
TEST_F(CancelQueueTest, DrainEmpty_ReturnsEmpty) {
  tt::ipc::BoostIpcCancelQueue q{Q_NAME, 64};

  std::vector<tt::domain::TaskID> out;
  q.tryPopAll(out);

  EXPECT_TRUE(out.empty());
}

// Pushing multiple IDs and draining once returns all of them.
TEST_F(CancelQueueTest, MultipleMessages_AllDrained) {
  tt::ipc::BoostIpcCancelQueue q{Q_NAME, 64};

  std::vector<std::string> pushed;
  for (int i = 0; i < 5; ++i) {
    tt::domain::TaskID id(tt::domain::TaskID::generate());
    pushed.push_back(id.id);
    q.push(id);
  }

  std::vector<tt::domain::TaskID> out;
  q.tryPopAll(out);

  ASSERT_EQ(out.size(), 5u);
  std::vector<std::string> received;
  for (const auto& tid : out) received.push_back(tid.id);
  EXPECT_EQ(received, pushed);
}

// Pushing beyond capacity does not block or crash; excess messages are dropped.
TEST_F(CancelQueueTest, PushWhenFull_DropsGracefully) {
  tt::ipc::BoostIpcCancelQueue q{Q_NAME, /*capacity=*/2};

  tt::domain::TaskID id1(tt::domain::TaskID::generate());
  tt::domain::TaskID id2(tt::domain::TaskID::generate());
  tt::domain::TaskID id3(tt::domain::TaskID::generate());  // will be dropped

  EXPECT_NO_FATAL_FAILURE(q.push(id1));
  EXPECT_NO_FATAL_FAILURE(q.push(id2));
  EXPECT_NO_FATAL_FAILURE(q.push(id3));  // queue is full — should drop silently

  std::vector<tt::domain::TaskID> out;
  q.tryPopAll(out);
  // Only the first two fit; the third was dropped.
  EXPECT_LE(out.size(), 2u);
}

// Draining twice: second drain returns nothing.
TEST_F(CancelQueueTest, DrainTwice_SecondDrainEmpty) {
  tt::ipc::BoostIpcCancelQueue q{Q_NAME, 64};

  tt::domain::TaskID id(tt::domain::TaskID::generate());
  q.push(id);

  std::vector<tt::domain::TaskID> first, second;
  q.tryPopAll(first);
  q.tryPopAll(second);

  EXPECT_EQ(first.size(), 1u);
  EXPECT_TRUE(second.empty());
}

}  // namespace
}  // namespace llm_engine
