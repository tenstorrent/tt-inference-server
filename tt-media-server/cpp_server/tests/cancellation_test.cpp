// SPDX-License-Identifier: Apache-2.0
#include "utils/id_generator.hpp"
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/sequence.hpp"
#include "ipc/boost_ipc_result_queue.hpp"
#include "ipc/cancel_queue.hpp"
#include "runners/llm_runner.hpp"
#include "runners/llm_runner/in_memory_task_queue.hpp"
#include "runners/llm_runner/schedulers/prefill_first_scheduler.hpp"

namespace tt::runners::schedulers {

using Config = tt::config::LLMConfig;
using tt::runners::llm_engine::InMemoryTaskQueue;

namespace {

std::shared_ptr<tt::ipc::ITaskQueue> makeQueue() {
  return std::make_shared<InMemoryTaskQueue>();
}

Config makeConfig(int numBlocks = 128, int blockSize = 8,
                  int maxBatchedTokens = 256, int eos = 0,
                  std::vector<int64_t> stopTokenIds = {}) {
  Config c;
  c.num_kvcache_blocks = numBlocks;
  c.kvcache_block_size = blockSize;
  c.max_num_batched_tokens = maxBatchedTokens;
  c.eos = eos;
  c.stop_token_ids = std::move(stopTokenIds);
  return c;
}

std::vector<int64_t> prompt(size_t len) {
  std::vector<int64_t> p;
  for (size_t i = 0; i < len; ++i) p.push_back(static_cast<int64_t>(i));
  return p;
}

uint32_t nextId() { return tt::utils::TaskIDGenerator::generate(); }

// ---------- In-memory cancel queue for testing ----------

class InMemoryCancelQueue : public tt::ipc::ICancelQueue {
 public:
  void push(uint32_t taskId) override { items.push_back(taskId); }

  void tryPopAll(std::vector<uint32_t>& out) override {
    out.insert(out.end(), items.begin(), items.end());
    items.clear();
  }

  void remove() override { items.clear(); }

  std::vector<uint32_t> items;
};

// ==========================================================
//  Scheduler abort tests
// ==========================================================

class SchedulerAbortTest : public ::testing::Test {
 protected:
  Config config = makeConfig(/*numBlocks=*/32, /*blockSize=*/8);
  std::shared_ptr<tt::ipc::ITaskQueue> queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), /*maxInFlightCount=*/4};
};

TEST_F(SchedulerAbortTest, AbortWaitingSequence) {
  uint32_t id = nextId();
  sched.addRequest(id, prompt(4), {.max_tokens = 10});

  // Schedule to get the sequence into sequences_ map
  auto [seqs, isPrefill] = sched.schedule();
  ASSERT_EQ(seqs.size(), 1u);
  ASSERT_TRUE(isPrefill);

  // Post-process one token to move to decode queue
  sched.postprocess(seqs, {100});

  // Now abort
  sched.abortRequest(id);

  // Sequence should be gone
  EXPECT_EQ(sched.findSequence(id), nullptr);
  EXPECT_TRUE(sched.isFinished());
}

TEST_F(SchedulerAbortTest, AbortDecodingSequence_FreesBlocks) {
  // Use small block count to verify blocks are freed
  Config smallConfig = makeConfig(/*numBlocks=*/4, /*blockSize=*/8);
  std::shared_ptr<tt::ipc::ITaskQueue> q = makeQueue();
  PrefillFirstScheduler s{smallConfig, q.get(), 4};

  size_t freeBlocksBefore = s.getBlockManager().numFreeBlocks();

  uint32_t id = nextId();
  s.addRequest(id, prompt(4), {.max_tokens = 100});

  auto [seqs, isPrefill] = s.schedule();
  ASSERT_EQ(seqs.size(), 1u);
  s.postprocess(seqs, {100});

  // Blocks are allocated
  EXPECT_LT(s.getBlockManager().numFreeBlocks(), freeBlocksBefore);

  // Abort frees blocks
  s.abortRequest(id);

  // All blocks should be free again
  EXPECT_EQ(s.getBlockManager().numFreeBlocks(), freeBlocksBefore);
}

TEST_F(SchedulerAbortTest, AbortIsIdempotent) {
  uint32_t id = nextId();
  sched.addRequest(id, prompt(4), {.max_tokens = 10});

  auto [seqs, isPrefill] = sched.schedule();
  sched.postprocess(seqs, {100});

  // Abort twice — second call should be a no-op
  sched.abortRequest(id);
  EXPECT_NO_FATAL_FAILURE(sched.abortRequest(id));
  EXPECT_EQ(sched.findSequence(id), nullptr);
}

TEST_F(SchedulerAbortTest, AbortUnknownTaskId_IsNoOp) {
  uint32_t unknownId = 99999;  // Use a number that's unlikely to exist
  EXPECT_NO_FATAL_FAILURE(sched.abortRequest(unknownId));
}

TEST_F(SchedulerAbortTest, AbortFinishedSequence_IsNoOp) {
  uint32_t id = nextId();
  // Use stop_token_ids={100} so token 100 triggers finish
  Config cfgWithStop = makeConfig(32, 8, 256, 0, {100});
  std::shared_ptr<tt::ipc::ITaskQueue> q = makeQueue();
  PrefillFirstScheduler s{cfgWithStop, q.get(), 4};

  s.addRequest(id, prompt(4), {.max_tokens = 10});

  auto [seqs, isPrefill] = s.schedule();
  ASSERT_EQ(seqs.size(), 1u);

  // Post-process with the stop token — sequence is marked FINISHED
  s.postprocess(seqs, {100});
  auto* seq = s.findSequence(id);
  ASSERT_NE(seq, nullptr);
  EXPECT_TRUE(seq->isFinished());

  // Abort after finish — should be a no-op (sequence stays finished)
  EXPECT_NO_FATAL_FAILURE(s.abortRequest(id));
  // Clean up like the runner would
  s.removeSequence(id);
  EXPECT_EQ(s.findSequence(id), nullptr);
}

TEST_F(SchedulerAbortTest, AbortBeforePrefillDequeue_SkipsStale) {
  // Add a request which goes into sequences_ AND the prefill queue
  uint32_t id = nextId();
  sched.addRequest(id, prompt(4), {.max_tokens = 10});

  // Abort before scheduling pops it from the queue
  sched.abortRequest(id);

  // Now schedule — should skip the stale copy
  auto [seqs, isPrefill] = sched.schedule();
  EXPECT_TRUE(seqs.empty());
  EXPECT_EQ(sched.findSequence(id), nullptr);
}

TEST_F(SchedulerAbortTest, AbortEnablesNewRequests) {
  // Use exactly 1 block with block_size=8. A prompt of 8 tokens needs 1 block.
  // With only 1 block total, only one request can be scheduled at a time.
  Config tightConfig = makeConfig(/*numBlocks=*/1, /*blockSize=*/8);
  std::shared_ptr<tt::ipc::ITaskQueue> q = makeQueue();
  PrefillFirstScheduler s{tightConfig, q.get(), 4};

  uint32_t id1 = nextId();
  s.addRequest(id1, prompt(8), {.max_tokens = 10});

  auto [seqs1, isPrefill1] = s.schedule();
  ASSERT_EQ(seqs1.size(), 1u);
  s.postprocess(seqs1, {100});

  // Record free blocks — should be 0 since id1 is decoding
  EXPECT_EQ(s.getBlockManager().numFreeBlocks(), 0);

  // Abort id1 — frees its block
  s.abortRequest(id1);
  EXPECT_EQ(s.getBlockManager().numFreeBlocks(), 1);

  // Now a new request can be scheduled
  uint32_t id2 = nextId();
  s.addRequest(id2, prompt(8), {.max_tokens = 10});

  auto [seqs2, isPrefill2] = s.schedule();
  ASSERT_EQ(seqs2.size(), 1u);
  EXPECT_EQ(seqs2[0]->taskId, id2);
}

// ==========================================================
//  LLMRunner cancel queue integration tests
// ==========================================================

TEST(LLMRunnerCancelTest, CancelledRequestStopsEmittingTokens) {
  setenv("LLM_MODE", "prefill", 1);
  Config config = makeConfig(128, 8, 256, 0);

  auto taskQueue = makeQueue();
  InMemoryCancelQueue cancelQueue;

  std::string rbName = "test_cancel_rb_" + std::to_string(getpid()) + "_stop";
  tt::ipc::BoostIpcResultQueue resultQueue(rbName,
                                           tt::ipc::RESULT_QUEUE_CAPACITY);

  tt::runners::LLMRunner runner{config, &resultQueue, taskQueue.get(),
                                &cancelQueue};

  uint32_t cancelId = nextId();
  uint32_t keepId = nextId();

  runner.getScheduler().addRequest(cancelId, prompt(4), {.max_tokens = 100});
  runner.getScheduler().addRequest(keepId, prompt(4), {.max_tokens = 5});

  std::unordered_map<uint32_t, int> tokenCounts;
  std::atomic<bool> keepFinished{false};
  bool cancelledAfterFirstToken = false;

  std::thread consumer([&]() {
    tt::ipc::SharedToken token;
    while (!keepFinished.load()) {
      if (resultQueue.tryPop(token)) {
        tokenCounts[token.task_id]++;

        // After first token from cancelId, push cancel signal
        if (token.task_id == cancelId && !cancelledAfterFirstToken) {
          cancelledAfterFirstToken = true;
          cancelQueue.push(cancelId);
        }

        if (token.task_id == keepId && token.isFinal()) {
          keepFinished.store(true);
        }
      }
    }
    runner.stop();
  });

  runner.start();
  consumer.join();

  // The kept request should have exactly 5 tokens
  EXPECT_EQ(tokenCounts[keepId], 5);

  // The cancelled request should have significantly fewer than 100 tokens.
  // It may get a few tokens before the cancel is processed.
  EXPECT_LT(tokenCounts[cancelId], 50);

  resultQueue.shutdown();
  resultQueue.remove();
}

TEST(LLMRunnerCancelTest, CancelBeforeAnyProcessing) {
  setenv("LLM_MODE", "prefill", 1);
  Config config = makeConfig(128, 8, 256, 0);

  auto taskQueue = makeQueue();
  InMemoryCancelQueue cancelQueue;

  std::string rbName = "test_cancel_rb_" + std::to_string(getpid()) + "_before";
  tt::ipc::BoostIpcResultQueue resultQueue(rbName,
                                           tt::ipc::RESULT_QUEUE_CAPACITY);

  tt::runners::LLMRunner runner{config, &resultQueue, taskQueue.get(),
                                &cancelQueue};

  uint32_t cancelId = nextId();
  uint32_t keepId = nextId();

  runner.getScheduler().addRequest(cancelId, prompt(4), {.max_tokens = 10});
  runner.getScheduler().addRequest(keepId, prompt(4), {.max_tokens = 5});

  cancelQueue.push(cancelId);

  std::unordered_map<uint32_t, int> tokenCounts;
  std::atomic<bool> keepFinished{false};

  std::thread consumer([&]() {
    tt::ipc::SharedToken token;
    while (!keepFinished.load()) {
      if (resultQueue.tryPop(token)) {
        tokenCounts[token.task_id]++;
        if (token.task_id == keepId && token.isFinal()) {
          keepFinished.store(true);
        }
      }
    }
    runner.stop();
  });

  runner.start();
  consumer.join();

  // keepId should complete normally
  EXPECT_EQ(tokenCounts[keepId], 5);

  // cancelId should have zero tokens (cancelled before scheduling)
  EXPECT_EQ(tokenCounts[cancelId], 0);

  resultQueue.shutdown();
  resultQueue.remove();
}

}  // namespace
}  // namespace tt::runners::schedulers
