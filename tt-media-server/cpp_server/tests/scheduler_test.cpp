// SPDX-License-Identifier: Apache-2.0
#include "utils/id_generator.hpp"
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <vector>

#include "config/runner_config.hpp"
#include "domain/sampling_params.hpp"
#include "domain/sequence.hpp"
#include "runners/llm_runner/in_memory_task_queue.hpp"
#include "runners/schedulers/max_occupancy_scheduler.hpp"
#include "runners/schedulers/prefill_first_scheduler.hpp"
#include "runners/schedulers/scheduler.hpp"

namespace tt::runners::schedulers {

using Config = tt::config::LLMConfig;
using tt::runners::llm_engine::InMemoryTaskQueue;
using SchedulingPolicy = tt::config::SchedulingPolicy;
using Sequence = tt::domain::Sequence;
using SamplingParams = tt::domain::SamplingParams;

namespace {

std::shared_ptr<tt::ipc::ITaskQueue> makeQueue() {
  return std::make_shared<InMemoryTaskQueue>();
}

Config makeConfig(int numBlocks = 32, int blockSize = 8,
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

// --- make_scheduler factory tests ---

TEST(MakeSchedulerTest, PrefillFirstPolicy_CreatesPrefillFirstScheduler) {
  Config config = makeConfig();
  config.scheduling_policy = SchedulingPolicy::PREFILL_FIRST;
  auto queue = makeQueue();
  auto sched = makeScheduler(config, queue.get(), 1);
  ASSERT_NE(sched, nullptr);
  EXPECT_NE(dynamic_cast<PrefillFirstScheduler*>(sched.get()), nullptr);
}

// --- PrefillFirstScheduler tests ---

TEST(PrefillFirstSchedulerTest, IsFinished_WhenEmpty_ReturnsTrue) {
  Config config = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  EXPECT_TRUE(sched.isFinished());
}

TEST(PrefillFirstSchedulerTest, IsFinished_AfterAdd_ReturnsFalse) {
  Config config = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  Sequence seq{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 10}};
  sched.add(seq);
  EXPECT_FALSE(sched.isFinished());
}

TEST(PrefillFirstSchedulerTest, Schedule_WithOneWaiting_ReturnsPrefillBatch) {
  Config config = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  Sequence seq{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 10}};
  uint32_t expectedId = seq.taskId;
  sched.add(seq);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(batch.size(), 1u);
  EXPECT_EQ(batch[0]->taskId, expectedId);
}

TEST(PrefillFirstSchedulerTest,
     Schedule_OneRequest_FirstCallPrefill_SecondCallEmpty) {
  Config config = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  sched.addRequest(nextId(), prompt(4), SamplingParams{.max_tokens = 10});

  auto [batch1, is_prefill1] = sched.schedule();
  ASSERT_TRUE(is_prefill1);
  EXPECT_EQ(batch1.size(), 1u);

  auto [batch2, is_prefill2] = sched.schedule();
  EXPECT_TRUE(batch2.empty());
}

TEST(PrefillFirstSchedulerTest,
     Schedule_WhenNoWaitingAndOneRunning_ReturnsDecodeBatch) {
  Config config = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  Sequence seq{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 10}};
  uint32_t expectedId = seq.taskId;
  sched.add(seq);
  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(prefill_batch.size(), 1u);
  std::vector<int64_t> tokens = {1};
  sched.postprocess(prefill_batch, tokens);

  auto [decode_batch, is_decode] = sched.schedule();
  ASSERT_FALSE(is_decode);
  ASSERT_EQ(decode_batch.size(), 1u);
  EXPECT_EQ(decode_batch[0]->taskId, expectedId);
}

TEST(PrefillFirstSchedulerTest, OneRequest_PrefillThenDecodeThenEos) {
  Config config = makeConfig(32, 8, 256, 99, std::vector<int64_t>{99});
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  sched.addRequest(nextId(), prompt(4),
                   {.max_tokens = 10, .ignore_eos = false});

  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(prefill_batch.size(), 1u);
  sched.postprocess(prefill_batch, {1});

  {
    auto [decode_batch, is_prefill] = sched.schedule();
    ASSERT_FALSE(is_prefill);
    ASSERT_EQ(decode_batch.size(), 1u);
    sched.postprocess(decode_batch, {99});
  }

  auto [final_batch, _] = sched.schedule();
  EXPECT_TRUE(final_batch.empty());
  EXPECT_TRUE(sched.isFinished());
}

TEST(PrefillFirstSchedulerTest, OneRequest_PrefillThenDecodeThenMaxTokens) {
  Config config = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  sched.addRequest(nextId(), prompt(4), {.max_tokens = 2});

  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(prefill_batch.size(), 1u);
  sched.postprocess(prefill_batch, {1});

  {
    auto [decode_batch, is_prefill] = sched.schedule();
    ASSERT_FALSE(is_prefill);
    ASSERT_EQ(decode_batch.size(), 1u);
    sched.postprocess(decode_batch, {2});
  }

  auto [final_batch, __] = sched.schedule();
  EXPECT_TRUE(final_batch.empty());
  EXPECT_TRUE(sched.isFinished());
}

TEST(PrefillFirstSchedulerTest,
     Postprocess_WhenTokenReachesMaxTokens_MarksFinished) {
  Config config = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  SamplingParams params;
  params.max_tokens = 2;
  Sequence seq{nextId(), 256, prompt(2), params};
  sched.add(seq);
  auto [batch1, _1] = sched.schedule();
  ASSERT_EQ(batch1.size(), 1u);
  sched.postprocess(batch1, {1});
  EXPECT_FALSE(batch1[0]->isFinished());
  auto [batch2, _2] = sched.schedule();
  ASSERT_EQ(batch2.size(), 1u);
  sched.postprocess(batch2, {2});
  EXPECT_TRUE(batch2[0]->isFinished());
}

TEST(PrefillFirstSchedulerTest, Postprocess_WhenEosToken_MarksFinished) {
  Config config = makeConfig(32, 8, 256, 99, std::vector<int64_t>{99});
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  Sequence seq{nextId(), 256, prompt(2),
               SamplingParams{.max_tokens = 100, .ignore_eos = false}};
  sched.add(seq);
  auto [batch, _] = sched.schedule();
  ASSERT_EQ(batch.size(), 1u);
  sched.postprocess(batch, {99});
  EXPECT_TRUE(batch[0]->isFinished());
}

TEST(PrefillFirstSchedulerTest, Preempt_MovesSequenceBackToWaiting) {
  Config config = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  Sequence seq{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 10}};
  uint32_t expectedId = seq.taskId;
  sched.add(seq);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(batch.size(), 1u);
  sched.preempt(*batch[0]);
  EXPECT_EQ(batch[0]->getStatus(), tt::domain::SequenceStatus::WAITING);
  auto [batch2, is_prefill2] = sched.schedule();
  EXPECT_TRUE(is_prefill2);
  EXPECT_EQ(batch2.size(), 1u);
  EXPECT_EQ(batch2[0]->taskId, expectedId);
}

TEST(PrefillFirstSchedulerTest, Schedule_PrefillPrioritizedOverDecode) {
  Config config = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  Sequence seq1{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 10}};
  Sequence seq2{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 10}};
  uint32_t seq2TaskId = seq2.taskId;
  sched.add(seq1);
  auto [batch1, prefill1] = sched.schedule();
  ASSERT_TRUE(prefill1);
  sched.postprocess(batch1, {1});
  sched.add(seq2);
  auto [batch2, prefill2] = sched.schedule();
  EXPECT_TRUE(prefill2) << "Prefill (seq2) should be chosen over decode (seq1)";
  ASSERT_EQ(batch2.size(), 1u);
  EXPECT_EQ(batch2[0]->taskId, seq2TaskId);
}

TEST(PrefillFirstSchedulerTest, Schedule_RespectsMaxNumBatchedTokens) {
  Config config = makeConfig(32, 8, 20, 0);
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  Sequence seq1{nextId(), 256, prompt(15), SamplingParams{.max_tokens = 5}};
  Sequence seq2{nextId(), 256, prompt(15), SamplingParams{.max_tokens = 5}};
  sched.add(seq1);
  sched.add(seq2);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  EXPECT_EQ(batch.size(), 1u)
      << "Only one sequence fits within max_num_batched_tokens";
}

TEST(PrefillFirstSchedulerTest, Schedule_RespectsHardcodedMaxNumSeqs) {
  Config config = makeConfig(32, 8, 256, 0);
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  Sequence seq1{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 5}};
  Sequence seq2{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 5}};
  Sequence seq3{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 5}};
  sched.add(seq1);
  sched.add(seq2);
  sched.add(seq3);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  EXPECT_EQ(batch.size(), 1u) << "At most 1 sequence in one batch";
}

TEST(PrefillFirstSchedulerTest,
     IsFinished_AfterAllSequencesFinish_ReturnsTrue) {
  Config config = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  Sequence seq{nextId(), 256, prompt(2), SamplingParams{.max_tokens = 1}};
  sched.add(seq);
  auto [batch, _] = sched.schedule();
  sched.postprocess(batch, {1});
  EXPECT_TRUE(sched.isFinished());
}

TEST(PrefillFirstSchedulerTest,
     Schedule_WhenSingleRunningNeedsBlockAndNoneFree_DoesNotSchedulePreempted) {
  Config config = makeConfig(1, 8, 256, 0);
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  Sequence seq{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 20}};
  sched.add(seq);

  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(prefill_batch.size(), 1u);
  Sequence* runningSeq = prefill_batch[0];
  sched.postprocess(prefill_batch, {1});

  for (int i = 0; i < 4; ++i) {
    auto [decode_batch, is_prefill] = sched.schedule();
    ASSERT_FALSE(is_prefill);
    ASSERT_EQ(decode_batch.size(), 1u);
    sched.postprocess(decode_batch, {static_cast<int64_t>(i + 2)});
  }
  ASSERT_EQ(runningSeq->size(), 9u);

  {
    auto [batch, is_prefill] = sched.schedule();
    EXPECT_TRUE(batch.empty())
        << "Preempted sequence must not be in the batch (it needed a block, "
           "had none, was preempted)";
  }
}

TEST(PrefillFirstSchedulerTest,
     Schedule_WhenSingleRunningNeedsBlock_TakesLastBlockAndContinuesDecode) {
  Config config = makeConfig(2, 8, 256, 0);
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  Sequence seq{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 20}};
  sched.add(seq);

  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(prefill_batch.size(), 1u);
  Sequence* runningSeq = prefill_batch[0];
  sched.postprocess(prefill_batch, {1});

  for (int i = 0; i < 4; ++i) {
    auto [decode_batch, is_prefill] = sched.schedule();
    ASSERT_FALSE(is_prefill);
    ASSERT_EQ(decode_batch.size(), 1u);
    sched.postprocess(decode_batch, {static_cast<int64_t>(i + 2)});
  }
  ASSERT_EQ(runningSeq->size(), 9u);

  {
    auto [batch, is_prefill] = sched.schedule();
    ASSERT_FALSE(is_prefill);
    EXPECT_FALSE(batch.empty()) << "Batch must not be empty as it should take "
                                   "the last block and continue decode";
  }
}

TEST(PrefillFirstSchedulerTest, PrefillsAllBeforeDecode) {
  Config config = makeConfig();
  auto queue = makeQueue();
  PrefillFirstScheduler sched{config, queue.get(), 1};
  Sequence seq1{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 10}};
  Sequence seq2{nextId(), 256, prompt(4), SamplingParams{.max_tokens = 10}};
  uint32_t seq2Id = seq2.taskId;
  sched.add(seq1);
  sched.add(seq2);

  auto [b1, pf1] = sched.schedule();
  ASSERT_TRUE(pf1);
  sched.postprocess(b1, {1});

  auto [b2, pf2] = sched.schedule();
  ASSERT_TRUE(pf2)
      << "PrefillFirst: seq2 should be prefilled even with seq1 running";
  ASSERT_EQ(b2.size(), 1u);
  EXPECT_EQ(b2[0]->taskId, seq2Id);
}

// --- make_scheduler factory test for MAX_OCCUPANCY ---

TEST(MakeSchedulerTest, MaxOccupancyPolicy_CreatesMaxOccupancyScheduler) {
  Config config = makeConfig();
  config.scheduling_policy = SchedulingPolicy::MAX_OCCUPANCY;
  auto queue = makeQueue();
  auto sched = makeScheduler(config, queue.get(), 1);
  ASSERT_NE(sched, nullptr);
  EXPECT_NE(dynamic_cast<MaxOccupancyScheduler*>(sched.get()), nullptr);
}

// --- MaxOccupancyScheduler tests ---

TEST(MaxOccupancySchedulerTest, PrefillsToFillGap) {
  Config config = makeConfig();
  auto queue = makeQueue();
  MaxOccupancyScheduler sched{config, queue.get(), 2};

  sched.addRequest(nextId(), prompt(4), {.max_tokens = 10});
  sched.addRequest(nextId(), prompt(4), {.max_tokens = 10});

  auto [b1, pf1] = sched.schedule();
  ASSERT_TRUE(pf1);
  EXPECT_EQ(b1.size(), 2u)
      << "Should prefill both to fill max_in_flight_count=2";
  sched.postprocess(b1, {1, 1});

  auto [b2, pf2] = sched.schedule();
  ASSERT_FALSE(pf2) << "No gaps, should decode";
  EXPECT_EQ(b2.size(), 2u);
}

TEST(MaxOccupancySchedulerTest, PrefillsOnlyGapCount_WhenOneFinishes) {
  Config config = makeConfig();
  auto queue = makeQueue();
  MaxOccupancyScheduler sched{config, queue.get(), 2};

  sched.addRequest(nextId(), prompt(4), {.max_tokens = 2});
  sched.addRequest(nextId(), prompt(4), {.max_tokens = 20});
  sched.addRequest(nextId(), prompt(4), {.max_tokens = 20});

  // Prefill first 2 (max_in_flight_count=2)
  auto [b1, pf1] = sched.schedule();
  ASSERT_TRUE(pf1);
  EXPECT_EQ(b1.size(), 2u);
  Sequence* shortSeq = b1[0];
  sched.postprocess(b1, {1, 1});

  // Decode: both get 1 token
  auto [b2, pf2] = sched.schedule();
  ASSERT_FALSE(pf2);
  EXPECT_EQ(b2.size(), 2u);
  sched.postprocess(b2, {2, 2});

  // short_seq hit max_tokens=2, should be finished
  EXPECT_TRUE(shortSeq->isFinished());

  // Now running=1, waiting=1: should prefill exactly 1 to fill the gap
  auto [b3, pf3] = sched.schedule();
  ASSERT_TRUE(pf3) << "Should prefill to fill the gap left by finished seq";
  EXPECT_EQ(b3.size(), 1u) << "Should prefill only 1 (the gap), not 2";
  sched.postprocess(b3, {1});

  // Now running=2, should decode at full capacity
  auto [b4, pf4] = sched.schedule();
  ASSERT_FALSE(pf4);
  EXPECT_EQ(b4.size(), 2u);
}

TEST(MaxOccupancySchedulerTest, DecodesAtFullCapacity_WhenNoWaiting) {
  Config config = makeConfig();
  auto queue = makeQueue();
  MaxOccupancyScheduler sched{config, queue.get(), 2};

  sched.addRequest(nextId(), prompt(4), {.max_tokens = 10});
  sched.addRequest(nextId(), prompt(4), {.max_tokens = 10});

  auto [b1, pf1] = sched.schedule();
  ASSERT_TRUE(pf1);
  sched.postprocess(b1, {1, 1});

  // No waiting, running at capacity: decode
  auto [b2, pf2] = sched.schedule();
  ASSERT_FALSE(pf2);
  EXPECT_EQ(b2.size(), 2u);
}

TEST(MaxOccupancySchedulerTest,
     DecodesWithoutPrefill_WhenNoWaitingAndGapExists) {
  Config config = makeConfig();
  auto queue = makeQueue();
  MaxOccupancyScheduler sched{config, queue.get(), 2};

  sched.addRequest(nextId(), prompt(4), {.max_tokens = 1});
  sched.addRequest(nextId(), prompt(4), {.max_tokens = 10});

  auto [b1, pf1] = sched.schedule();
  ASSERT_TRUE(pf1);
  sched.postprocess(b1, {1, 1});

  // seq1 finishes on decode
  auto [b2, pf2] = sched.schedule();
  ASSERT_FALSE(pf2);
  sched.postprocess(b2, {99, 2});

  // Gap exists (running=1) but nothing waiting: just decode
  auto [b3, pf3] = sched.schedule();
  ASSERT_FALSE(pf3) << "No waiting requests, should decode even with gap";
  EXPECT_EQ(b3.size(), 1u);
}

TEST(MaxOccupancySchedulerTest, PrefillsFromEmpty_LikePrefillFirst) {
  Config config = makeConfig();
  auto queue = makeQueue();
  MaxOccupancyScheduler sched{config, queue.get(), 2};

  sched.addRequest(nextId(), prompt(4), {.max_tokens = 10});

  auto [b1, pf1] = sched.schedule();
  ASSERT_TRUE(pf1) << "With 0 running and waiting, should prefill";
  EXPECT_EQ(b1.size(), 1u);
}

TEST(MaxOccupancySchedulerTest, IsFinished_AfterAllComplete) {
  Config config = makeConfig();
  auto queue = makeQueue();
  MaxOccupancyScheduler sched{config, queue.get(), 2};

  sched.addRequest(nextId(), prompt(2), {.max_tokens = 1});

  auto [b1, _1] = sched.schedule();
  sched.postprocess(b1, {1});
  EXPECT_TRUE(sched.isFinished());
}

TEST(MaxOccupancySchedulerTest, ContinuousRefill_MaintainsFullOccupancy) {
  Config config = makeConfig(64, 8, 256, 0);
  auto queue = makeQueue();
  MaxOccupancyScheduler sched{config, queue.get(), 3};

  // Add 5 requests: 3 will be prefilled, 2 will wait
  for (int i = 0; i < 5; ++i) {
    sched.addRequest(nextId(), prompt(4), {.max_tokens = 2});
  }

  // Prefill 3
  auto [b1, pf1] = sched.schedule();
  ASSERT_TRUE(pf1);
  EXPECT_EQ(b1.size(), 3u);
  sched.postprocess(b1, {1, 1, 1});

  // Decode 3: first one finishes (max_tokens=2)
  auto [b2, pf2] = sched.schedule();
  ASSERT_FALSE(pf2);
  EXPECT_EQ(b2.size(), 3u);
  sched.postprocess(b2, {2, 2, 2});

  // All 3 finished. 2 waiting. Prefill 2.
  auto [b3, pf3] = sched.schedule();
  ASSERT_TRUE(pf3);
  EXPECT_EQ(b3.size(), 2u);
}
}  // namespace
}  // namespace tt::runners::schedulers
