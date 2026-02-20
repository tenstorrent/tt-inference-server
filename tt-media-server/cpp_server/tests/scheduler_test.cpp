// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/scheduler.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/sampling_params.hpp"
#include <gtest/gtest.h>
#include <vector>
#include "runners/llm_runner/in_memory_task_queue.hpp"

namespace llm_engine {
namespace {

  std::shared_ptr<ITaskQueue> make_queue() {
    return std::make_shared<InMemoryTaskQueue>();
  }

Config make_config(int num_blocks = 32, int block_size = 8,
                   int max_batched_tokens = 256, int eos = 0) {
  Config c;
  c.num_kvcache_blocks = num_blocks;
  c.kvcache_block_size = block_size;
  c.max_num_batched_tokens = max_batched_tokens;
  c.eos = eos;
  return c;
}

std::vector<int64_t> prompt(size_t len) {
  std::vector<int64_t> p;
  for (size_t i = 0; i < len; ++i) p.push_back(static_cast<int64_t>(i));
  return p;
}

TEST(SchedulerTest, IsFinished_WhenEmpty_ReturnsTrue) {
  Config config = make_config();
  Scheduler sched{config, make_queue().get()};
  EXPECT_TRUE(sched.is_finished());
}

TEST(SchedulerTest, IsFinished_AfterAdd_ReturnsFalse) {
  Config config = make_config();
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 10}};
  sched.add(seq);
  EXPECT_FALSE(sched.is_finished());
}

TEST(SchedulerTest, Schedule_WithOneWaiting_ReturnsPrefillBatch) {
  Config config = make_config();
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 10}};
  TaskID expected_id = seq.task_id;
  sched.add(seq);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(batch.size(), 1u);
  EXPECT_EQ(batch[0]->task_id, expected_id);
  EXPECT_EQ(batch[0]->status_, SequenceStatus::IN_FLIGHT);
}

TEST(SchedulerTest, Schedule_OneRequest_FirstCallPrefill_SecondCallEmpty) {
  Config config = make_config();
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  sched.add_request(prompt(4), SamplingParams{.max_tokens = 10});

  auto [batch1, is_prefill1] = sched.schedule();
  ASSERT_TRUE(is_prefill1);
  EXPECT_EQ(batch1.size(), 1u);

  auto [batch2, is_prefill2] = sched.schedule();
  EXPECT_TRUE(batch2.empty());
}

TEST(SchedulerTest, Schedule_WhenNoWaitingAndOneRunning_ReturnsDecodeBatch) {
  Config config = make_config();
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 10}};
  TaskID expected_id = seq.task_id;
  sched.add(seq);
  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(prefill_batch.size(), 1u);
  std::vector<int64_t> tokens = {1};
  sched.postprocess(prefill_batch, tokens);

  auto [decode_batch, is_decode] = sched.schedule();
  ASSERT_FALSE(is_decode);
  ASSERT_EQ(decode_batch.size(), 1u);
  EXPECT_EQ(decode_batch[0]->task_id, expected_id);
}

TEST(SchedulerTest, OneRequest_PrefillThenDecodeThenEos) {
  Config config = make_config(32, 8, 256, 99);
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  sched.add_request(prompt(4), {.max_tokens = 10, .ignore_eos = false});

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
  EXPECT_TRUE(sched.is_finished());
}

TEST(SchedulerTest, OneRequest_PrefillThenDecodeThenMaxTokens) {
  Config config = make_config();
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  sched.add_request(prompt(4), {.max_tokens = 2});

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
  EXPECT_TRUE(sched.is_finished());
}

TEST(SchedulerTest, Postprocess_WhenTokenReachesMaxTokens_MarksFinished) {
  Config config = make_config();
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  SamplingParams params;
  params.max_tokens = 2;
  Sequence seq{prompt(2), params};
  sched.add(seq);
  auto [batch1, _1] = sched.schedule();
  ASSERT_EQ(batch1.size(), 1u);
  sched.postprocess(batch1, {1});
  EXPECT_FALSE(batch1[0]->is_finished());
  auto [batch2, _2] = sched.schedule();
  ASSERT_EQ(batch2.size(), 1u);
  sched.postprocess(batch2, {2});
  EXPECT_TRUE(batch2[0]->is_finished());
}

TEST(SchedulerTest, Postprocess_WhenEosToken_MarksFinished) {
  Config config = make_config(32, 8, 256, 99);
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq{prompt(2), SamplingParams{.max_tokens = 100, .ignore_eos = false}};
  sched.add(seq);
  auto [batch, _] = sched.schedule();
  ASSERT_EQ(batch.size(), 1u);
  sched.postprocess(batch, {99});
  EXPECT_TRUE(batch[0]->is_finished());
}

TEST(SchedulerTest, Preempt_MovesSequenceBackToWaiting) {
  Config config = make_config();
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 10}};
  TaskID expected_id = seq.task_id;
  sched.add(seq);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(batch.size(), 1u);
  sched.preempt(*batch[0]);
  EXPECT_EQ(batch[0]->status_, SequenceStatus::WAITING);
  auto [batch2, is_prefill2] = sched.schedule();
  EXPECT_TRUE(is_prefill2);
  EXPECT_EQ(batch2.size(), 1u);
  EXPECT_EQ(batch2[0]->task_id, expected_id);
}

TEST(SchedulerTest, Schedule_PrefillPrioritizedOverDecode) {
  Config config = make_config();
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq1{prompt(4), SamplingParams{.max_tokens = 10}};
  Sequence seq2{prompt(4), SamplingParams{.max_tokens = 10}};
  TaskID seq2_task_id = seq2.task_id;
  sched.add(seq1);
  auto [batch1, prefill1] = sched.schedule();
  ASSERT_TRUE(prefill1);
  sched.postprocess(batch1, {1});
  sched.add(seq2);
  auto [batch2, prefill2] = sched.schedule();
  EXPECT_TRUE(prefill2) << "Prefill (seq2) should be chosen over decode (seq1)";
  ASSERT_EQ(batch2.size(), 1u);
  EXPECT_EQ(batch2[0]->task_id, seq2_task_id);
}

TEST(SchedulerTest, Schedule_RespectsMaxNumBatchedTokens) {
  Config config = make_config(32, 8, 20, 0);
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq1{prompt(15), SamplingParams{.max_tokens = 5}};
  Sequence seq2{prompt(15), SamplingParams{.max_tokens = 5}};
  sched.add(seq1);
  sched.add(seq2);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  EXPECT_EQ(batch.size(), 1u) << "Only one sequence fits within max_num_batched_tokens";
}

TEST(SchedulerTest, Schedule_RespectsHardcodedMaxNumSeqs) {
  Config config = make_config(32, 8, 256, 0);
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq1{prompt(4), SamplingParams{.max_tokens = 5}};
  Sequence seq2{prompt(4), SamplingParams{.max_tokens = 5}};
  Sequence seq3{prompt(4), SamplingParams{.max_tokens = 5}};
  sched.add(seq1);
  sched.add(seq2);
  sched.add(seq3);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  EXPECT_EQ(batch.size(), 1u) << "At most 1 sequence in one batch";
}

TEST(SchedulerTest, IsFinished_AfterAllSequencesFinish_ReturnsTrue) {
  Config config = make_config();
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq{prompt(2), SamplingParams{.max_tokens = 1}};
  sched.add(seq);
  auto [batch, _] = sched.schedule();
  sched.postprocess(batch, {1});
  EXPECT_TRUE(sched.is_finished());
}

TEST(SchedulerTest, Schedule_WhenSingleRunningNeedsBlockAndNoneFree_DoesNotSchedulePreempted) {
  Config config = make_config(1, 8, 256, 0);
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 20}};
  sched.add(seq);

  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(prefill_batch.size(), 1u);
  Sequence* running_seq = prefill_batch[0];
  sched.postprocess(prefill_batch, {1});

  for (int i = 0; i < 4; ++i) {
    auto [decode_batch, is_prefill] = sched.schedule();
    ASSERT_FALSE(is_prefill);
    ASSERT_EQ(decode_batch.size(), 1u);
    sched.postprocess(decode_batch, {static_cast<int64_t>(i + 2)});
  }
  ASSERT_EQ(running_seq->size(), 9u);

  {
    auto [batch, is_prefill] = sched.schedule();
    EXPECT_TRUE(batch.empty())
        << "Preempted sequence must not be in the batch (it needed a block, had none, was preempted)";
  }
}

TEST(SchedulerTest, Schedule_WhenSingleRunningNeedsBlock_TakesLastBlockAndContinuesDecode) {
  Config config = make_config(2, 8, 256, 0);
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 20}};
  sched.add(seq);

  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(prefill_batch.size(), 1u);
  Sequence* running_seq = prefill_batch[0];
  sched.postprocess(prefill_batch, {1});

  for (int i = 0; i < 4; ++i) {
    auto [decode_batch, is_prefill] = sched.schedule();
    ASSERT_FALSE(is_prefill);
    ASSERT_EQ(decode_batch.size(), 1u);
    sched.postprocess(decode_batch, {static_cast<int64_t>(i + 2)});
  }
  ASSERT_EQ(running_seq->size(), 9u);

  {
    auto [batch, is_prefill] = sched.schedule();
    ASSERT_FALSE(is_prefill);
    EXPECT_FALSE(batch.empty())
        << "Batch must not be empty as it should take the last block and continue decode";
  }
}

} 
}
