// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "llm_engine/config.hpp"
#include "llm_engine/engine/boost_ipc_task_queue.hpp"
#include "llm_engine/engine/scheduler.hpp"
#include "llm_engine/engine/sequence.hpp"
#include "llm_engine/sampling_params.hpp"

#include <gtest/gtest.h>

#include <string>
#include <unistd.h>
#include <vector>

namespace llm_engine {
namespace {

/// Test fixture that creates and cleans up an IPC message queue per test.
class SchedulerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    queue_name_ = "/tt_sched_test_" + std::to_string(getpid()) + "_" +
                  std::to_string(counter_++);
    BoostIpcTaskQueue::remove(queue_name_);
  }

  void TearDown() override {
    BoostIpcTaskQueue::remove(queue_name_);
  }

  Config make_config(int num_blocks = 32, int block_size = 8,
                     int max_batched_tokens = 256, int max_seqs = 4,
                     int eos = 0) {
    Config c;
    c.num_kvcache_blocks = num_blocks;
    c.kvcache_block_size = block_size;
    c.max_num_batched_tokens = max_batched_tokens;
    c.max_num_seqs = max_seqs;
    c.eos = eos;
    c.task_queue_name = queue_name_;
    c.task_queue_capacity = 128;
    return c;
  }

  static std::vector<int64_t> prompt(size_t len) {
    std::vector<int64_t> p;
    for (size_t i = 0; i < len; ++i) p.push_back(static_cast<int64_t>(i));
    return p;
  }

  std::string queue_name_;
  static int counter_;
};

int SchedulerTest::counter_ = 0;

TEST_F(SchedulerTest, IsFinished_WhenEmpty_ReturnsTrue) {
  Config config = make_config();
  Scheduler sched{config};
  EXPECT_TRUE(sched.is_finished());
}

TEST_F(SchedulerTest, IsFinished_AfterAdd_ReturnsFalse) {
  Config config = make_config();
  Scheduler sched{config};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 10}};
  sched.add(seq);
  EXPECT_FALSE(sched.is_finished());
}

TEST_F(SchedulerTest, Schedule_WithOneWaiting_ReturnsPrefillBatch) {
  Config config = make_config();
  Scheduler sched{config};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 10}};
  sched.add(seq);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(batch.size(), 1u);
  // The returned pointer is a new Sequence from IPC deserialization.
  EXPECT_EQ(batch[0]->seq_id, seq.seq_id);
  EXPECT_EQ(batch[0]->status_, SequenceStatus::RUNNING);
  delete batch[0];
}

TEST_F(SchedulerTest, Schedule_WhenNoWaitingAndOneRunning_ReturnsDecodeBatch) {
  Config config = make_config();
  Scheduler sched{config};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 10}};
  sched.add(seq);
  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(prefill_batch.size(), 1u);
  // Simulate model output: one token per sequence.
  std::vector<int64_t> tokens = {1};
  sched.postprocess(prefill_batch, tokens);

  auto [decode_batch, is_decode] = sched.schedule();
  ASSERT_FALSE(is_decode);
  ASSERT_EQ(decode_batch.size(), 1u);
  EXPECT_EQ(decode_batch[0]->seq_id, prefill_batch[0]->seq_id);
  delete prefill_batch[0];
}

TEST_F(SchedulerTest, Postprocess_WhenTokenReachesMaxTokens_MarksFinished) {
  Config config = make_config();
  Scheduler sched{config};
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
  delete batch1[0];
}

TEST_F(SchedulerTest, Postprocess_WhenEosToken_MarksFinished) {
  Config config = make_config(32, 8, 256, 4, 99);
  Scheduler sched{config};
  Sequence seq{prompt(2),
               SamplingParams{.max_tokens = 100, .ignore_eos = false}};
  sched.add(seq);
  auto [batch, _] = sched.schedule();
  ASSERT_EQ(batch.size(), 1u);
  sched.postprocess(batch, {99});
  EXPECT_TRUE(batch[0]->is_finished());
  delete batch[0];
}

TEST_F(SchedulerTest, Preempt_MovesSequenceBackToQueue) {
  Config config = make_config();
  Scheduler sched{config};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 10}};
  sched.add(seq);

  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(batch.size(), 1u);
  int original_seq_id = batch[0]->seq_id;

  sched.preempt(*batch[0]);
  delete batch[0];

  // The preempted sequence should be back in the queue.
  auto [batch2, is_prefill2] = sched.schedule();
  EXPECT_TRUE(is_prefill2);
  ASSERT_EQ(batch2.size(), 1u);
  EXPECT_EQ(batch2[0]->seq_id, original_seq_id);
  delete batch2[0];
}

TEST_F(SchedulerTest, Schedule_PrefillPrioritizedOverDecode) {
  Config config = make_config();
  Scheduler sched{config};
  Sequence seq1{prompt(4), SamplingParams{.max_tokens = 10}};
  Sequence seq2{prompt(4), SamplingParams{.max_tokens = 10}};

  sched.add(seq1);
  auto [batch1, prefill1] = sched.schedule();
  ASSERT_TRUE(prefill1);
  sched.postprocess(batch1, {1});

  // Add seq2 while seq1 is still running.
  sched.add(seq2);
  auto [batch2, prefill2] = sched.schedule();
  EXPECT_TRUE(prefill2) << "Prefill (seq2) should be chosen over decode (seq1)";
  ASSERT_EQ(batch2.size(), 1u);
  EXPECT_EQ(batch2[0]->seq_id, seq2.seq_id);

  delete batch1[0];
  delete batch2[0];
}

TEST_F(SchedulerTest, Schedule_RespectsMaxNumBatchedTokens) {
  Config config = make_config(32, 8, 20, 4, 0);
  Scheduler sched{config};
  Sequence seq1{prompt(15), SamplingParams{.max_tokens = 5}};
  Sequence seq2{prompt(15), SamplingParams{.max_tokens = 5}};
  sched.add(seq1);
  sched.add(seq2);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  EXPECT_EQ(batch.size(), 1u)
      << "Only one sequence fits within max_num_batched_tokens";
  for (auto* s : batch) delete s;
}

TEST_F(SchedulerTest, Schedule_RespectsMaxNumSeqs) {
  Config config = make_config(32, 8, 256, 2, 0);
  Scheduler sched{config};
  Sequence seq1{prompt(4), SamplingParams{.max_tokens = 5}};
  Sequence seq2{prompt(4), SamplingParams{.max_tokens = 5}};
  Sequence seq3{prompt(4), SamplingParams{.max_tokens = 5}};
  sched.add(seq1);
  sched.add(seq2);
  sched.add(seq3);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  EXPECT_EQ(batch.size(), 2u) << "At most max_num_seqs in one batch";
  for (auto* s : batch) delete s;
}

TEST_F(SchedulerTest, IsFinished_AfterAllSequencesFinish_ReturnsTrue) {
  Config config = make_config();
  Scheduler sched{config};
  Sequence seq{prompt(2), SamplingParams{.max_tokens = 1}};
  sched.add(seq);
  auto [batch, _] = sched.schedule();
  sched.postprocess(batch, {1});
  EXPECT_TRUE(sched.is_finished());
  delete batch[0];
}

}  // namespace
}  // namespace llm_engine
