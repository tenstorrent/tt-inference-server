// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "llm_engine/config.hpp"
#include "llm_engine/engine/scheduler.hpp"
#include "llm_engine/engine/sequence.hpp"
#include "llm_engine/sampling_params.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>

namespace llm_engine {
namespace {

Config make_config(int num_blocks = 32, int block_size = 8,
                   int max_batched_tokens = 256, int max_seqs = 4, int eos = 0) {
  Config c;
  c.num_kvcache_blocks = num_blocks;
  c.kvcache_block_size = block_size;
  c.max_num_batched_tokens = max_batched_tokens;
  c.max_num_seqs = max_seqs;
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
  Scheduler sched{config};
  EXPECT_TRUE(sched.is_finished());
}

TEST(SchedulerTest, IsFinished_AfterAdd_ReturnsFalse) {
  Config config = make_config();
  Scheduler sched{config};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 10}};
  sched.add(seq);
  EXPECT_FALSE(sched.is_finished());
}

TEST(SchedulerTest, Schedule_WithOneWaiting_ReturnsPrefillBatch) {
  Config config = make_config();
  Scheduler sched{config};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 10}};
  sched.add(seq);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(batch.size(), 1u);
  EXPECT_EQ(batch[0], &seq);
  EXPECT_EQ(batch[0]->status_, SequenceStatus::RUNNING);
}

TEST(SchedulerTest, Schedule_WhenNoWaitingAndOneRunning_ReturnsDecodeBatch) {
  Config config = make_config();
  Scheduler sched{config};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 10}};
  sched.add(seq);
  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(prefill_batch.size(), 1u);
  std::vector<int64_t> tokens = {1};
  sched.postprocess(prefill_batch, tokens);

  auto [decode_batch, is_decode] = sched.schedule();
  ASSERT_FALSE(is_decode);
  ASSERT_EQ(decode_batch.size(), 1u);
  EXPECT_EQ(decode_batch[0], &seq);
}

TEST(SchedulerTest, Postprocess_WhenTokenReachesMaxTokens_MarksFinished) {
  Config config = make_config();
  Scheduler sched{config};
  SamplingParams params;
  params.max_tokens = 2;
  Sequence seq{prompt(2), params};
  sched.add(seq);
  auto [batch1, _1] = sched.schedule();
  ASSERT_EQ(batch1.size(), 1u);
  sched.postprocess(batch1, {1});
  EXPECT_FALSE(seq.is_finished());
  auto [batch2, _2] = sched.schedule();
  ASSERT_EQ(batch2.size(), 1u);
  sched.postprocess(batch2, {2});
  EXPECT_TRUE(seq.is_finished());
}

TEST(SchedulerTest, Postprocess_WhenEosToken_MarksFinished) {
  Config config = make_config(32, 8, 256, 4, 99);
  Scheduler sched{config};
  Sequence seq{prompt(2), SamplingParams{.max_tokens = 100, .ignore_eos = false}};
  sched.add(seq);
  auto [batch, _] = sched.schedule();
  ASSERT_EQ(batch.size(), 1u);
  sched.postprocess(batch, {99});
  EXPECT_TRUE(seq.is_finished());
}

TEST(SchedulerTest, Preempt_MovesSequenceBackToWaiting) {
  Config config = make_config();
  Scheduler sched{config};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 10}};
  sched.add(seq);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  sched.preempt(seq);
  EXPECT_EQ(seq.status_, SequenceStatus::WAITING);
  auto [batch2, is_prefill2] = sched.schedule();
  EXPECT_TRUE(is_prefill2);
  EXPECT_EQ(batch2.size(), 1u);
  EXPECT_EQ(batch2[0], &seq);
}

TEST(SchedulerTest, Schedule_PrefillPrioritizedOverDecode) {
  Config config = make_config();
  Scheduler sched{config};
  Sequence seq1{prompt(4), SamplingParams{.max_tokens = 10}};
  Sequence seq2{prompt(4), SamplingParams{.max_tokens = 10}};
  sched.add(seq1);
  auto [batch1, prefill1] = sched.schedule();
  ASSERT_TRUE(prefill1);
  sched.postprocess(batch1, {1});
  sched.add(seq2);
  auto [batch2, prefill2] = sched.schedule();
  EXPECT_TRUE(prefill2) << "Prefill (seq2) should be chosen over decode (seq1)";
  ASSERT_EQ(batch2.size(), 1u);
  EXPECT_EQ(batch2[0], &seq2);
}

TEST(SchedulerTest, Schedule_RespectsMaxNumBatchedTokens) {
  Config config = make_config(32, 8, 20, 4, 0);
  Scheduler sched{config};
  Sequence seq1{prompt(15), SamplingParams{.max_tokens = 5}};
  Sequence seq2{prompt(15), SamplingParams{.max_tokens = 5}};
  sched.add(seq1);
  sched.add(seq2);
  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  EXPECT_EQ(batch.size(), 1u) << "Only one sequence fits within max_num_batched_tokens";
}

TEST(SchedulerTest, Schedule_RespectsMaxNumSeqs) {
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
}

TEST(SchedulerTest, IsFinished_AfterAllSequencesFinish_ReturnsTrue) {
  Config config = make_config();
  Scheduler sched{config};
  Sequence seq{prompt(2), SamplingParams{.max_tokens = 1}};
  sched.add(seq);
  auto [batch, _] = sched.schedule();
  sched.postprocess(batch, {1});
  EXPECT_TRUE(sched.is_finished());
}

TEST(SchedulerTest, Schedule_WhenSingleRunningNeedsBlockAndNoneFree_DoesNotSchedulePreempted) {
  Config config = make_config(1, 8, 256, 4, 0);
  Scheduler sched{config};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 20}};
  sched.add(seq);

  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(prefill_batch.size(), 1u);
  sched.postprocess(prefill_batch, {1});

  for (int i = 0; i < 4; ++i) {
    auto [decode_batch, is_prefill] = sched.schedule();
    ASSERT_FALSE(is_prefill);
    ASSERT_EQ(decode_batch.size(), 1u);
    sched.postprocess(decode_batch, {static_cast<int64_t>(i + 2)});
  }
  ASSERT_EQ(seq.size(), 9u);

  {
    auto [batch, is_prefill] = sched.schedule();
    EXPECT_TRUE(batch.empty())
        << "Preempted sequence must not be in the batch (it needed a block, had none, was preempted)";
  }
}

TEST(SchedulerTest, Schedule_WhenSingleRunningNeedsBlock_TakesLastBlockAndContinuesDecode) {
  Config config = make_config(2, 8, 256, 4, 0);
  Scheduler sched{config};
  Sequence seq{prompt(4), SamplingParams{.max_tokens = 20}};
  sched.add(seq);

  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  ASSERT_EQ(prefill_batch.size(), 1u);
  sched.postprocess(prefill_batch, {1});

  for (int i = 0; i < 4; ++i) {
    auto [decode_batch, is_prefill] = sched.schedule();
    ASSERT_FALSE(is_prefill);
    ASSERT_EQ(decode_batch.size(), 1u);
    sched.postprocess(decode_batch, {static_cast<int64_t>(i + 2)});
  }
  ASSERT_EQ(seq.size(), 9u);

  {
    auto [batch, is_prefill] = sched.schedule();
    ASSERT_FALSE(is_prefill);
    EXPECT_FALSE(batch.empty())
        << "Batch must not be empty as it should take the last block and continue decode";
  }
}

// ---------------------------------------------------------------------------
// Race-detection test: exercises the exact interleaving that occurs when the
// device-to-host reader thread calls postprocess (via on_decode_token) while
// the main engine thread calls schedule (via step).
//
// Under ThreadSanitizer (build with --tsan) this test flags data races on
// Scheduler's running_/waiting_ deques and BlockManager state.
// Without TSan the test may still crash or corrupt data on real races.
// ---------------------------------------------------------------------------
TEST(SchedulerTest, ConcurrentScheduleAndPostprocess_DetectsRace) {
  Config config = make_config(/*num_blocks=*/128, /*block_size=*/8,
                              /*max_batched_tokens=*/256, /*max_seqs=*/1,
                              /*eos=*/0);
  Scheduler sched{config};

  constexpr int NUM_SEQS = 4;
  constexpr int MAX_TOKENS = 50;
  std::vector<Sequence> seqs;
  seqs.reserve(NUM_SEQS);
  for (int i = 0; i < NUM_SEQS; ++i) {
    seqs.emplace_back(prompt(4), SamplingParams{.max_tokens = MAX_TOKENS});
    sched.add(seqs.back());
  }

  auto [prefill_batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  std::vector<int64_t> prefill_tokens(prefill_batch.size(), 1);
  sched.postprocess(prefill_batch, prefill_tokens);

  // Now all sequences are in RUNNING state — the race-prone region.
  // Thread A: main engine thread calling schedule() (reads running_/waiting_)
  // Thread B: reader thread calling postprocess() (mutates running_, deallocates blocks)
  std::atomic<bool> stop{false};

  std::thread reader_thread([&]() {
    while (!stop.load(std::memory_order_relaxed)) {
      for (auto& seq : seqs) {
        if (seq.is_finished()) continue;
        std::vector<Sequence*> batch = {&seq};
        std::vector<int64_t> tokens = {1};
        sched.postprocess(batch, tokens);
      }
      std::this_thread::yield();
    }
  });

  for (int i = 0; i < 200; ++i) {
    auto [batch, is_pf] = sched.schedule();
    (void)is_pf;
    (void)batch;
    std::this_thread::yield();
  }

  stop.store(true, std::memory_order_relaxed);
  reader_thread.join();
}

}  // namespace
}  // namespace llm_engine
