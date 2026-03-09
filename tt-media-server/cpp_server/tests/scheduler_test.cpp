// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/scheduler.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/sampling_params.hpp"
#include <gtest/gtest.h>
#include <optional>
#include <vector>
#include "runners/llm_runner/in_memory_task_queue.hpp"

namespace llm_engine {
namespace {

  std::shared_ptr<ITaskQueue> make_queue() {
    return std::make_shared<InMemoryTaskQueue>();
  }

Config make_config(int num_blocks = 32, int block_size = 8,
                   int max_batched_tokens = 256, int eos = 0,
                   std::optional<std::vector<int64_t>> stop_ids = std::nullopt) {
  Config c;
  c.num_kvcache_blocks = num_blocks;
  c.kvcache_block_size = block_size;
  c.max_num_batched_tokens = max_batched_tokens;
  c.eos = eos;
  if (stop_ids) {
    c.stop_token_ids = std::move(*stop_ids);
  }
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
  Sequence seq{256, prompt(4), SamplingParams{.max_tokens = 10}};
  sched.add(seq);
  EXPECT_FALSE(sched.is_finished());
}

TEST(SchedulerTest, Schedule_WithOneWaiting_ReturnsPrefillBatch) {
  Config config = make_config();
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq{256, prompt(4), SamplingParams{.max_tokens = 10}};
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
  Sequence seq{256, prompt(4), SamplingParams{.max_tokens = 10}};
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
  Config config = make_config(32, 8, 256, 99, std::vector<int64_t>{99});
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
  Sequence seq{256, prompt(2), params};
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
  Config config = make_config(32, 8, 256, 99, std::vector<int64_t>{99});
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq{256, prompt(2), SamplingParams{.max_tokens = 100, .ignore_eos = false}};
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
  Sequence seq{256, prompt(4), SamplingParams{.max_tokens = 10}};
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
  Sequence seq1{256, prompt(4), SamplingParams{.max_tokens = 10}};
  Sequence seq2{256, prompt(4), SamplingParams{.max_tokens = 10}};
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
  Sequence seq1{256, prompt(15), SamplingParams{.max_tokens = 5}};
  Sequence seq2{256, prompt(15), SamplingParams{.max_tokens = 5}};
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
  Sequence seq1{256, prompt(4), SamplingParams{.max_tokens = 5}};
  Sequence seq2{256, prompt(4), SamplingParams{.max_tokens = 5}};
  Sequence seq3{256, prompt(4), SamplingParams{.max_tokens = 5}};
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
  Sequence seq{256, prompt(2), SamplingParams{.max_tokens = 1}};
  sched.add(seq);
  auto [batch, _] = sched.schedule();
  sched.postprocess(batch, {1});
  EXPECT_TRUE(sched.is_finished());
}

TEST(SchedulerTest, Schedule_WhenSingleRunningNeedsBlockAndNoneFree_DoesNotSchedulePreempted) {
  Config config = make_config(1, 8, 256, 0);
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};
  Sequence seq{256, prompt(4), SamplingParams{.max_tokens = 20}};
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
  Sequence seq{256, prompt(4), SamplingParams{.max_tokens = 20}};
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

// --- max_in_flight_count tests ---

TEST(SchedulerTest, MaxInFlight_LimitsPrefillBatchSize) {
  Config config = make_config(64, 8, 256, 0);
  config.max_num_seqs = 16;
  config.max_in_flight_count = 3;
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};

  for (int i = 0; i < 10; ++i) {
    sched.add_request(prompt(4), {.max_tokens = 20});
  }

  auto [batch, is_prefill] = sched.schedule();
  ASSERT_TRUE(is_prefill);
  EXPECT_EQ(batch.size(), 3u) << "Prefill batch capped by max_in_flight_count=3";
}

TEST(SchedulerTest, MaxInFlight_LimitsDecodeBatchSize) {
  Config config = make_config(64, 8, 256, 0);
  config.max_num_seqs = 16;
  config.max_in_flight_count = 3;
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};

  // Prefill 3, then add 3 more waiting to force a second prefill round
  for (int i = 0; i < 6; ++i) {
    sched.add_request(prompt(4), {.max_tokens = 20});
  }

  auto [b1, pf1] = sched.schedule();
  ASSERT_TRUE(pf1);
  ASSERT_EQ(b1.size(), 3u);
  sched.postprocess(b1, {1, 1, 1});

  // 3 running, 3 waiting. Prefill wins (waiting non-empty).
  auto [b2, pf2] = sched.schedule();
  ASSERT_TRUE(pf2);
  ASSERT_EQ(b2.size(), 3u);
  sched.postprocess(b2, {1, 1, 1});

  // 6 running, 0 waiting. Decode capped at 3.
  auto [b3, pf3] = sched.schedule();
  ASSERT_FALSE(pf3);
  EXPECT_EQ(b3.size(), 3u) << "Decode batch capped by max_in_flight_count=3";
}

TEST(SchedulerTest, MaxInFlight_PrefillPrioritizedOverDecodeWhenWaiting) {
  Config config = make_config(64, 8, 256, 0);
  config.max_num_seqs = 16;
  config.max_in_flight_count = 3;
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};

  for (int i = 0; i < 5; ++i) {
    sched.add_request(prompt(4), {.max_tokens = 20});
  }

  auto [b1, pf1] = sched.schedule();
  ASSERT_TRUE(pf1);
  ASSERT_EQ(b1.size(), 3u);
  sched.postprocess(b1, {1, 1, 1});

  // 3 running, 2 waiting: prefill wins over decode
  auto [b2, pf2] = sched.schedule();
  ASSERT_TRUE(pf2) << "With waiting requests, prefill should be chosen over decode";
  EXPECT_EQ(b2.size(), 2u) << "Remaining 2 waiting should be prefilled";
}

TEST(SchedulerTest, MaxInFlight_DecodesWhenNothingWaiting) {
  Config config = make_config(64, 8, 256, 0);
  config.max_num_seqs = 16;
  config.max_in_flight_count = 3;
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};

  for (int i = 0; i < 3; ++i) {
    sched.add_request(prompt(4), {.max_tokens = 20});
  }

  auto [b1, pf1] = sched.schedule();
  ASSERT_TRUE(pf1);
  ASSERT_EQ(b1.size(), 3u);
  sched.postprocess(b1, {1, 1, 1});

  // 3 running, 0 waiting → decode
  auto [b2, pf2] = sched.schedule();
  ASSERT_FALSE(pf2);
  EXPECT_EQ(b2.size(), 3u);
}

TEST(SchedulerTest, MaxInFlight_DecodesRemainingAfterSeqFinishes) {
  Config config = make_config(64, 8, 256, 0);
  config.max_num_seqs = 16;
  config.max_in_flight_count = 3;
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};

  sched.add_request(prompt(4), {.max_tokens = 1});
  sched.add_request(prompt(4), {.max_tokens = 20});
  sched.add_request(prompt(4), {.max_tokens = 20});

  // Prefill all 3
  auto [b1, pf1] = sched.schedule();
  ASSERT_TRUE(pf1);
  ASSERT_EQ(b1.size(), 3u);
  sched.postprocess(b1, {1, 1, 1});

  // short_seq already finished in postprocess (max_tokens=1, got 1 completion token)
  // running=2, waiting=0. Decode remaining 2.
  auto [b2, pf2] = sched.schedule();
  ASSERT_FALSE(pf2);
  EXPECT_EQ(b2.size(), 2u) << "Only 2 remain after short seq finished during prefill postprocess";
}

TEST(SchedulerTest, MaxInFlight_RefillsAfterBatchDrains) {
  Config config = make_config(128, 8, 256, 0);
  config.max_num_seqs = 16;
  config.max_in_flight_count = 4;
  auto queue = make_queue();
  Scheduler sched{config, queue.get()};

  // 4 short requests then 4 long requests
  for (int i = 0; i < 4; ++i) {
    sched.add_request(prompt(4), {.max_tokens = 2});
  }
  for (int i = 0; i < 4; ++i) {
    sched.add_request(prompt(4), {.max_tokens = 20});
  }

  // Prefill 4 short
  auto [b1, pf1] = sched.schedule();
  ASSERT_TRUE(pf1);
  ASSERT_EQ(b1.size(), 4u);
  sched.postprocess(b1, {1, 1, 1, 1});

  // 4 running, 4 waiting. Prefill wins → prefill 4 long
  auto [b2, pf2] = sched.schedule();
  ASSERT_TRUE(pf2) << "Waiting is non-empty, prefill takes priority";
  ASSERT_EQ(b2.size(), 4u);
  sched.postprocess(b2, {1, 1, 1, 1});

  // 8 running, 0 waiting. Decode 4 (capped by max_in_flight=4)
  auto [b3, pf3] = sched.schedule();
  ASSERT_FALSE(pf3);
  EXPECT_EQ(b3.size(), 4u);
  sched.postprocess(b3, {2, 2, 2, 2});

  // 4 short seqs finished (max_tokens=2), 4 long remain. Decode 4.
  auto [b4, pf4] = sched.schedule();
  ASSERT_FALSE(pf4);
  EXPECT_EQ(b4.size(), 4u) << "Long requests continue decoding at full capacity";
}

}
}
