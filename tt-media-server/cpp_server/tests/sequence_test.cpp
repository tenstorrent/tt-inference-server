// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/sampling_params.hpp"

#include <gtest/gtest.h>
#include <sstream>

namespace llm_engine {
namespace {

TEST(SequenceIDTest, SerializeDeserialize_RoundTrip) {
  TaskID orig;

  std::vector<char> buf = orig.serialize();
  ASSERT_EQ(buf.size(), 36);

  TaskID restored = TaskID::deserialize(buf.data(), buf.size());
  EXPECT_EQ(restored.id, orig.id);
}

TEST(SamplingParamsTest, SerializeDeserialize_DefaultParams) {
  SamplingParams orig;

  std::ostringstream os;
  orig.serialize(os);
  std::istringstream is(os.str());
  std::unique_ptr<SamplingParams> restored(SamplingParams::deserialize(is));
  ASSERT_NE(restored.get(), nullptr);

  EXPECT_FLOAT_EQ(restored->temperature, orig.temperature);
  EXPECT_EQ(restored->max_tokens, orig.max_tokens);
  EXPECT_EQ(restored->ignore_eos, orig.ignore_eos);
  EXPECT_EQ(restored->top_p, orig.top_p);
  EXPECT_FLOAT_EQ(restored->presence_penalty, orig.presence_penalty);
  EXPECT_FLOAT_EQ(restored->frequency_penalty, orig.frequency_penalty);
  EXPECT_EQ(restored->seed, orig.seed);
  EXPECT_EQ(restored->use_beam_search, orig.use_beam_search);
  EXPECT_EQ(restored->top_k, orig.top_k);
  EXPECT_EQ(restored->min_p, orig.min_p);
  EXPECT_EQ(restored->repetition_penalty, orig.repetition_penalty);
  EXPECT_FLOAT_EQ(restored->length_penalty, orig.length_penalty);
  EXPECT_EQ(restored->stop_token_ids, orig.stop_token_ids);
  EXPECT_EQ(restored->include_stop_str_in_output, orig.include_stop_str_in_output);
  EXPECT_EQ(restored->min_tokens, orig.min_tokens);
  EXPECT_EQ(restored->skip_special_tokens, orig.skip_special_tokens);
  EXPECT_EQ(restored->spaces_between_special_tokens, orig.spaces_between_special_tokens);
  EXPECT_EQ(restored->allowed_token_ids, orig.allowed_token_ids);
  EXPECT_EQ(restored->prompt_logprobs, orig.prompt_logprobs);
  EXPECT_EQ(restored->truncate_prompt_tokens, orig.truncate_prompt_tokens);
}

TEST(SamplingParamsTest, SerializeDeserialize_AllOptionalFieldsSet) {
  SamplingParams orig;
  orig.temperature = 0.8f;
  orig.max_tokens = 512;
  orig.ignore_eos = true;
  orig.top_p = 0.9f;
  orig.presence_penalty = 0.1f;
  orig.frequency_penalty = 0.2f;
  orig.seed = 42;
  orig.use_beam_search = true;
  orig.top_k = 50;
  orig.min_p = 0.05f;
  orig.repetition_penalty = 1.2f;
  orig.length_penalty = 0.9f;
  orig.stop_token_ids = {1, 2, 3};
  orig.include_stop_str_in_output = true;
  orig.min_tokens = 10;
  orig.skip_special_tokens = false;
  orig.spaces_between_special_tokens = false;
  orig.allowed_token_ids = {100, 200, 300};
  orig.prompt_logprobs = 5;
  orig.truncate_prompt_tokens = 128;

  std::ostringstream os;
  orig.serialize(os);
  std::istringstream is(os.str());
  std::unique_ptr<SamplingParams> restored(SamplingParams::deserialize(is));
  ASSERT_NE(restored.get(), nullptr);

  EXPECT_FLOAT_EQ(restored->temperature, orig.temperature);
  EXPECT_EQ(restored->max_tokens, orig.max_tokens);
  EXPECT_EQ(restored->ignore_eos, orig.ignore_eos);
  ASSERT_TRUE(restored->top_p.has_value());
  EXPECT_FLOAT_EQ(*restored->top_p, *orig.top_p);
  EXPECT_FLOAT_EQ(restored->presence_penalty, orig.presence_penalty);
  EXPECT_FLOAT_EQ(restored->frequency_penalty, orig.frequency_penalty);
  ASSERT_TRUE(restored->seed.has_value());
  EXPECT_EQ(*restored->seed, *orig.seed);
  EXPECT_EQ(restored->use_beam_search, orig.use_beam_search);
  ASSERT_TRUE(restored->top_k.has_value());
  EXPECT_EQ(*restored->top_k, *orig.top_k);
  ASSERT_TRUE(restored->min_p.has_value());
  EXPECT_FLOAT_EQ(*restored->min_p, *orig.min_p);
  ASSERT_TRUE(restored->repetition_penalty.has_value());
  EXPECT_FLOAT_EQ(*restored->repetition_penalty, *orig.repetition_penalty);
  EXPECT_FLOAT_EQ(restored->length_penalty, orig.length_penalty);
  EXPECT_EQ(restored->stop_token_ids, orig.stop_token_ids);
  EXPECT_EQ(restored->include_stop_str_in_output, orig.include_stop_str_in_output);
  EXPECT_EQ(restored->min_tokens, orig.min_tokens);
  EXPECT_EQ(restored->skip_special_tokens, orig.skip_special_tokens);
  EXPECT_EQ(restored->spaces_between_special_tokens, orig.spaces_between_special_tokens);
  ASSERT_TRUE(restored->allowed_token_ids.has_value());
  EXPECT_EQ(*restored->allowed_token_ids, *orig.allowed_token_ids);
  ASSERT_TRUE(restored->prompt_logprobs.has_value());
  EXPECT_EQ(*restored->prompt_logprobs, *orig.prompt_logprobs);
  ASSERT_TRUE(restored->truncate_prompt_tokens.has_value());
  EXPECT_EQ(*restored->truncate_prompt_tokens, *orig.truncate_prompt_tokens);
}

TEST(SequenceTest, SerializeDeserialize_RoundTrip_PreservesAllFields) {
  SamplingParams params;
  params.temperature = 0.7f;
  params.max_tokens = 20;
  params.ignore_eos = true;
  params.top_p = 0.95f;
  params.seed = 7;
  params.stop_token_ids = {10, 20};
  params.allowed_token_ids = {1, 2, 3};

  Sequence orig(256, {1, 2, 3, 4, 5}, params);
  orig.num_cached_tokens_ = 256;
  orig.block_table_ = {0, 1};
  orig.status_ = SequenceStatus::IN_FLIGHT;
  orig.last_token = 5;

  std::ostringstream os;
  orig.serialize(os);
  std::istringstream is(os.str());

  std::unique_ptr<Sequence> restored(Sequence::deserialize(is));
  ASSERT_NE(restored.get(), nullptr);

  EXPECT_EQ(restored->task_id, orig.task_id);
  EXPECT_EQ(restored->last_token, orig.last_token);
  EXPECT_EQ(restored->num_prompt_tokens_, orig.num_prompt_tokens_);
  EXPECT_EQ(restored->num_cached_tokens_, orig.num_cached_tokens_);
  EXPECT_EQ(restored->token_ids_, orig.token_ids_);
  EXPECT_EQ(restored->block_table_, orig.block_table_);
  EXPECT_EQ(restored->status_, orig.status_);

  const auto& sp = *restored->sampling_params;
  const auto& sp_orig = *orig.sampling_params;
  EXPECT_FLOAT_EQ(sp.temperature, sp_orig.temperature);
  EXPECT_EQ(sp.max_tokens, sp_orig.max_tokens);
  EXPECT_EQ(sp.ignore_eos, sp_orig.ignore_eos);
  ASSERT_TRUE(sp.top_p.has_value());
  EXPECT_FLOAT_EQ(*sp.top_p, *sp_orig.top_p);
  ASSERT_TRUE(sp.seed.has_value());
  EXPECT_EQ(*sp.seed, *sp_orig.seed);
  EXPECT_EQ(sp.stop_token_ids, sp_orig.stop_token_ids);
  ASSERT_TRUE(sp.allowed_token_ids.has_value());
  EXPECT_EQ(*sp.allowed_token_ids, *sp_orig.allowed_token_ids);
}

TEST(SequenceTest, SerializeDeserialize_EmptyTokenIds) {
  Sequence orig(256, {}, SamplingParams{.max_tokens = 10});
  orig.task_id = TaskID();
  orig.last_token = 0;

  std::ostringstream os;
  orig.serialize(os);
  std::istringstream is(os.str());

  std::unique_ptr<Sequence> restored(Sequence::deserialize(is));
  ASSERT_NE(restored.get(), nullptr);
  EXPECT_EQ(restored->task_id, orig.task_id);
  EXPECT_TRUE(restored->token_ids_.empty());
  EXPECT_EQ(restored->num_prompt_tokens_, 0u);
  EXPECT_EQ(restored->last_token, 0);
}

TEST(SequenceTest, SerializeDeserialize_AfterAppendToken) {
  Sequence orig(256, {10, 20}, SamplingParams{.max_tokens = 5});
  orig.append_token(30);
  orig.append_token(40);
  orig.num_cached_tokens_ = 256;

  std::ostringstream os;
  orig.serialize(os);
  std::istringstream is(os.str());

  std::unique_ptr<Sequence> restored(Sequence::deserialize(is));
  ASSERT_NE(restored.get(), nullptr);
  EXPECT_EQ(restored->size(), 4u);
  EXPECT_EQ((*restored)[0], 10);
  EXPECT_EQ((*restored)[1], 20);
  EXPECT_EQ((*restored)[2], 30);
  EXPECT_EQ((*restored)[3], 40);
  EXPECT_EQ(restored->last_token, 40);
  EXPECT_EQ(restored->num_prompt_tokens_, 2u);
  EXPECT_EQ(restored->num_cached_tokens_, 256u);
}

}  // namespace
}  // namespace llm_engine
