// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "llm_engine/engine/sequence.hpp"
#include "llm_engine/sampling_params.hpp"

#include <gtest/gtest.h>
#include <sstream>

namespace llm_engine {
namespace {

TEST(SequenceTest, SerializeDeserialize_RoundTrip_PreservesAllFields) {
  Sequence orig({1, 2, 3, 4, 5}, SamplingParams{.temperature = 0.7f, .max_tokens = 20, .ignore_eos = true});
  orig.num_cached_tokens_ = 256;
  orig.block_table_ = {0, 1};
  orig.status_ = SequenceStatus::WAITING;
  orig.last_token = 5;

  std::ostringstream os;
  orig.serialize(os);
  std::string data = os.str();
  std::istringstream is(data);

  std::unique_ptr<Sequence> restored(Sequence::deserialize(is));
  ASSERT_NE(restored.get(), nullptr);

  EXPECT_EQ(restored->seq_id, orig.seq_id);
  EXPECT_EQ(restored->last_token, orig.last_token);
  EXPECT_EQ(restored->num_prompt_tokens_, orig.num_prompt_tokens_);
  EXPECT_EQ(restored->num_cached_tokens_, orig.num_cached_tokens_);
  EXPECT_EQ(restored->max_tokens, orig.max_tokens);
  EXPECT_EQ(restored->ignore_eos, orig.ignore_eos);
  EXPECT_EQ(restored->token_ids_, orig.token_ids_);
  EXPECT_EQ(restored->block_table_, orig.block_table_);
  EXPECT_FLOAT_EQ(restored->temperature, orig.temperature);
  EXPECT_EQ(restored->status_, orig.status_);
}

TEST(SequenceTest, SerializeDeserialize_EmptyTokenIds) {
  Sequence orig({}, SamplingParams{.max_tokens = 10});
  orig.seq_id = SequenceID();
  orig.last_token = 0;

  std::ostringstream os;
  orig.serialize(os);
  std::istringstream is(os.str());

  std::unique_ptr<Sequence> restored(Sequence::deserialize(is));
  ASSERT_NE(restored.get(), nullptr);
  EXPECT_EQ(restored->seq_id, orig.seq_id);
  EXPECT_TRUE(restored->token_ids_.empty());
  EXPECT_EQ(restored->num_prompt_tokens_, 0u);
  EXPECT_EQ(restored->last_token, 0);
}

TEST(SequenceTest, SerializeDeserialize_AfterAppendToken) {
  Sequence orig({10, 20}, SamplingParams{.max_tokens = 5});
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
