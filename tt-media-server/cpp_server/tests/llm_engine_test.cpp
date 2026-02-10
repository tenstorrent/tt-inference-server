// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "llm_engine/config.hpp"
#include "llm_engine/engine/llm_engine.hpp"
#include <gtest/gtest.h>
#include <map>
#include <vector>

namespace llm_engine {
namespace {

Config make_engine_config(int num_blocks = 128, int block_size = 8,
                          int eos = 0) {
  Config c;
  c.num_kvcache_blocks = num_blocks;
  c.kvcache_block_size = block_size;
  c.eos = eos;
  return c;
}

TEST(LLMEngineTest, AllTokensPublishedInOrder) {
  Config config = make_engine_config();

  struct Request {
    std::vector<int64_t> prompt;
    int max_tokens;
  };
  std::vector<Request> requests = {
      {{1, 2, 3}, 30},
      {{4, 5, 6, 7}, 10},
      {{7, 8, 9, 10, 11, 12}, 20},
  };

  std::map<int, std::vector<int64_t>> received_tokens;
  int finished_count = 0;
  int total_requests = static_cast<int>(requests.size());

  LLMEngine engine{config,
    [&](int seq_id, int64_t token_id, bool finished) {
      received_tokens[seq_id].push_back(token_id);
      if (finished && ++finished_count == total_requests) {
        engine.stop();
      }
    }};

  std::vector<int> seq_ids;
  for (const auto& req : requests) {
    Sequence& seq =
        engine.scheduler().add_request(req.prompt, {.max_tokens = req.max_tokens});
    seq_ids.push_back(seq.seq_id);
  }

  engine.run();

  ASSERT_EQ(finished_count, total_requests);

  for (size_t i = 0; i < requests.size(); ++i) {
    int sid = seq_ids[i];
    int64_t last_prompt_token = requests[i].prompt.back();
    int max_tok = requests[i].max_tokens;

    ASSERT_EQ(static_cast<int>(received_tokens[sid].size()), max_tok)
        << "seq " << sid << " expected " << max_tok << " tokens";

    for (int t = 0; t < max_tok; ++t) {
      int64_t expected = last_prompt_token + 1 + t;
      EXPECT_EQ(received_tokens[sid][t], expected)
          << "seq " << sid << " token[" << t << "]";
    }
  }
}

}  // namespace
}  // namespace llm_engine
