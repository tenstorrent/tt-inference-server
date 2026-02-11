// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "llm_engine/config.hpp"
#include "llm_engine/engine/llm_engine.hpp"
#include <gtest/gtest.h>
#include <map>
#include <vector>
#include "llm_engine/engine/in_memory_task_queue.hpp"
namespace llm_engine {
namespace {
  
std::unique_ptr<ITaskQueue> make_queue() {
  return std::make_unique<InMemoryTaskQueue>();
}
  
std::unique_ptr<Scheduler> make_scheduler(const Config& config) {
  return std::make_unique<Scheduler>(config, make_queue());
}

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
  
  auto scheduler = make_scheduler(config);

  LLMEngine engine{config, [&](int seq_id, int64_t token_id, bool finished) {
      received_tokens[seq_id].push_back(token_id);
      if (finished && ++finished_count == total_requests) {
        engine.stop();
      }
    }, std::move(scheduler)};

  std::vector<int> seq_ids;
  for (const auto& req : requests) {
    Sequence& seq =
        engine.scheduler().add_request(req.prompt, {.max_tokens = req.max_tokens});
    seq_ids.push_back(seq.seq_id);
  }

  engine.run();

  ASSERT_EQ(finished_count, total_requests);

  const std::vector<int64_t> expected_seq0 = {
      4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
      19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
  };
  const std::vector<int64_t> expected_seq1 = {
      8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  };
  const std::vector<int64_t> expected_seq2 = {
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
      23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
  };

  EXPECT_EQ(received_tokens[seq_ids[0]], expected_seq0);
  EXPECT_EQ(received_tokens[seq_ids[1]], expected_seq1);
  EXPECT_EQ(received_tokens[seq_ids[2]], expected_seq2);
}

}  // namespace
}  // namespace llm_engine
