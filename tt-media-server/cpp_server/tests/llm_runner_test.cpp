// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "llm_engine/config.hpp"
#include "runners/llm_runner.hpp"
#include "llm_engine/engine/sequence.hpp"
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>
#include "llm_engine/engine/in_memory_task_queue.hpp"
namespace llm_engine {
namespace {
  
std::shared_ptr<ITaskQueue> make_queue() {
  return std::make_shared<InMemoryTaskQueue>();
}

std::unique_ptr<Scheduler> make_scheduler(const Config& config,
                                           ITaskQueue* task_queue) {
  return std::make_unique<Scheduler>(config, task_queue);
}

Config make_engine_config(int num_blocks = 128, int block_size = 8,
                          int eos = 0) {
  Config c;
  c.num_kvcache_blocks = num_blocks;
  c.kvcache_block_size = block_size;
  c.eos = eos;
  return c;
}

TEST(LLMRunnerTest, AllTokensPublishedInOrder) {
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

  std::unordered_map<TaskID, std::vector<int64_t>> received_tokens;
  int finished_count = 0;
  int total_requests = static_cast<int>(requests.size());

  auto task_queue = make_queue();
  auto scheduler = make_scheduler(config, task_queue.get());

  LLMRunner engine{config, [&](TaskID task_id, int64_t token_id, bool finished) {
      received_tokens[task_id].push_back(token_id);
      if (finished && ++finished_count == total_requests) {
        engine.stop();
      }
    }, std::move(scheduler)};

  std::vector<TaskID> task_ids;
  for (const auto& req : requests) {
    Sequence& seq =
        engine.scheduler().add_request(req.prompt, {.max_tokens = req.max_tokens});
    task_ids.push_back(seq.task_id);
  }

  engine.run();

  ASSERT_EQ(finished_count, total_requests);

  // 1st published token in the mocked prefill is always whitespace token id=223
  // The followed tokens using the mocked runner are increments of 223
  const std::vector<int64_t> expected_seq0 = {
    223, 224,225, 226, 227, 
    228, 229, 230, 231, 232, 
    233, 234, 235, 236, 237, 
    238, 239, 240, 241, 242, 
    243, 244, 245, 246, 247, 
    248, 249, 250, 251, 252,
  };
  const std::vector<int64_t> expected_seq1 = {
    223, 224,225, 226, 227, 
    228, 229, 230, 231, 232, 
  };
  const std::vector<int64_t> expected_seq2 = {
    223, 224,225, 226, 227, 
    228, 229, 230, 231, 232, 
    233, 234, 235, 236, 237, 
    238, 239, 240, 241, 242, 
  };

  EXPECT_EQ(received_tokens[task_ids[0]], expected_seq0);
  EXPECT_EQ(received_tokens[task_ids[1]], expected_seq1);
  EXPECT_EQ(received_tokens[task_ids[2]], expected_seq2);
}

}  // namespace
}  // namespace llm_engine
