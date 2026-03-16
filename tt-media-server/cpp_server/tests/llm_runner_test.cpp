// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "config/runner_config.hpp"
#include "runners/llm_runner.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "ipc/shared_memory.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>
#include "runners/llm_runner/in_memory_task_queue.hpp"
namespace llm_engine {

using Config = tt::config::LLMConfig;

namespace {

std::shared_ptr<ITaskQueue> make_queue() {
  return std::make_shared<InMemoryTaskQueue>();
}

Config make_engine_config(int num_blocks = 128, int block_size = 8,
                          int eos = 32) {
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

  int total_requests = static_cast<int>(requests.size());
  auto task_queue = make_queue();

  tt::ipc::TokenRingBuffer<65536> result_queue("/test_llm_runner_tokens", true);

  tt::runners::LLMRunner engine{config, &result_queue, task_queue.get()};

  std::vector<TaskID> task_ids;
  int id_counter = 0;
  for (const auto& req : requests) {
    Sequence& seq =
        engine.scheduler().add_request(std::move(TaskID(TaskID::generate())), req.prompt, {.max_tokens = req.max_tokens});
    task_ids.push_back(seq.task_id);
  }

  std::unordered_map<TaskID, std::vector<int64_t>> received_tokens;
  std::atomic<int> finished_count{0};

  std::thread consumer([&]() {
    tt::ipc::SharedToken token;
    while (finished_count.load() < total_requests) {
      if (result_queue.pop(token)) {
        TaskID tid(TaskID(std::string(token.task_id)));
        tid.id = std::string(token.task_id);
        received_tokens[tid].push_back(static_cast<int64_t>(token.token_id));
        if (token.is_final()) {
          finished_count.fetch_add(1);
        }
      }
    }
    engine.stop();
  });

  engine.start();
  consumer.join();

  ASSERT_EQ(finished_count.load(), total_requests);

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

  result_queue.shutdown();
}

}  // namespace
}  // namespace llm_engine
