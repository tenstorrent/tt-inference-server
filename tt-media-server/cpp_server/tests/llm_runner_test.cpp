// SPDX-License-Identifier: Apache-2.0
#include "utils/id_generator.hpp"
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>

#include "config/runner_config.hpp"
#include "ipc/boost_ipc_result_queue.hpp"
#include "runners/llm_runner.hpp"
#include "runners/llm_runner/in_memory_task_queue.hpp"
#include "runners/llm_runner/sequence.hpp"
namespace tt::runners::llm_engine {

using Config = tt::config::LLMConfig;

namespace {

std::shared_ptr<ITaskQueue> makeQueue() {
  return std::make_shared<InMemoryTaskQueue>();
}

Config makeEngineConfig(int numBlocks = 128, int blockSize = 8, int eos = 32) {
  Config c;
  c.num_kvcache_blocks = numBlocks;
  c.kvcache_block_size = blockSize;
  c.eos = eos;
  return c;
}

TEST(LLMRunnerTest, AllTokensPublishedInOrder) {
  setenv("LLM_MODE", "prefill", 1);
  Config config = makeEngineConfig();

  struct Request {
    std::vector<int64_t> prompt;
    int max_tokens;
  };
  std::vector<Request> requests = {
      {{1, 2, 3}, 30},
      {{4, 5, 6, 7}, 10},
      {{7, 8, 9, 10, 11, 12}, 20},
  };

  int totalRequests = static_cast<int>(requests.size());
  auto taskQueue = makeQueue();

  tt::ipc::BoostIpcResultQueue resultQueue("test_llm_runner_tokens",
                                           tt::ipc::RESULT_QUEUE_CAPACITY);

  tt::runners::LLMRunner engine{config, &resultQueue, taskQueue.get()};

  std::vector<uint32_t> taskIds;
  int idCounter = 0;
  for (const auto& req : requests) {
    Sequence& seq = engine.scheduler().addRequest(
        tt::utils::TaskIDGenerator::generate(), req.prompt,
        {.max_tokens = req.max_tokens});
    taskIds.push_back(seq.taskId);
  }

  std::unordered_map<uint32_t, std::vector<int64_t>> receivedTokens;
  std::atomic<int> finishedCount{0};

  std::thread consumer([&]() {
    tt::ipc::SharedToken token;
    while (finishedCount.load() < totalRequests) {
      if (resultQueue.tryPop(token)) {
        uint32_t tid = token.task_id;
        receivedTokens[tid].push_back(static_cast<int64_t>(token.token_id));
        if (token.isFinal()) {
          finishedCount.fetch_add(1);
        }
      }
    }
    engine.stop();
  });

  engine.start();
  consumer.join();

  ASSERT_EQ(finishedCount.load(), totalRequests);

  // 1st published token in the mocked prefill is always whitespace token id=223
  // The followed tokens using the mocked runner are increments of 223
  const std::vector<int64_t> expectedSeq0 = {
      223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
      238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252,
  };
  const std::vector<int64_t> expectedSeq1 = {
      223, 224, 225, 226, 227, 228, 229, 230, 231, 232,
  };
  const std::vector<int64_t> expectedSeq2 = {
      223, 224, 225, 226, 227, 228, 229, 230, 231, 232,
      233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
  };

  EXPECT_EQ(receivedTokens[taskIds[0]], expectedSeq0);
  EXPECT_EQ(receivedTokens[taskIds[1]], expectedSeq1);
  EXPECT_EQ(receivedTokens[taskIds[2]], expectedSeq2);

  resultQueue.shutdown();
  resultQueue.remove();
}

}  // namespace
}  // namespace tt::runners::llm_engine
