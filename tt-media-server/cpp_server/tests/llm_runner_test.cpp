// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "runners/llm_runner.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/response_format.hpp"
#include "domain/sampling_params.hpp"
#include "domain/sequence.hpp"
#include "ipc/boost_ipc_result_queue.hpp"
#include "runners/llm_runner/in_memory_task_queue.hpp"
#include "utils/id_generator.hpp"
#include "utils/tokenizers/tokenizer.hpp"
namespace tt::runners::llm_engine {

using Config = tt::config::LLMConfig;

namespace {

std::shared_ptr<tt::ipc::ITaskQueue> makeQueue() {
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
    tt::domain::Sequence& seq = engine.getScheduler().addRequest(
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

// With a json_schema response format the mock runner drives the grammar to
// completion. The sequence must finish via grammar termination (finish_reason
// "stop"), not by exhausting max_tokens.  For {"x": integer} the grammar
// completes in ~8 tokens, well under the 100-token budget.
TEST(LLMRunnerTest, StructuredOutputCompletesBeforeMaxTokens) {
  setenv("LLM_MODE", "prefill", 1);

  // Load stop tokens before constructing the engine so the scheduler can
  // recognise the EOS token and mark the sequence finished without dangling
  // pointers in the decode queue.
  Config config = makeEngineConfig();
  for (int64_t id : tt::utils::tokenizers::activeTokenizer().stopTokenIds()) {
    config.stop_token_ids.push_back(id);
  }
  auto taskQueue = makeQueue();
  tt::ipc::BoostIpcResultQueue resultQueue("test_structured_output",
                                           tt::ipc::RESULT_QUEUE_CAPACITY);
  tt::runners::LLMRunner engine{config, &resultQueue, taskQueue.get()};

  tt::domain::SamplingParams sp;
  sp.max_tokens = 100;
  sp.response_format_type = tt::domain::ResponseFormatType::JSON_SCHEMA;
  sp.json_schema_str =
      R"({"type":"object","properties":{"x":{"type":"integer"}})"
      R"(,"required":["x"],"additionalProperties":false})";

  auto& seq = engine.getScheduler().addRequest(
      tt::utils::TaskIDGenerator::generate(), {1, 2, 3}, sp);
  uint32_t taskId = seq.taskId;

  std::vector<int64_t> tokens;
  std::atomic<bool> done{false};

  std::thread consumer([&]() {
    tt::ipc::SharedToken token;
    while (!done.load()) {
      if (resultQueue.tryPop(token) && token.task_id == taskId) {
        tokens.push_back(static_cast<int64_t>(token.token_id));
        if (token.isFinal()) {
          done.store(true);
          engine.stop();
        }
      }
    }
  });

  engine.start();
  consumer.join();

  EXPECT_FALSE(tokens.empty());
  // Grammar for {"x": N} completes in ~8 tokens; hitting 100 would indicate
  // the sequence ran out of budget rather than terminating grammatically.
  EXPECT_LT(static_cast<int>(tokens.size()), 100)
      << "Structured output should complete via grammar, not max_tokens";

  resultQueue.shutdown();
  resultQueue.remove();
}

}  // namespace
}  // namespace tt::runners::llm_engine
