// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "runtime/runners/llm_runner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>

#include "config/runner_config.hpp"
#include "config/settings.hpp"
#include "domain/llm/sampling_params.hpp"
#include "domain/llm/sequence.hpp"
#include "domain/response_format.hpp"
#include "ipc/boost/boost_result_queue.hpp"
#include "ipc/in_memory/in_memory_task_queue.hpp"
#include "utils/id_generator.hpp"
#include "utils/tokenizers/tokenizer.hpp"
namespace tt::runners::llm_engine {

using namespace tt::domain::llm;
using tt::ipc::in_memory::TaskQueue;

using Config = tt::config::LLMConfig;

namespace {

std::shared_ptr<tt::ipc::ITaskQueue> makeQueue() {
  return std::make_shared<TaskQueue>();
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

  tt::ipc::boost::ResultQueue resultQueue("test_llm_runner_tokens",
                                          tt::config::resultQueueCapacity());

  tt::runners::LLMRunner engine{config, &resultQueue, taskQueue.get()};

  std::vector<uint32_t> taskIds;
  int idCounter = 0;
  for (const auto& req : requests) {
    tt::domain::llm::Sequence& seq = engine.getScheduler().addRequest(
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

  // Mock runner generates: <think_start> + 10 think tokens + <think_end> +
  // visible tokens. Think/visible tokens alternate with whitespace (223).
  // Think content = 77291, visible content = 15329.
  constexpr int64_t kThinkStart = 128798;
  constexpr int64_t kThinkEnd = 128799;
  constexpr int64_t kThinkContent = 77291;
  constexpr int64_t kVisibleContent = 15329;
  constexpr int64_t kWhitespace = 223;

  // 30 tokens: think_start + 10 think + think_end + 18 visible
  const std::vector<int64_t> expectedSeq0 = {
      kThinkStart,
      kThinkContent, kWhitespace, kThinkContent, kWhitespace, kThinkContent,
      kWhitespace, kThinkContent, kWhitespace, kThinkContent, kWhitespace,
      kThinkEnd,
      kVisibleContent, kWhitespace, kVisibleContent, kWhitespace,
      kVisibleContent, kWhitespace, kVisibleContent, kWhitespace,
      kVisibleContent, kWhitespace, kVisibleContent, kWhitespace,
      kVisibleContent, kWhitespace, kVisibleContent, kWhitespace,
      kVisibleContent, kWhitespace,
  };
  // 10 tokens: think_start + 9 think
  const std::vector<int64_t> expectedSeq1 = {
      kThinkStart,
      kThinkContent, kWhitespace, kThinkContent, kWhitespace, kThinkContent,
      kWhitespace, kThinkContent, kWhitespace, kThinkContent,
  };
  // 20 tokens: think_start + 10 think + think_end + 8 visible
  const std::vector<int64_t> expectedSeq2 = {
      kThinkStart,
      kThinkContent, kWhitespace, kThinkContent, kWhitespace, kThinkContent,
      kWhitespace, kThinkContent, kWhitespace, kThinkContent, kWhitespace,
      kThinkEnd,
      kVisibleContent, kWhitespace, kVisibleContent, kWhitespace,
      kVisibleContent, kWhitespace, kVisibleContent, kWhitespace,
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
  tt::ipc::boost::ResultQueue resultQueue("test_structured_output",
                                          tt::config::resultQueueCapacity());
  tt::runners::LLMRunner engine{config, &resultQueue, taskQueue.get()};

  tt::domain::llm::SamplingParams sp;
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

  ASSERT_FALSE(tokens.empty());
  EXPECT_LT(static_cast<int>(tokens.size()), 100)
      << "Structured output should complete via grammar, not max_tokens";
  const auto& stopIds = tt::utils::tokenizers::activeTokenizer().stopTokenIds();
  EXPECT_NE(std::find(stopIds.begin(), stopIds.end(), tokens.back()),
            stopIds.end())
      << "Sequence should terminate on EOS, not via grammar rejection";

  resultQueue.shutdown();
  resultQueue.remove();
}

}  // namespace
}  // namespace tt::runners::llm_engine
