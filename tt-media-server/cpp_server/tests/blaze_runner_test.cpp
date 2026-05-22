// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_runner.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "config/runner_config.hpp"
#include "config/settings.hpp"
#include "domain/llm/sampling_params.hpp"
#include "domain/llm/sequence.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/boost/boost_memory_queue.hpp"
#include "ipc/in_memory/in_memory_cancel_queue.hpp"
#include "ipc/in_memory/in_memory_result_queue.hpp"
#include "ipc/in_memory/in_memory_task_queue.hpp"

namespace tt::runners::blaze {

namespace {

constexpr auto DEADLINE = std::chrono::seconds(10);
constexpr auto POLL_INTERVAL = std::chrono::milliseconds(50);
constexpr uint64_t MOCK_PIPELINE_TOKEN_ID = 12345u;
constexpr int DEFAULT_BLOCK_SIZE = 1;
const std::vector<int64_t> DEFAULT_STOP_TOKEN_IDS = {987654321};

std::string makeQueueSuffix(std::string_view label) {
  const auto ticks =
      std::chrono::steady_clock::now().time_since_epoch().count();
  return std::to_string(::getpid()) + "_" + std::string(label) + "_" +
         std::to_string(ticks);
}

class BlazeRunnerHarness {
 public:
  explicit BlazeRunnerHarness(
      std::string_view label,
      std::vector<int64_t> stopTokenIds = DEFAULT_STOP_TOKEN_IDS)
      : memoryReqQueueName("test_blaze_mem_req_" + makeQueueSuffix(label)),
        memoryResQueueName("test_blaze_mem_res_" + makeQueueSuffix(label)) {
    setenv("TT_MEMORY_REQUEST_QUEUE", memoryReqQueueName.c_str(), 1);
    setenv("TT_MEMORY_RESULT_QUEUE", memoryResQueueName.c_str(), 1);

    // Remove any existing queues from previous test runs.
    tt::ipc::boost::MemoryRequestQueue::remove(memoryReqQueueName);
    tt::ipc::boost::MemoryResultQueue::remove(memoryResQueueName);

    memoryRequestQueue = std::make_unique<tt::ipc::boost::MemoryRequestQueue>(
        memoryReqQueueName,
        static_cast<int>(tt::config::memoryQueueCapacity()));
    memoryResultQueue = std::make_unique<tt::ipc::boost::MemoryResultQueue>(
        memoryResQueueName,
        static_cast<int>(tt::config::memoryQueueCapacity()));

    config.runner_type = tt::config::ModelRunnerType::MOCK_PIPELINE;
    config.stop_token_ids = std::move(stopTokenIds);
    runner = std::make_unique<BlazeRunner>(config, &resultQueue, &taskQueue,
                                           &cancelQueue);

    runnerThread = std::thread([this]() {
      try {
        runner->start();
      } catch (...) {
        runnerError = std::current_exception();
      }
    });
  }

  ~BlazeRunnerHarness() { shutdown(); }

  tt::domain::ManageMemoryResult allocate(uint32_t taskId) {
    tt::domain::ManageMemoryTask request{};
    request.taskId = taskId;
    request.action = tt::domain::MemoryManagementAction::ALLOCATE;
    memoryRequestQueue->push(request);

    tt::domain::ManageMemoryResult response{};
    memoryResultQueue->receive(response);
    return response;
  }

  void submitSequence(uint32_t taskId, uint32_t slotId,
                      const std::vector<int64_t>& promptTokens,
                      const tt::domain::llm::SamplingParams& samplingParams) {
    tt::domain::llm::Sequence seq(taskId, DEFAULT_BLOCK_SIZE, promptTokens,
                                  samplingParams);
    seq.setKVCacheSlot(slotId);
    taskQueue.push(seq);
  }

  bool waitPopFor(tt::ipc::SharedToken& token) {
    return resultQueue.waitPopFor(token, POLL_INTERVAL);
  }

  std::vector<tt::ipc::SharedToken> collectTaskTokensUntilFinal(
      uint32_t taskId) {
    std::vector<tt::ipc::SharedToken> tokens;
    const auto deadline = std::chrono::steady_clock::now() + DEADLINE;
    while (std::chrono::steady_clock::now() < deadline) {
      tt::ipc::SharedToken token{};
      if (!waitPopFor(token) || token.task_id != taskId) {
        continue;
      }
      tokens.push_back(token);
      if (token.isFinal()) {
        break;
      }
    }
    return tokens;
  }

  void requestCancel(uint32_t taskId) { cancelQueue.push(taskId); }

  void assertRunnerHealthy() const {
    if (runnerError) {
      std::rethrow_exception(runnerError);
    }
  }

 private:
  void shutdown() {
    if (isShutdown) {
      return;
    }
    isShutdown = true;
    if (runner) {
      runner->stop();
    }
    if (runnerThread.joinable()) {
      runnerThread.join();
    }
    resultQueue.shutdown();
    memoryRequestQueue.reset();
    memoryResultQueue.reset();
    tt::ipc::boost::MemoryRequestQueue::remove(memoryReqQueueName);
    tt::ipc::boost::MemoryResultQueue::remove(memoryResQueueName);
  }

  std::string memoryReqQueueName;
  std::string memoryResQueueName;
  std::unique_ptr<tt::ipc::boost::MemoryRequestQueue> memoryRequestQueue;
  std::unique_ptr<tt::ipc::boost::MemoryResultQueue> memoryResultQueue;
  tt::ipc::in_memory::ResultQueue resultQueue;
  tt::ipc::in_memory::TaskQueue taskQueue;
  tt::ipc::in_memory::CancelQueue cancelQueue;
  tt::config::LLMConfig config{};
  std::unique_ptr<BlazeRunner> runner;
  std::thread runnerThread;
  std::exception_ptr runnerError;
  bool isShutdown = false;
};

}  // namespace

TEST(BlazeRunnerIntegrationTest, InMemoryQueuesRoundTripThroughSimulator) {
  BlazeRunnerHarness harness("blaze_integration");

  const uint32_t taskId = 4242;
  const auto allocateResponse = harness.allocate(taskId);
  ASSERT_EQ(allocateResponse.taskId, taskId);
  ASSERT_EQ(allocateResponse.status, tt::domain::ManageMemoryStatus::SUCCESS);
  ASSERT_NE(allocateResponse.slotId, tt::domain::INVALID_SLOT_ID);

  tt::domain::llm::SamplingParams samplingParams;
  samplingParams.max_tokens = 3;
  samplingParams.ignore_eos = false;

  harness.submitSequence(taskId, allocateResponse.slotId, {11, 22, 33},
                         samplingParams);
  const auto producedTokens = harness.collectTaskTokensUntilFinal(taskId);
  harness.assertRunnerHealthy();

  ASSERT_FALSE(producedTokens.empty())
      << "Expected BlazeRunner to emit at least one token";
  EXPECT_TRUE(producedTokens.back().isFinal())
      << "Expected BlazeRunner to emit a final token";
  EXPECT_EQ(producedTokens.size(), samplingParams.max_tokens.value());

  for (const auto& token : producedTokens) {
    EXPECT_EQ(token.task_id, taskId);

    // Mock simulator is configured to emit token 12345 for all tasks.
    EXPECT_EQ(token.token_id, MOCK_PIPELINE_TOKEN_ID);
  }
}

TEST(BlazeRunnerIntegrationTest, CancelFlowEmitsAbortToken) {
  BlazeRunnerHarness harness("blaze_cancel");

  const uint32_t taskId = 5252;
  const auto allocateResponse = harness.allocate(taskId);
  ASSERT_EQ(allocateResponse.taskId, taskId);
  ASSERT_EQ(allocateResponse.status, tt::domain::ManageMemoryStatus::SUCCESS);
  ASSERT_NE(allocateResponse.slotId, tt::domain::INVALID_SLOT_ID);

  tt::domain::llm::SamplingParams samplingParams;
  samplingParams.max_tokens = 1000;
  samplingParams.ignore_eos = true;

  harness.submitSequence(taskId, allocateResponse.slotId, {44, 55, 66},
                         samplingParams);

  bool sawInitialToken = false;
  const auto tokenDeadline = std::chrono::steady_clock::now() + DEADLINE;
  while (std::chrono::steady_clock::now() < tokenDeadline) {
    tt::ipc::SharedToken token{};
    if (!harness.waitPopFor(token)) {
      continue;
    }
    if (token.task_id != taskId) {
      continue;
    }
    if (!token.isAbort()) {
      sawInitialToken = true;
      break;
    }
  }
  ASSERT_TRUE(sawInitialToken)
      << "Expected at least one generated token before cancellation";

  harness.requestCancel(taskId);

  bool sawAbort = false;
  tt::ipc::SharedToken abortToken{};
  const auto abortDeadline = std::chrono::steady_clock::now() + DEADLINE;
  while (std::chrono::steady_clock::now() < abortDeadline) {
    tt::ipc::SharedToken token{};
    if (!harness.waitPopFor(token)) {
      continue;
    }
    if (token.task_id != taskId) {
      continue;
    }
    if (token.isAbort()) {
      abortToken = token;
      sawAbort = true;
      break;
    }
  }
  harness.assertRunnerHealthy();

  ASSERT_TRUE(sawAbort) << "Expected abort token after cancel request";
  EXPECT_EQ(abortToken.task_id, taskId);
  EXPECT_TRUE(abortToken.isAbort());
  EXPECT_EQ(abortToken.token_id, 0u);
}

TEST(BlazeRunnerIntegrationTest, StopsOnConfiguredStopToken) {
  // Mock simulator emits token_id=12345; this forces stop-token completion.
  BlazeRunnerHarness harness("blaze_stop_token", {MOCK_PIPELINE_TOKEN_ID});

  const uint32_t taskId = 6262;
  const auto allocateResponse = harness.allocate(taskId);
  ASSERT_EQ(allocateResponse.taskId, taskId);
  ASSERT_EQ(allocateResponse.status, tt::domain::ManageMemoryStatus::SUCCESS);
  ASSERT_NE(allocateResponse.slotId, tt::domain::INVALID_SLOT_ID);

  tt::domain::llm::SamplingParams samplingParams;
  samplingParams.max_tokens = 1000;
  samplingParams.ignore_eos = false;

  harness.submitSequence(taskId, allocateResponse.slotId, {77}, samplingParams);
  const auto producedTokens = harness.collectTaskTokensUntilFinal(taskId);
  harness.assertRunnerHealthy();

  ASSERT_FALSE(producedTokens.empty())
      << "Expected at least one token for stop-token test";
  ASSERT_TRUE(producedTokens.back().isFinal())
      << "Expected stop token to end generation";
  EXPECT_EQ(producedTokens.size(), 1u)
      << "Stop token should finalize immediately in mock pipeline";
  EXPECT_EQ(producedTokens.front().task_id, taskId);
  EXPECT_EQ(producedTokens.front().token_id, MOCK_PIPELINE_TOKEN_ID);
  EXPECT_FALSE(producedTokens.front().isAbort());
}

TEST(BlazeRunnerIntegrationTest, TwoConcurrentTasksStayIsolated) {
  BlazeRunnerHarness harness("blaze_two_task_isolation");

  const uint32_t taskA = 7001;
  const uint32_t taskB = 7002;

  uint32_t slotA = tt::domain::INVALID_SLOT_ID;
  uint32_t slotB = tt::domain::INVALID_SLOT_ID;

  const auto responseA = harness.allocate(taskA);
  const auto responseB = harness.allocate(taskB);
  for (const auto& response : {responseA, responseB}) {
    ASSERT_EQ(response.status, tt::domain::ManageMemoryStatus::SUCCESS);
    ASSERT_NE(response.slotId, tt::domain::INVALID_SLOT_ID);
    if (response.taskId == taskA) {
      slotA = response.slotId;
    } else if (response.taskId == taskB) {
      slotB = response.slotId;
    } else {
      FAIL() << "Unexpected taskId in allocate response: " << response.taskId;
    }
  }

  ASSERT_NE(slotA, tt::domain::INVALID_SLOT_ID);
  ASSERT_NE(slotB, tt::domain::INVALID_SLOT_ID);

  tt::domain::llm::SamplingParams samplingParams;
  samplingParams.max_tokens = 3;
  samplingParams.ignore_eos = false;

  harness.submitSequence(taskA, slotA, {11, 12}, samplingParams);
  harness.submitSequence(taskB, slotB, {21, 22}, samplingParams);

  std::vector<tt::ipc::SharedToken> tokensA;
  std::vector<tt::ipc::SharedToken> tokensB;
  bool sawFinalA = false;
  bool sawFinalB = false;
  const auto deadline = std::chrono::steady_clock::now() + DEADLINE;
  while (std::chrono::steady_clock::now() < deadline &&
         (!sawFinalA || !sawFinalB)) {
    tt::ipc::SharedToken token{};
    if (!harness.waitPopFor(token)) {
      continue;
    }
    if (token.task_id == taskA) {
      tokensA.push_back(token);
      sawFinalA = token.isFinal();
    } else if (token.task_id == taskB) {
      tokensB.push_back(token);
      sawFinalB = token.isFinal();
    } else {
      ADD_FAILURE() << "Received token for unexpected task_id="
                    << token.task_id;
    }
  }
  harness.assertRunnerHealthy();

  ASSERT_TRUE(sawFinalA) << "Expected final token for taskA";
  ASSERT_TRUE(sawFinalB) << "Expected final token for taskB";
  ASSERT_FALSE(tokensA.empty()) << "Expected tokens for taskA";
  ASSERT_FALSE(tokensB.empty()) << "Expected tokens for taskB";
  EXPECT_TRUE(tokensA.back().isFinal());
  EXPECT_TRUE(tokensB.back().isFinal());

  for (const auto& token : tokensA) {
    EXPECT_EQ(token.task_id, taskA);
    EXPECT_EQ(token.token_id, MOCK_PIPELINE_TOKEN_ID);
  }
  for (const auto& token : tokensB) {
    EXPECT_EQ(token.task_id, taskB);
    EXPECT_EQ(token.token_id, MOCK_PIPELINE_TOKEN_ID);
  }
}

}  // namespace tt::runners::blaze
