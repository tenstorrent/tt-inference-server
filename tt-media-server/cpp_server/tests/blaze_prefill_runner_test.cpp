// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_prefill_runner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "config/runner_config.hpp"
#include "config/settings.hpp"
#include "domain/llm/sampling_params.hpp"
#include "domain/llm/sequence.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/in_memory/in_memory_cancel_queue.hpp"
#include "ipc/in_memory/in_memory_memory_queue.hpp"
#include "ipc/in_memory/in_memory_result_queue.hpp"
#include "ipc/in_memory/in_memory_task_queue.hpp"
#include "services/memory_services/memory_manager.hpp"

namespace tt::runners::blaze {

namespace {

constexpr auto DEADLINE = std::chrono::seconds(10);
constexpr auto POLL_INTERVAL = std::chrono::milliseconds(50);
constexpr uint64_t MOCK_DECODE_TOKEN_ID = 12345u;
constexpr uint64_t EMPTY_TOKEN_ID = std::numeric_limits<uint32_t>::max();
constexpr int DEFAULT_BLOCK_SIZE = 1;

class BlazePrefillRunnerHarness {
 public:
  BlazePrefillRunnerHarness() {
    memoryRequestQueue =
        std::make_shared<tt::ipc::in_memory::MemoryRequestQueue>();
    memoryResultQueue =
        std::make_shared<tt::ipc::in_memory::MemoryResultQueue>();
    auto memoryManager = std::make_unique<tt::services::MemoryManager>(
        memoryRequestQueue, memoryResultQueue);
    config.runner_type = tt::config::ModelRunnerType::MOCK_PIPELINE;
    runner = std::make_unique<BlazePrefillRunner>(config, &resultQueue,
                                                  &taskQueue, &cancelQueue,
                                                  std::move(memoryManager));

    runnerThread = std::thread([this]() {
      try {
        runner->start();
      } catch (...) {
        runnerError = std::current_exception();
      }
    });
  }

  ~BlazePrefillRunnerHarness() { shutdown(); }

  tt::domain::ManageMemoryResult allocate(uint32_t taskId) {
    tt::domain::ManageMemoryTask request{};
    request.taskId = taskId;
    request.action = tt::domain::MemoryManagementAction::ALLOCATE;
    memoryRequestQueue->push(request);

    tt::domain::ManageMemoryResult response{};
    memoryResultQueue->waitPop(response);
    return response;
  }

  void submitSequence(uint32_t taskId, uint32_t slotId,
                      const std::vector<int64_t>& promptTokens,
                      const tt::domain::llm::SamplingParams& samplingParams) {
    tt::domain::llm::Sequence seq(taskId, DEFAULT_BLOCK_SIZE, promptTokens,
                                  samplingParams);
    seq.setPrefillKVCacheSlot(slotId);
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
  }

  std::shared_ptr<tt::ipc::in_memory::MemoryRequestQueue> memoryRequestQueue;
  std::shared_ptr<tt::ipc::in_memory::MemoryResultQueue> memoryResultQueue;
  tt::ipc::in_memory::ResultQueue resultQueue;
  tt::ipc::in_memory::TaskQueue taskQueue;
  tt::ipc::in_memory::CancelQueue cancelQueue;
  tt::config::LLMConfig config{};
  std::unique_ptr<BlazePrefillRunner> runner;
  std::thread runnerThread;
  std::exception_ptr runnerError;
  bool isShutdown = false;
};

void expectNoDecodeTokens(const std::vector<tt::ipc::SharedToken>& tokens) {
  for (const auto& token : tokens) {
    EXPECT_NE(token.token_id, MOCK_DECODE_TOKEN_ID)
        << "Prefill runner must not emit decode tokens";
  }
}

}  // namespace

TEST(BlazePrefillRunnerIntegrationTest,
     InMemoryQueuesRoundTripThroughSimulator) {
  BlazePrefillRunnerHarness harness;

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
      << "Expected BlazePrefillRunner to emit a prefill-complete signal";
  EXPECT_TRUE(producedTokens.back().isFinal())
      << "Expected BlazePrefillRunner to emit a final completion signal";
  EXPECT_EQ(producedTokens.size(), 1u)
      << "Prefill should emit exactly one completion signal, not decode tokens";

  for (const auto& token : producedTokens) {
    EXPECT_EQ(token.task_id, taskId);
    EXPECT_EQ(token.token_id, EMPTY_TOKEN_ID);
    EXPECT_FALSE(token.isAbort());
  }
  expectNoDecodeTokens(producedTokens);
}

TEST(BlazePrefillRunnerIntegrationTest, CancelFlowEmitsAbortToken) {
  BlazePrefillRunnerHarness harness;

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
    expectNoDecodeTokens({token});
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

TEST(BlazePrefillRunnerIntegrationTest, TwoConcurrentTasksStayIsolated) {
  BlazePrefillRunnerHarness harness;

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

  ASSERT_TRUE(sawFinalA) << "Expected final signal for taskA";
  ASSERT_TRUE(sawFinalB) << "Expected final signal for taskB";
  ASSERT_EQ(tokensA.size(), 1u) << "Expected one prefill completion for taskA";
  ASSERT_EQ(tokensB.size(), 1u) << "Expected one prefill completion for taskB";
  EXPECT_TRUE(tokensA.back().isFinal());
  EXPECT_TRUE(tokensB.back().isFinal());

  for (const auto& token : tokensA) {
    EXPECT_EQ(token.task_id, taskA);
    EXPECT_EQ(token.token_id, EMPTY_TOKEN_ID);
  }
  for (const auto& token : tokensB) {
    EXPECT_EQ(token.task_id, taskB);
    EXPECT_EQ(token.token_id, EMPTY_TOKEN_ID);
  }
  expectNoDecodeTokens(tokensA);
  expectNoDecodeTokens(tokensB);
}

TEST(BlazePrefillRunnerIntegrationTest,
     ManyConcurrentUsersCompletePrefillUnderBackpressure) {
  BlazePrefillRunnerHarness harness;

  constexpr uint32_t kFirstTaskId = 9000;
  constexpr size_t kRequestedUsers = 128;
  const size_t userCount = std::min(kRequestedUsers, tt::config::pmMaxUsers());
  ASSERT_GE(userCount, 2u)
      << "Need at least two scheduler users for backpressure coverage";

  std::vector<uint32_t> taskIds;
  std::vector<uint32_t> slotIds;
  taskIds.reserve(userCount);
  slotIds.reserve(userCount);

  for (size_t i = 0; i < userCount; ++i) {
    const uint32_t taskId = kFirstTaskId + static_cast<uint32_t>(i);
    const auto allocateResponse = harness.allocate(taskId);
    ASSERT_EQ(allocateResponse.taskId, taskId);
    ASSERT_EQ(allocateResponse.status, tt::domain::ManageMemoryStatus::SUCCESS);
    ASSERT_NE(allocateResponse.slotId, tt::domain::INVALID_SLOT_ID);
    taskIds.push_back(taskId);
    slotIds.push_back(allocateResponse.slotId);
  }

  for (const auto slotId : slotIds) {
    EXPECT_EQ(std::count(slotIds.begin(), slotIds.end(), slotId), 1)
        << "Expected each concurrent user to get a unique slot";
  }

  tt::domain::llm::SamplingParams samplingParams;
  samplingParams.max_tokens = 4;
  samplingParams.ignore_eos = false;

  for (size_t i = 0; i < userCount; ++i) {
    harness.submitSequence(
        taskIds[i], slotIds[i],
        {static_cast<int64_t>(100 + i), static_cast<int64_t>(200 + i)},
        samplingParams);
  }

  std::vector<size_t> tokenCounts(userCount, 0);
  std::vector<bool> sawFinal(userCount, false);
  size_t finalCount = 0;
  const auto deadline = std::chrono::steady_clock::now() + DEADLINE;
  while (std::chrono::steady_clock::now() < deadline &&
         finalCount < userCount) {
    tt::ipc::SharedToken token{};
    if (!harness.waitPopFor(token)) {
      continue;
    }

    if (token.task_id < kFirstTaskId ||
        token.task_id >= kFirstTaskId + userCount) {
      ADD_FAILURE() << "Received token for unexpected task_id="
                    << token.task_id;
      continue;
    }

    const size_t userIndex = token.task_id - kFirstTaskId;
    EXPECT_FALSE(sawFinal[userIndex])
        << "Received token after final for task_id=" << token.task_id;
    tokenCounts[userIndex]++;
    EXPECT_EQ(token.token_id, EMPTY_TOKEN_ID);
    EXPECT_NE(token.token_id, MOCK_DECODE_TOKEN_ID);
    EXPECT_FALSE(token.isAbort());

    if (token.isFinal()) {
      sawFinal[userIndex] = true;
      finalCount++;
    }
  }
  harness.assertRunnerHealthy();

  ASSERT_EQ(finalCount, userCount)
      << "Expected every concurrent user to reach a final prefill signal";
  for (size_t i = 0; i < userCount; ++i) {
    EXPECT_TRUE(sawFinal[i])
        << "Missing final signal for task_id=" << taskIds[i];
    EXPECT_EQ(tokenCounts[i], 1u)
        << "Unexpected completion count for task_id=" << taskIds[i];
  }
}

}  // namespace tt::runners::blaze
