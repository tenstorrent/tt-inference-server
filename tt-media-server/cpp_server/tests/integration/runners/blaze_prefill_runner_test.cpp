// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_prefill_runner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#include "../integration_test_helpers.hpp"
#include "config/settings.hpp"
#include "domain/manage_memory.hpp"

namespace tt::runners::blaze {

namespace {

constexpr uint32_t MOCK_DECODE_TOKEN_ID = 12345u;
constexpr uint32_t EMPTY_TOKEN_ID = std::numeric_limits<uint32_t>::max();

// Specialized harness for prefill runner that uses setPrefillKVCacheSlot.
class BlazePrefillRunnerHarness
    : public test::RunnerTestHarness<BlazePrefillRunner> {
 public:
  BlazePrefillRunnerHarness()
      : test::RunnerTestHarness<BlazePrefillRunner>(config::BlazeConfig{}) {}

 protected:
  void setKVCacheSlot(domain::llm::Sequence& seq, uint32_t slotId) override {
    seq.setPrefillKVCacheSlot(slotId);
  }
};

void expectNoDecodeTokens(const std::vector<ipc::SharedToken>& tokens) {
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
  ASSERT_EQ(allocateResponse.status, domain::ManageMemoryStatus::SUCCESS);
  ASSERT_NE(allocateResponse.slotId, domain::INVALID_SLOT_ID);

  domain::llm::SamplingParams samplingParams;
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
  // Timing-sensitive: cancel must land while the slot is RUNNING.
  GTEST_SKIP() << "Flaky with mock prefill; decode cancel test still covers "
                  "the abort-token contract";

  BlazePrefillRunnerHarness harness;

  const uint32_t taskId = 5252;
  const auto allocateResponse = harness.allocate(taskId);
  ASSERT_EQ(allocateResponse.taskId, taskId);
  ASSERT_EQ(allocateResponse.status, domain::ManageMemoryStatus::SUCCESS);
  ASSERT_NE(allocateResponse.slotId, domain::INVALID_SLOT_ID);

  domain::llm::SamplingParams samplingParams;
  samplingParams.max_tokens = 1000;
  samplingParams.ignore_eos = true;

  harness.submitSequence(taskId, allocateResponse.slotId, {44, 55, 66},
                         samplingParams);

  // Prefill has no streaming tokens to sync on (unlike decode). The runner
  // also drains cancels before new tasks each step(), so a single early cancel
  // can be ignored. Retry cancel in a tight loop until we see ABORT or prefill
  // completes (FINAL without ABORT).
  bool sawAbort = false;
  ipc::SharedToken abortToken{};
  const auto abortDeadline =
      std::chrono::steady_clock::now() + test::kTestDeadline;
  while (std::chrono::steady_clock::now() < abortDeadline && !sawAbort) {
    harness.requestCancel(taskId);
    ipc::SharedToken token{};
    if (!harness.tryPopResult(token)) {
      std::this_thread::yield();
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
    if (token.isFinal()) {
      FAIL() << "Prefill completed before cancel could emit an abort token";
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

  uint32_t slotA = domain::INVALID_SLOT_ID;
  uint32_t slotB = domain::INVALID_SLOT_ID;

  const auto responseA = harness.allocate(taskA);
  const auto responseB = harness.allocate(taskB);
  for (const auto& response : {responseA, responseB}) {
    ASSERT_EQ(response.status, domain::ManageMemoryStatus::SUCCESS);
    ASSERT_NE(response.slotId, domain::INVALID_SLOT_ID);
    if (response.taskId == taskA) {
      slotA = response.slotId;
    } else if (response.taskId == taskB) {
      slotB = response.slotId;
    } else {
      FAIL() << "Unexpected taskId in allocate response: " << response.taskId;
    }
  }

  ASSERT_NE(slotA, domain::INVALID_SLOT_ID);
  ASSERT_NE(slotB, domain::INVALID_SLOT_ID);

  domain::llm::SamplingParams samplingParams;
  samplingParams.max_tokens = 3;
  samplingParams.ignore_eos = false;

  harness.submitSequence(taskA, slotA, {11, 12}, samplingParams);
  harness.submitSequence(taskB, slotB, {21, 22}, samplingParams);

  std::vector<ipc::SharedToken> tokensA;
  std::vector<ipc::SharedToken> tokensB;
  bool sawFinalA = false;
  bool sawFinalB = false;
  const auto deadline = std::chrono::steady_clock::now() + test::kTestDeadline;
  while (std::chrono::steady_clock::now() < deadline &&
         (!sawFinalA || !sawFinalB)) {
    ipc::SharedToken token{};
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
  const size_t userCount = std::min(kRequestedUsers, config::pmMaxUsers());
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
    ASSERT_EQ(allocateResponse.status, domain::ManageMemoryStatus::SUCCESS);
    ASSERT_NE(allocateResponse.slotId, domain::INVALID_SLOT_ID);
    taskIds.push_back(taskId);
    slotIds.push_back(allocateResponse.slotId);
  }

  for (const auto slotId : slotIds) {
    EXPECT_EQ(std::count(slotIds.begin(), slotIds.end(), slotId), 1)
        << "Expected each concurrent user to get a unique slot";
  }

  domain::llm::SamplingParams samplingParams;
  samplingParams.max_tokens = 4;
  samplingParams.ignore_eos = false;

  for (size_t i = 0; i < userCount; ++i) {
    harness.submitSequence(
        taskIds[i], slotIds[i],
        {static_cast<uint32_t>(100 + i), static_cast<uint32_t>(200 + i)},
        samplingParams);
  }

  std::vector<size_t> tokenCounts(userCount, 0);
  std::vector<bool> sawFinal(userCount, false);
  size_t finalCount = 0;
  const auto deadline = std::chrono::steady_clock::now() + test::kTestDeadline;
  while (std::chrono::steady_clock::now() < deadline &&
         finalCount < userCount) {
    ipc::SharedToken token{};
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
