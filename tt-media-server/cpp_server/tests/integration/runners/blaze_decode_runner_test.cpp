// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_decode_runner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

#include "../integration_test_helpers.hpp"
#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "tt_llm_engine/pipeline/pipeline_types.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::runners::blaze {
namespace {

class EnvSetter {
  public:
    EnvSetter(const char* key, const char* value) {
    key_ = key;
    setenv(key_, value, 1);
  }
  ~EnvSetter() {
      unsetenv(key_);
    }
  private:
  const char* key_;
};

constexpr uint64_t MOCK_PIPELINE_TOKEN_ID = 12345u;
const std::vector<int64_t> DEFAULT_STOP_TOKEN_IDS = {987654321};

// Length of the mock pipeline's reasoning preamble: think-open, three decode
// tokens, think-close. Mirrors PipelineSimulator::kThinkPreambleLen.
constexpr size_t THINK_PREAMBLE_LEN = 5;

// When the active model defines think tokens, the mock PipelineSimulator emits
// a reasoning preamble ahead of the fixed decode token. The runner casts the
// think ids to uint32_t, so a model without think tokens collapses to
// EMPTY_TOKEN and the preamble is disabled.
inline bool mockPreambleEnabled() {
  const auto think = tt::utils::tokenizers::thinkTokenIds();
  return static_cast<uint32_t>(think.first) !=
             tt_llm_engine::pipeline::EMPTY_TOKEN &&
         static_cast<uint32_t>(think.second) !=
             tt_llm_engine::pipeline::EMPTY_TOKEN;
}

// Token the mock pipeline publishes at position `index` of a fresh generation.
// With the preamble the stream is [think-open, decode, decode, decode,
// think-close, decode, ...]; otherwise it is a flat stream of the decode token.
inline uint64_t expectedMockToken(size_t index) {
  if (!mockPreambleEnabled()) {
    return MOCK_PIPELINE_TOKEN_ID;
  }
  const auto think = tt::utils::tokenizers::thinkTokenIds();
  const uint64_t thinkOpen = static_cast<uint32_t>(think.first);
  const uint64_t thinkClose = static_cast<uint32_t>(think.second);
  // The reasoning block the mock simulator emits, in order, before settling
  // into a flat stream of the decode token.
  const std::array<uint64_t, THINK_PREAMBLE_LEN> preamble = {
      thinkOpen, MOCK_PIPELINE_TOKEN_ID, MOCK_PIPELINE_TOKEN_ID,
      MOCK_PIPELINE_TOKEN_ID, thinkClose};
  return index < preamble.size() ? preamble[index] : MOCK_PIPELINE_TOKEN_ID;
}

// Specialized harness for decode runner that uses setKVCacheSlot (default).
class BlazeDecodeRunnerHarness
    : public test::RunnerTestHarness<BlazeDecodeRunner> {
 public:
  explicit BlazeDecodeRunnerHarness(
      std::vector<int64_t> stopTokenIds = DEFAULT_STOP_TOKEN_IDS)
      : test::RunnerTestHarness<BlazeDecodeRunner>(
            test::makeLLMConfig(128, 8, 0, std::move(stopTokenIds))) {}
};

}  // namespace

TEST(BlazeDecodeRunnerIntegrationTest,
     InMemoryQueuesRoundTripThroughSimulator) {
  BlazeDecodeRunnerHarness harness;

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
      << "Expected BlazeDecodeRunner to emit at least one token";
  EXPECT_TRUE(producedTokens.back().isFinal())
      << "Expected BlazeDecodeRunner to emit a final token";

  for (size_t i = 0; i < producedTokens.size(); ++i) {
    EXPECT_EQ(producedTokens[i].task_id, taskId);

    // When think tokens are configured, the mock simulator opens generation
    // with a reasoning block (think-open, decode tokens, think-close) before
    // settling into a flat stream of the decode token. expectedMockToken()
    // returns the token expected at each position of that stream.
    EXPECT_EQ(producedTokens[i].token_id, expectedMockToken(i))
        << "Unexpected token at index " << i;
  }
}

TEST(BlazeDecodeRunnerIntegrationTest, CancelFlowEmitsAbortToken) {
  BlazeDecodeRunnerHarness harness;

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

  bool sawInitialToken = false;
  const auto tokenDeadline =
      std::chrono::steady_clock::now() + test::kTestDeadline;
  while (std::chrono::steady_clock::now() < tokenDeadline) {
    ipc::SharedToken token{};
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
  ipc::SharedToken abortToken{};
  const auto abortDeadline =
      std::chrono::steady_clock::now() + test::kTestDeadline;
  while (std::chrono::steady_clock::now() < abortDeadline) {
    ipc::SharedToken token{};
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

TEST(BlazeDecodeRunnerIntegrationTest, StopsOnConfiguredStopToken) {
  // Mock simulator emits token_id=12345; this forces stop-token completion.
  BlazeDecodeRunnerHarness harness({MOCK_PIPELINE_TOKEN_ID});

  const uint32_t taskId = 6262;
  const auto allocateResponse = harness.allocate(taskId);
  ASSERT_EQ(allocateResponse.taskId, taskId);
  ASSERT_EQ(allocateResponse.status, domain::ManageMemoryStatus::SUCCESS);
  ASSERT_NE(allocateResponse.slotId, domain::INVALID_SLOT_ID);

  domain::llm::SamplingParams samplingParams;
  samplingParams.max_tokens = 1000;
  samplingParams.ignore_eos = false;
  samplingParams.stop_token_ids = {MOCK_PIPELINE_TOKEN_ID};

  harness.submitSequence(taskId, allocateResponse.slotId, {77}, samplingParams);
  const auto producedTokens = harness.collectTaskTokensUntilFinal(taskId);
  harness.assertRunnerHealthy();

  ASSERT_FALSE(producedTokens.empty())
      << "Expected at least one token for stop-token test";
  ASSERT_TRUE(producedTokens.back().isFinal())
      << "Expected stop token to end generation";
  // The configured stop token is the decode token, so generation halts the
  // first time the decode token is emitted. With the reasoning preamble that is
  // one step in (after the think-open token); otherwise it is the first token.
  const std::vector<uint64_t> expected =
      mockPreambleEnabled()
          ? std::vector<uint64_t>{expectedMockToken(0), MOCK_PIPELINE_TOKEN_ID}
          : std::vector<uint64_t>{MOCK_PIPELINE_TOKEN_ID};

  ASSERT_EQ(producedTokens.size(), expected.size());
  for (size_t i = 0; i < producedTokens.size(); ++i) {
    EXPECT_EQ(producedTokens[i].task_id, taskId);
    EXPECT_EQ(producedTokens[i].token_id, expected[i])
        << "Unexpected token at index " << i;
    EXPECT_FALSE(producedTokens[i].isAbort());
  }
}

TEST(BlazeDecodeRunnerIntegrationTest, TwoConcurrentTasksStayIsolated) {
  BlazeDecodeRunnerHarness harness;

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

  ASSERT_TRUE(sawFinalA) << "Expected final token for taskA";
  ASSERT_TRUE(sawFinalB) << "Expected final token for taskB";
  ASSERT_FALSE(tokensA.empty()) << "Expected tokens for taskA";
  ASSERT_FALSE(tokensB.empty()) << "Expected tokens for taskB";
  EXPECT_TRUE(tokensA.back().isFinal());
  EXPECT_TRUE(tokensB.back().isFinal());

  for (size_t i = 0; i < tokensA.size(); ++i) {
    EXPECT_EQ(tokensA[i].task_id, taskA);
    EXPECT_EQ(tokensA[i].token_id, expectedMockToken(i))
        << "Unexpected taskA token at index " << i;
  }
  for (size_t i = 0; i < tokensB.size(); ++i) {
    EXPECT_EQ(tokensB[i].task_id, taskB);
    EXPECT_EQ(tokensB[i].token_id, expectedMockToken(i))
        << "Unexpected taskB token at index " << i;
  }
}

TEST(BlazeDecodeRunnerIntegrationTest,
     ManyConcurrentUsersGenerateTokensUnderBackpressure) {
  BlazeDecodeRunnerHarness harness;

  constexpr uint32_t kFirstTaskId = 9000;
  constexpr size_t kRequestedUsers = 128;
  constexpr int kMaxTokensPerUser = 4;
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
  samplingParams.max_tokens = kMaxTokensPerUser;
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
    EXPECT_EQ(token.token_id, expectedMockToken(tokenCounts[userIndex]))
        << "Unexpected token for task_id=" << token.task_id << " at index "
        << tokenCounts[userIndex];
    tokenCounts[userIndex]++;
    EXPECT_FALSE(token.isAbort());

    if (token.isFinal()) {
      sawFinal[userIndex] = true;
      finalCount++;
    }
  }
  harness.assertRunnerHealthy();

  ASSERT_EQ(finalCount, userCount)
      << "Expected every concurrent user to reach a final token";
  for (size_t i = 0; i < userCount; ++i) {
    EXPECT_TRUE(sawFinal[i])
        << "Missing final token for task_id=" << taskIds[i];
    EXPECT_EQ(tokenCounts[i], static_cast<size_t>(kMaxTokensPerUser))
        << "Unexpected token count for task_id=" << taskIds[i];
  }
}

TEST(BlazeDecodeRunnerIntegrationTest, MockSchedulerFlatTokenStream) {
  EnvSetter mockUseScheduler("MOCK_USE_SCHEDULER", "1");
  EnvSetter mockPrefillLatencyMs("MOCK_PREFILL_LATENCY_MS", "0");
  EnvSetter mockDecodeTokenLatencyUs("MOCK_DECODE_TOKEN_LATENCY_US", "0");

  BlazeDecodeRunnerHarness harness;

  const uint32_t taskId = 5150;
  const auto allocateResponse = harness.allocate(taskId);
  ASSERT_EQ(allocateResponse.status, domain::ManageMemoryStatus::SUCCESS);

  domain::llm::SamplingParams samplingParams;
  samplingParams.max_tokens = 3;
  samplingParams.ignore_eos = true;

  harness.submitSequence(taskId, allocateResponse.slotId, {11, 22, 33},
                         samplingParams);
  const auto producedTokens = harness.collectTaskTokensUntilFinal(taskId);
  harness.assertRunnerHealthy();

  ASSERT_EQ(producedTokens.size(), 3u);
  for (const auto& token : producedTokens) {
    EXPECT_EQ(token.token_id, MOCK_PIPELINE_TOKEN_ID);
  }
  EXPECT_TRUE(producedTokens.back().isFinal());
}

}  // namespace tt::runners::blaze
