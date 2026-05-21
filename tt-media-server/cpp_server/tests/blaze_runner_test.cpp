// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>
#include <unistd.h>
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
#include "runtime/runners/blaze_runner/blaze_runner.hpp"

namespace tt::runners::blaze {

TEST(BlazeRunnerIntegrationTest, InMemoryQueuesRoundTripThroughSimulator) {
  const std::string suffix = std::to_string(::getpid()) + "_blaze_integration";
  const std::string memoryReqQueueName = "test_blaze_mem_req_" + suffix;
  const std::string memoryResQueueName = "test_blaze_mem_res_" + suffix;

  setenv("TT_MEMORY_REQUEST_QUEUE", memoryReqQueueName.c_str(), 1);
  setenv("TT_MEMORY_RESULT_QUEUE", memoryResQueueName.c_str(), 1);

  // Remove any existing queues from previous tests.
  // These could be left over from previous test runs and can survive if previous tests failed before cleanup.
  tt::ipc::boost::MemoryRequestQueue::remove(memoryReqQueueName);
  tt::ipc::boost::MemoryResultQueue::remove(memoryResQueueName);

  tt::ipc::boost::MemoryRequestQueue memoryRequestQueue(
      memoryReqQueueName, static_cast<int>(tt::config::memoryQueueCapacity())); // 128 is the default capacity
  tt::ipc::boost::MemoryResultQueue memoryResultQueue(
      memoryResQueueName, static_cast<int>(tt::config::memoryQueueCapacity())); // 128 is the default capacity

  tt::ipc::in_memory::ResultQueue resultQueue;
  tt::ipc::in_memory::TaskQueue taskQueue;
  tt::ipc::in_memory::CancelQueue cancelQueue;

  tt::config::LLMConfig config{};
  config.runner_type = tt::config::ModelRunnerType::MOCK_PIPELINE;
  // This is used as a fake stop token so it won’t accidentally stop on 12345.
  config.stop_token_ids = {987654321};

  BlazeRunner runner(config, &resultQueue, &taskQueue, &cancelQueue);

  // The way to keep the exception from the runner thread to the main thread.
  // Run the BlazeRunner in a separate thread and catch any exceptions.
  std::exception_ptr runnerError;
  std::thread runnerThread([&]() {
    try {
      runner.start();
    } catch (...) {
      runnerError = std::current_exception();
    }
  });

  const uint32_t taskId = 4242;
  tt::domain::ManageMemoryTask allocateRequest{};
  allocateRequest.taskId = taskId;
  allocateRequest.action = tt::domain::MemoryManagementAction::ALLOCATE;
  memoryRequestQueue.push(allocateRequest);

  tt::domain::ManageMemoryResult allocateResponse{};
  memoryResultQueue.receive(allocateResponse);
  ASSERT_EQ(allocateResponse.taskId, taskId);
  ASSERT_EQ(allocateResponse.status, tt::domain::ManageMemoryStatus::SUCCESS);
  ASSERT_NE(allocateResponse.slotId, tt::domain::INVALID_SLOT_ID);

  tt::domain::llm::SamplingParams samplingParams;
  samplingParams.max_tokens = 3;
  samplingParams.ignore_eos = false;

  tt::domain::llm::Sequence seq(taskId, 1, {11, 22, 33}, samplingParams);
  seq.setKVCacheSlot(allocateResponse.slotId);
  taskQueue.push(seq);

  std::vector<tt::ipc::SharedToken> producedTokens;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < deadline) {
    tt::ipc::SharedToken token{};
    if (!resultQueue.waitPopFor(token, std::chrono::milliseconds(50))) {
      continue;
    }
    producedTokens.push_back(token);
    if (token.isFinal()) {
      break;
    }
  }

  runner.stop();
  if (runnerThread.joinable()) {
    runnerThread.join();
  }
  resultQueue.shutdown();
  tt::ipc::boost::MemoryRequestQueue::remove(memoryReqQueueName);
  tt::ipc::boost::MemoryResultQueue::remove(memoryResQueueName);

  if (runnerError) {
    std::rethrow_exception(runnerError);
  }

  ASSERT_FALSE(producedTokens.empty())
      << "Expected BlazeRunner to emit at least one token";
  EXPECT_TRUE(producedTokens.back().isFinal())
      << "Expected BlazeRunner to emit a final token";

  for (const auto& token : producedTokens) {
    EXPECT_EQ(token.task_id, taskId);

    // Mock simulator is configured to emit token 12345 for all tasks.
    EXPECT_EQ(token.token_id, 12345u);
  }
}

TEST(BlazeRunnerIntegrationTest, CancelFlowEmitsAbortToken) {
  const std::string suffix = std::to_string(::getpid()) + "_blaze_cancel";
  const std::string memoryReqQueueName = "test_blaze_mem_req_" + suffix;
  const std::string memoryResQueueName = "test_blaze_mem_res_" + suffix;

  setenv("TT_MEMORY_REQUEST_QUEUE", memoryReqQueueName.c_str(), 1);
  setenv("TT_MEMORY_RESULT_QUEUE", memoryResQueueName.c_str(), 1);

  tt::ipc::boost::MemoryRequestQueue::remove(memoryReqQueueName);
  tt::ipc::boost::MemoryResultQueue::remove(memoryResQueueName);

  tt::ipc::boost::MemoryRequestQueue memoryRequestQueue(
      memoryReqQueueName, static_cast<int>(tt::config::memoryQueueCapacity()));
  tt::ipc::boost::MemoryResultQueue memoryResultQueue(
      memoryResQueueName, static_cast<int>(tt::config::memoryQueueCapacity()));

  tt::ipc::in_memory::ResultQueue resultQueue;
  tt::ipc::in_memory::TaskQueue taskQueue;
  tt::ipc::in_memory::CancelQueue cancelQueue;

  tt::config::LLMConfig config{};
  config.runner_type = tt::config::ModelRunnerType::MOCK_PIPELINE;
  config.stop_token_ids = {987654321};

  BlazeRunner runner(config, &resultQueue, &taskQueue, &cancelQueue);

  std::exception_ptr runnerError;
  std::thread runnerThread([&]() {
    try {
      runner.start();
    } catch (...) {
      runnerError = std::current_exception();
    }
  });

  const uint32_t taskId = 5252;
  tt::domain::ManageMemoryTask allocateRequest{};
  allocateRequest.taskId = taskId;
  allocateRequest.action = tt::domain::MemoryManagementAction::ALLOCATE;
  memoryRequestQueue.push(allocateRequest);

  tt::domain::ManageMemoryResult allocateResponse{};
  memoryResultQueue.receive(allocateResponse);
  ASSERT_EQ(allocateResponse.taskId, taskId);
  ASSERT_EQ(allocateResponse.status, tt::domain::ManageMemoryStatus::SUCCESS);
  ASSERT_NE(allocateResponse.slotId, tt::domain::INVALID_SLOT_ID);

  tt::domain::llm::SamplingParams samplingParams;
  samplingParams.max_tokens = 1000;
  samplingParams.ignore_eos = true;

  tt::domain::llm::Sequence seq(taskId, 1, {44, 55, 66}, samplingParams);
  seq.setKVCacheSlot(allocateResponse.slotId);
  taskQueue.push(seq);

  bool sawInitialToken = false;
  const auto tokenDeadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < tokenDeadline) {
    tt::ipc::SharedToken token{};
    if (!resultQueue.waitPopFor(token, std::chrono::milliseconds(50))) {
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

  cancelQueue.push(taskId);

  bool sawAbort = false;
  tt::ipc::SharedToken abortToken{};
  const auto abortDeadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < abortDeadline) {
    tt::ipc::SharedToken token{};
    if (!resultQueue.waitPopFor(token, std::chrono::milliseconds(50))) {
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

  runner.stop();
  if (runnerThread.joinable()) {
    runnerThread.join();
  }
  resultQueue.shutdown();
  tt::ipc::boost::MemoryRequestQueue::remove(memoryReqQueueName);
  tt::ipc::boost::MemoryResultQueue::remove(memoryResQueueName);

  if (runnerError) {
    std::rethrow_exception(runnerError);
  }

  ASSERT_TRUE(sawAbort) << "Expected abort token after cancel request";
  EXPECT_EQ(abortToken.task_id, taskId);
  EXPECT_TRUE(abortToken.isAbort());
  EXPECT_EQ(abortToken.token_id, 0u);
}

}  // namespace tt::runners::blaze
