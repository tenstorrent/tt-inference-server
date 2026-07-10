// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/llm_service.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <vector>

#include "config/settings.hpp"
#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "domain/llm/sequence.hpp"
#include "ipc/in_memory/in_memory_task_queue.hpp"
#include "ipc/queue_manager.hpp"
#include "runtime/worker/worker_manager.hpp"

namespace {

void configureEnvForTest() {
  setenv("LLM_DEVICE_BACKEND", "pipeline_manager", 1);
  setenv("MODEL", "moonshotai/Kimi-K2.6", 1);
}

std::shared_ptr<tt::services::LLMService> makeService(
    std::shared_ptr<tt::ipc::ITaskQueue> taskQueue) {
  return std::make_shared<tt::services::LLMService>(
      std::move(taskQueue), std::make_unique<tt::worker::WorkerManager>(1),
      std::make_unique<tt::ipc::QueueManager>(1));
}

}  // namespace

TEST(LLMServiceProcessStreamingRequest, PushesSequenceToInjectedTaskQueue) {
  auto taskQueue = std::make_shared<tt::ipc::in_memory::TaskQueue>();

  auto llmService = makeService(taskQueue);

  tt::domain::llm::LLMRequest request{/*taskId=*/7};
  request.prompt = std::vector<uint32_t>{10, 20, 30};
  request.skip_special_tokens = true;

  ASSERT_NO_THROW(llmService->submitStreamingRequest(
      request, [](const tt::domain::llm::LLMStreamChunk&, bool) {},
      /*skipPreProcess=*/true));

  ASSERT_FALSE(taskQueue->empty());
  auto pushed = taskQueue->tryPop();
  ASSERT_NE(pushed, nullptr);
  EXPECT_EQ(pushed->taskId, 7u);
  EXPECT_EQ(pushed->getNumPromptTokens(), 3u);
  EXPECT_TRUE(taskQueue->empty());
}

TEST(LLMServiceProcessStreamingRequest,
     PushesSequenceToInjectedTaskQueueWithoutThinking) {
  auto taskQueue = std::make_shared<tt::ipc::in_memory::TaskQueue>();
  auto llmService = makeService(taskQueue);
  tt::domain::llm::LLMRequest request{/*taskId=*/7};
  request.prompt = std::vector<uint32_t>{10, 20, 30};
  request.skip_special_tokens = true;

  llmService->submitStreamingRequest(
      request, [](const tt::domain::llm::LLMStreamChunk&, bool) {},
      /*skipPreProcess=*/true);

  ASSERT_FALSE(taskQueue->empty());
  auto pushed = taskQueue->tryPop();
  ASSERT_NE(pushed, nullptr);
  EXPECT_EQ(pushed->taskId, 7u);
  EXPECT_EQ(pushed->getNumPromptTokens(), 3u);
  EXPECT_FALSE(pushed->getStartsInThinking());
}

TEST(LLMServiceProcessStreamingRequest,
     PushesSequenceToInjectedTaskQueueWithThinking) {
  auto taskQueue = std::make_shared<tt::ipc::in_memory::TaskQueue>();
  auto llmService = makeService(taskQueue);
  tt::domain::llm::LLMRequest request{/*taskId=*/7};
  uint32_t thinkStartKimi26 = 163606;
  request.prompt = std::vector<uint32_t>{10, 20, 30, thinkStartKimi26};
  request.skip_special_tokens = true;

  llmService->submitStreamingRequest(
      request, [](const tt::domain::llm::LLMStreamChunk&, bool) {},
      /*skipPreProcess=*/true);

  ASSERT_FALSE(taskQueue->empty());
  auto pushed = taskQueue->tryPop();
  ASSERT_NE(pushed, nullptr);
  EXPECT_EQ(pushed->taskId, 7u);
  EXPECT_EQ(pushed->getNumPromptTokens(), 4u);
  EXPECT_TRUE(pushed->getStartsInThinking());
}

TEST(LLMServiceProcessStreamingRequest,
     PushesSequenceToInjectedTaskQueueWithDisabledThinking) {
  auto taskQueue = std::make_shared<tt::ipc::in_memory::TaskQueue>();
  auto llmService = makeService(taskQueue);
  tt::domain::llm::LLMRequest request{/*taskId=*/7};
  uint32_t thinkStartKimi26 = 163606;
  uint32_t thinkEndKimi26 = 163607;
  request.prompt =
      std::vector<uint32_t>{10, 20, 30, thinkStartKimi26, thinkEndKimi26};
  request.skip_special_tokens = true;

  llmService->submitStreamingRequest(
      request, [](const tt::domain::llm::LLMStreamChunk&, bool) {},
      /*skipPreProcess=*/true);

  ASSERT_FALSE(taskQueue->empty());
  auto pushed = taskQueue->tryPop();
  ASSERT_NE(pushed, nullptr);
  EXPECT_EQ(pushed->taskId, 7u);
  EXPECT_EQ(pushed->getNumPromptTokens(), 5u);
  EXPECT_FALSE(pushed->getStartsInThinking());
}

TEST(LLMServiceProcessStreamingRequest, PropagatesMigrationStartPosition) {
  auto taskQueue = std::make_shared<tt::ipc::in_memory::TaskQueue>();
  auto llmService = makeService(taskQueue);
  tt::domain::llm::LLMRequest request{/*taskId=*/7};
  request.prompt = std::vector<uint32_t>{10, 20, 30};
  request.skip_special_tokens = true;
  request.migrationStartPosition = 64;

  llmService->submitStreamingRequest(
      request, [](const tt::domain::llm::LLMStreamChunk&, bool) {},
      /*skipPreProcess=*/true);

  ASSERT_FALSE(taskQueue->empty());
  auto pushed = taskQueue->tryPop();
  ASSERT_NE(pushed, nullptr);
  ASSERT_TRUE(pushed->getMigrationStartPosition().has_value());
  EXPECT_EQ(*pushed->getMigrationStartPosition(), 64u);
}

int main(int argc, char** argv) {
  configureEnvForTest();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
