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
#include "runners/llm_runner/in_memory_task_queue.hpp"
#include "services/reasoning_parser.hpp"
#include "services/tool_call_parser.hpp"
#include "worker/worker_manager.hpp"

namespace {

void configureEnvForTest() {
  setenv("LLM_DEVICE_BACKEND", "pipeline_manager", 1);
}

std::shared_ptr<tt::services::LLMService> makeService(
    std::shared_ptr<tt::ipc::ITaskQueue> taskQueue) {
  return std::make_shared<tt::services::LLMService>(
      std::move(taskQueue), std::make_unique<tt::worker::WorkerManager>(1),
      std::make_unique<tt::services::ReasoningParser>(),
      tt::services::createToolCallParser(tt::config::modelType()));
}

}  // namespace

TEST(LLMServiceProcessStreamingRequest, PushesSequenceToInjectedTaskQueue) {
  auto taskQueue =
      std::make_shared<tt::runners::llm_engine::InMemoryTaskQueue>();

  auto llmService = makeService(taskQueue);

  tt::domain::llm::LLMRequest request{/*taskId=*/7};
  request.prompt = std::vector<int>{10, 20, 30};
  request.skip_special_tokens = true;
  request.enable_reasoning = true;

  ASSERT_NO_THROW(llmService->processStreamingRequest(
      std::move(request), [](tt::domain::llm::LLMStreamChunk&, bool) {}));

  ASSERT_FALSE(taskQueue->empty());
  auto pushed = taskQueue->tryPop();
  ASSERT_NE(pushed, nullptr);
  EXPECT_EQ(pushed->taskId, 7u);
  EXPECT_EQ(pushed->getNumPromptTokens(), 3u);
  EXPECT_TRUE(taskQueue->empty());
}

int main(int argc, char** argv) {
  configureEnvForTest();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
