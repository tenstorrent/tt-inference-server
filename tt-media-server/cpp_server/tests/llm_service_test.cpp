// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// First-step unit test for LLMService::processStreamingRequest.
//
// Goal of this test: bootstrap a real LLMService instance in-process and
// drive a single call to processStreamingRequest. We deliberately do NOT
// call LLMService::start(), so no worker subprocesses are forked and no
// consumer threads are spawned -- the request lands in the IPC task queue
// and the streaming callback is registered, but no tokens ever flow back.
//
// Subsequent steps will:
//   1) Refactor LLMService so its dependencies (tokenizer, queue manager,
//      worker manager, metrics, parsers) can be injected.
//   2) Replace the live globals here with test doubles so we can assert
//      on what processStreamingRequest actually pushes to the queue and
//      registers in its internal maps.

#include "services/llm_service.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>

#include "config/settings.hpp"
#include "domain/llm_request.hpp"
#include "domain/llm_response.hpp"
#include "ipc/queue_manager.hpp"
#include "services/reasoning_parser.hpp"
#include "services/tool_call_parser.hpp"
#include "utils/tokenizers/tokenizer.hpp"
#include "worker/worker_manager.hpp"

namespace {

// Subclass exposes the protected `processStreamingRequest` to the test.
// The Streamable<...> base declares it protected; bringing it into the
// public scope of the test-only subclass is the least-invasive way to
// invoke it directly without altering production access control.
class TestableLLMService : public tt::services::LLMService {
 public:
  using tt::services::LLMService::processStreamingRequest;
};

// Set the environment variables that LLMService's constructor depends on
// (DEVICE_IDS, LLM_DEVICE_BACKEND, IPC queue names) before the service is
// constructed. Many of those values are cached in function-local statics
// the first time they're read, so this happens once for the whole test
// binary in main().
void configureEnvForTest() {
  // Single worker — WorkerManager throws if numWorkers < 1.
  setenv("DEVICE_IDS", "(0)", 1);

  // mock_pipeline maps to ModelType::DEEPSEEK_R1_0528, which is the
  // tokenizer shipped under cpp_server/tokenizers/deepseek-ai/.
  setenv("LLM_DEVICE_BACKEND", "mock_pipeline", 1);

  // Use unique IPC queue names so we don't collide with a real server
  // running on the same host (Boost.Interprocess uses /dev/shm).
  setenv("TT_TASK_QUEUE", "tt_tasks_llm_service_test", 1);
  setenv("TT_RESULT_QUEUE", "tt_results_llm_service_test_", 1);
  setenv("TT_CANCEL_QUEUE", "tt_cancels_llm_service_test_", 1);
  setenv("TT_WARMUP_SIGNALS_QUEUE", "tt_warmup_llm_service_test", 1);
  setenv("TT_WORKER_METRICS_SHM", "/tt_worker_metrics_llm_service_test", 1);
}

}  // namespace

TEST(LLMServiceProcessStreamingRequest, AcceptsRequestWithoutStart) {
  TestableLLMService service;

  tt::domain::LLMRequest request{/*taskId=*/42};
  // processStreamingRequest expects an already-tokenized prompt
  // (std::vector<int>). The unrelated string→tokens path lives in
  // preProcess(), which we are deliberately bypassing in this first test.
  request.prompt = std::vector<int>{1, 2, 3, 4};
  request.skip_special_tokens = true;
  request.enable_reasoning = true;

  std::atomic<int> callbackInvocations{0};
  auto callback = [&](tt::domain::LLMStreamChunk& /*chunk*/, bool /*isFinal*/) {
    callbackInvocations.fetch_add(1);
  };

  ASSERT_NO_THROW(service.processStreamingRequest(std::move(request), callback))
      << "processStreamingRequest must accept a well-formed request even when "
         "the service has not been start()ed yet";

  // No workers and no consumer threads are running (start() was not called),
  // so the callback must not fire synchronously from processStreamingRequest.
  EXPECT_EQ(callbackInvocations.load(), 0);
}

// Same scenario as above, but built through the new injection constructor.
// This pins down the seam: every collaborator is created in the test, not
// inside LLMService. As we add fakes/spies, only the construction site
// changes -- the rest of the test stays the same.
TEST(LLMServiceProcessStreamingRequest, AcceptsRequestViaInjectedDeps) {
  // Use a dedicated set of IPC queue names so this case does not clash
  // with the default-constructed service in the previous test.
  setenv("TT_TASK_QUEUE", "tt_tasks_llm_service_test_inj", 1);
  setenv("TT_RESULT_QUEUE", "tt_results_llm_service_test_inj_", 1);
  setenv("TT_CANCEL_QUEUE", "tt_cancels_llm_service_test_inj_", 1);

  const auto numWorkers = tt::config::numWorkers();

  class TestableInjectedService : public tt::services::LLMService {
   public:
    using tt::services::LLMService::LLMService;
    using tt::services::LLMService::processStreamingRequest;
  };

  TestableInjectedService service{
      &tt::utils::tokenizers::activeTokenizer(),
      std::make_unique<tt::worker::WorkerManager>(numWorkers),
      std::make_unique<tt::ipc::QueueManager>(static_cast<int>(numWorkers)),
      std::make_unique<tt::services::ReasoningParser>(),
      tt::services::createToolCallParser(tt::config::modelType()),
      tt::config::maxQueueSize()};

  tt::domain::LLMRequest request{/*taskId=*/7};
  request.prompt = std::vector<int>{10, 20, 30};
  request.skip_special_tokens = true;
  request.enable_reasoning = true;

  std::atomic<int> callbackInvocations{0};
  auto callback = [&](tt::domain::LLMStreamChunk& /*chunk*/, bool /*isFinal*/) {
    callbackInvocations.fetch_add(1);
  };

  ASSERT_NO_THROW(
      service.processStreamingRequest(std::move(request), callback));
  EXPECT_EQ(callbackInvocations.load(), 0);
}

// Null-dep guards in the injection constructor: the production constructor
// can't reach these branches, but they protect future test fakes from
// accidentally passing nullptr for collaborators that processStreamingRequest
// dereferences unconditionally.
TEST(LLMServiceConstructor, RejectsNullTokenizer) {
  EXPECT_THROW(
      tt::services::LLMService(
          /*tokenizer=*/nullptr, std::make_unique<tt::worker::WorkerManager>(1),
          std::make_unique<tt::ipc::QueueManager>(1),
          std::make_unique<tt::services::ReasoningParser>(),
          tt::services::createToolCallParser(tt::config::modelType())),
      std::invalid_argument);
}

TEST(LLMServiceConstructor, RejectsNullWorkerManager) {
  EXPECT_THROW(
      tt::services::LLMService(
          &tt::utils::tokenizers::activeTokenizer(),
          /*workerManager=*/nullptr, std::make_unique<tt::ipc::QueueManager>(1),
          std::make_unique<tt::services::ReasoningParser>(),
          tt::services::createToolCallParser(tt::config::modelType())),
      std::invalid_argument);
}

TEST(LLMServiceConstructor, RejectsNullQueueManager) {
  EXPECT_THROW(tt::services::LLMService(
                   &tt::utils::tokenizers::activeTokenizer(),
                   std::make_unique<tt::worker::WorkerManager>(1),
                   /*queueManager=*/nullptr,
                   std::make_unique<tt::services::ReasoningParser>(),
                   tt::services::createToolCallParser(tt::config::modelType())),
               std::invalid_argument);
}

int main(int argc, char** argv) {
  configureEnvForTest();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
