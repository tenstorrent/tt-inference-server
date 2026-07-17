// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Shared helpers for integration tests.

#pragma once

#include <gtest/gtest.h>
#include <trantor/net/EventLoop.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <future>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "config/runner_config.hpp"
#include "config/settings.hpp"
#include "domain/llm/sampling_params.hpp"
#include "domain/llm/sequence.hpp"
#include "domain/manage_memory.hpp"
#include "domain/session.hpp"
#include "ipc/in_memory/in_memory_cancel_queue.hpp"
#include "ipc/in_memory/in_memory_memory_queue.hpp"
#include "ipc/in_memory/in_memory_result_queue.hpp"
#include "ipc/in_memory/in_memory_task_queue.hpp"
#include "ipc/interface/result_queue.hpp"
#ifdef ENABLE_BLAZE
#include "runtime/runners/blaze_runner/blaze_decode_runner.hpp"
#include "runtime/runners/blaze_runner/blaze_prefill_runner.hpp"
#include "runtime/runners/blaze_runner/blaze_scheduler_factory.hpp"
#endif
#include "../support/session_manager_helpers.hpp"
#include "services/memory_services/memory_manager.hpp"
#include "services/session_manager.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/id_generator.hpp"

namespace tt::test {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr auto kTestDeadline = std::chrono::seconds(10);
constexpr auto kPollInterval = std::chrono::milliseconds(50);

// ---------------------------------------------------------------------------
// Environment configuration
// ---------------------------------------------------------------------------

inline void configureTestEnv(const char* mode, const char* backend = "mock") {
  setenv("LLM_DEVICE_BACKEND", backend, 1);
  setenv("LLM_MODE", mode, 1);
  setenv("DEVICE_IDS", "(0)", 1);
  setenv("MAX_NUM_SESSIONS", "4", 1);
  setenv("KV_CACHE_FIRST_BLOCK_SIZE", "32", 1);
  setenv("KV_CACHE_BLOCK_SIZE", "32", 1);
  setenv("PREFIX_CACHE_HIT_THRESHOLD", "0", 1);
}

inline void configureRegularEnv() { configureTestEnv("regular"); }

inline void configurePrefillEnv() { configureTestEnv("prefill"); }

inline void configureDecodeEnv() { configureTestEnv("decode"); }

// ---------------------------------------------------------------------------
// Queue and config factories
// ---------------------------------------------------------------------------

inline std::shared_ptr<ipc::ITaskQueue> makeInMemoryTaskQueue() {
  return std::make_shared<ipc::in_memory::TaskQueue>();
}

inline config::BlazeConfig makeBlazeConfig(
    config::ModelRunnerType runnerType =
        config::ModelRunnerType::MOCK_PIPELINE) {
  // Start from the env-backed builder so all scheduler/pipeline knobs reflect
  // the current process env (set by configureProcess() before this call), then
  // override the runner type per-test. `blazeConfig()` reads the same
  // static-cached accessors the runners previously read directly, so this
  // preserves the existing per-process caching semantics.
  auto cfg = config::blazeConfig();
  cfg.runner_type = runnerType;
  return cfg;
}

// ---------------------------------------------------------------------------
// ID and prompt generation
// ---------------------------------------------------------------------------

inline uint32_t generateTaskId() { return utils::TaskIDGenerator::generate(); }

inline std::vector<uint32_t> makeSequentialPrompt(size_t length,
                                                  uint32_t start = 0) {
  std::vector<uint32_t> prompt(length);
  std::iota(prompt.begin(), prompt.end(), start);
  return prompt;
}

// ---------------------------------------------------------------------------
// Token collection helpers
// ---------------------------------------------------------------------------

inline std::vector<ipc::SharedToken> collectTokensUntilFinal(
    ipc::in_memory::ResultQueue& queue, uint32_t taskId,
    std::chrono::seconds deadline = kTestDeadline) {
  std::vector<ipc::SharedToken> tokens;
  const auto end = std::chrono::steady_clock::now() + deadline;
  while (std::chrono::steady_clock::now() < end) {
    ipc::SharedToken token{};
    if (!queue.waitPopFor(token, kPollInterval) || token.task_id != taskId) {
      continue;
    }
    tokens.push_back(token);
    if (token.isFinal()) {
      break;
    }
  }
  return tokens;
}

// ---------------------------------------------------------------------------
// Session manager helpers
// ---------------------------------------------------------------------------

inline std::string createTestSession(services::SessionManager& manager,
                                     trantor::EventLoop* loop,
                                     uint32_t slotId) {
  std::promise<std::string> promise;
  auto future = promise.get_future();

  manager.createSession(
      [&promise](const domain::Session& s) {
        promise.set_value(s.getSessionId());
      },
      [&promise](std::string_view err) {
        promise.set_exception(
            std::make_exception_ptr(std::runtime_error(std::string(err))));
      },
      loop, {}, slotId);

  return future.get();
}

inline std::string createTestSession(
    services::SessionManager& manager, trantor::EventLoop* loop,
    uint32_t slotId, const std::vector<utils::BlockHashInfo>& blockInfos) {
  std::promise<std::string> promise;
  auto future = promise.get_future();

  manager.createSession(
      [&promise](const domain::Session& s) {
        promise.set_value(s.getSessionId());
      },
      [&promise](std::string_view err) {
        promise.set_exception(
            std::make_exception_ptr(std::runtime_error(std::string(err))));
      },
      loop, blockInfos, slotId);

  return future.get();
}

inline uint32_t acquireInFlight(services::SessionManager& manager,
                                const std::string& sessionId) {
  return manager.acquireInFlight(sessionId, nullptr);
}

inline void releaseSlot(services::SessionManager& manager,
                        const std::string& sessionId) {
  if (auto session = manager.getSession(sessionId)) {
    session->release();
  }
}

// Simulates a completed turn-1 session registered under responseId.
inline std::string bootstrapSessionWithResponseId(
    services::SessionManager& manager, trantor::EventLoop* loop,
    uint32_t slotId, const std::string& responseId,
    const std::vector<utils::BlockHashInfo>& blockInfos = {}) {
  auto sessionId = blockInfos.empty()
                       ? createTestSession(manager, loop, slotId)
                       : createTestSession(manager, loop, slotId, blockInfos);
  manager.registerResponseId(sessionId, responseId);
  return sessionId;
}

// ---------------------------------------------------------------------------
// Concurrency helpers
// ---------------------------------------------------------------------------

// Runs a function from two threads simultaneously using a shared latch.
template <typename F>
void runConcurrently(F&& f, int numThreads = 2) {
  std::atomic<bool> ready{false};
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(numThreads));

  for (int i = 0; i < numThreads; ++i) {
    threads.emplace_back([&] {
      while (!ready.load(std::memory_order_acquire)) {
      }
      f();
    });
  }

  ready.store(true, std::memory_order_release);

  for (auto& t : threads) {
    t.join();
  }
}

// ---------------------------------------------------------------------------
// Base runner test harness
// ---------------------------------------------------------------------------

#ifdef ENABLE_BLAZE
// Base harness for testing Blaze runners (prefill and decode).
// Manages IPC queues, memory manager, and runner lifecycle.
template <typename RunnerType>
class RunnerTestHarness {
 public:
  explicit RunnerTestHarness(config::BlazeConfig config = {})
      : config_(config) {
    if (config_.runner_type == config::ModelRunnerType::MOCK) {
      config_.runner_type = config::ModelRunnerType::MOCK_PIPELINE;
    }
    init();
  }

  ~RunnerTestHarness() { shutdown(); }

  // Non-copyable, non-movable (owns threads and resources).
  RunnerTestHarness(const RunnerTestHarness&) = delete;
  RunnerTestHarness& operator=(const RunnerTestHarness&) = delete;
  RunnerTestHarness(RunnerTestHarness&&) = delete;
  RunnerTestHarness& operator=(RunnerTestHarness&&) = delete;

  // Allocate memory for a task via the memory manager.
  domain::ManageMemoryResult allocate(uint32_t taskId) {
    domain::ManageMemoryTask request{};
    request.taskId = taskId;
    request.action = domain::MemoryManagementAction::ALLOCATE;
    memoryRequestQueue_->push(request);

    domain::ManageMemoryResult response{};
    memoryResultQueue_->waitPop(response);
    return response;
  }

  // Submit a sequence to the runner's task queue.
  // Derived classes may override to set the correct KV cache slot method.
  void submitSequence(uint32_t taskId, uint32_t slotId,
                      const std::vector<uint32_t>& promptTokens,
                      const domain::llm::SamplingParams& samplingParams) {
    domain::llm::Sequence seq(taskId, promptTokens, samplingParams);
    setKVCacheSlot(seq, slotId);
    taskQueue_.push(seq);
  }

  bool waitPopFor(ipc::SharedToken& token) {
    return resultQueue_.waitPopFor(token, kPollInterval);
  }

  bool tryPopResult(ipc::SharedToken& token) {
    return resultQueue_.tryPop(token);
  }

  std::vector<ipc::SharedToken> collectTaskTokensUntilFinal(uint32_t taskId) {
    return collectTokensUntilFinal(resultQueue_, taskId, kTestDeadline);
  }

  void requestCancel(uint32_t taskId) { cancelQueue_.push(taskId); }

  void assertRunnerHealthy() const {
    if (runnerError_) {
      std::rethrow_exception(runnerError_);
    }
  }

  // Access to queues for advanced test scenarios.
  ipc::in_memory::ResultQueue& resultQueue() { return resultQueue_; }
  ipc::in_memory::TaskQueue& taskQueue() { return taskQueue_; }
  ipc::in_memory::CancelQueue& cancelQueue() { return cancelQueue_; }

 protected:
  // Override to use setPrefillKVCacheSlot vs setKVCacheSlot.
  virtual void setKVCacheSlot(domain::llm::Sequence& seq, uint32_t slotId) {
    seq.setKVCacheSlot(slotId);
  }

  config::BlazeConfig config_;

 private:
  void init() {
    memoryRequestQueue_ =
        std::make_shared<ipc::in_memory::MemoryRequestQueue>();
    memoryResultQueue_ = std::make_shared<ipc::in_memory::MemoryResultQueue>();
    auto memoryManager = std::make_unique<services::MemoryManager>(
        memoryRequestQueue_, memoryResultQueue_);

    if constexpr (std::is_same_v<RunnerType,
                                 runners::blaze::BlazeDecodeRunner>) {
      runner_ = std::make_unique<RunnerType>(
          config_, runners::blaze::makeDecodeScheduler(config_), &resultQueue_,
          &taskQueue_, &cancelQueue_, std::move(memoryManager));
    } else if constexpr (std::is_same_v<RunnerType,
                                        runners::blaze::BlazePrefillRunner>) {
      runner_ = std::make_unique<RunnerType>(
          config_, runners::blaze::makePrefillScheduler(config_), &resultQueue_,
          &taskQueue_, &cancelQueue_, std::move(memoryManager));
    } else {
      static_assert(sizeof(RunnerType) == 0,
                    "RunnerTestHarness only supports Blaze decode/prefill "
                    "runners");
    }

    runnerThread_ = std::thread([this]() {
      try {
        runner_->start();
      } catch (...) {
        runnerError_ = std::current_exception();
      }
    });
  }

  void shutdown() {
    if (isShutdown_) {
      return;
    }
    isShutdown_ = true;
    if (runner_) {
      runner_->stop();
    }
    if (runnerThread_.joinable()) {
      runnerThread_.join();
    }
    resultQueue_.shutdown();
    memoryRequestQueue_.reset();
    memoryResultQueue_.reset();
  }

  std::shared_ptr<ipc::in_memory::MemoryRequestQueue> memoryRequestQueue_;
  std::shared_ptr<ipc::in_memory::MemoryResultQueue> memoryResultQueue_;
  ipc::in_memory::ResultQueue resultQueue_;
  ipc::in_memory::TaskQueue taskQueue_;
  ipc::in_memory::CancelQueue cancelQueue_;
  std::unique_ptr<RunnerType> runner_;
  std::thread runnerThread_;
  std::exception_ptr runnerError_;
  bool isShutdown_ = false;
};
#endif  // ENABLE_BLAZE

}  // namespace tt::test
