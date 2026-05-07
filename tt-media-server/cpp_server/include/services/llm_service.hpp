// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_set>
#include <vector>

#include "domain/llm_request.hpp"
#include "domain/llm_response.hpp"
#include "domain/tool_calls/tool_choice.hpp"
#include "ipc/queue_manager.hpp"
#include "ipc/task_queue.hpp"
#include "services/base_service.hpp"
#include "services/reasoning_parser.hpp"
#include "services/streamable.hpp"
#include "services/tool_call_parser.hpp"
#include "utils/concurrent_map.hpp"
#include "utils/tokenizers/tokenizer.hpp"
#include "worker/worker_manager.hpp"

namespace tt::services {

class LLMService
    : public BaseService<domain::LLMRequest, domain::LLMResponse>,
      public Streamable<domain::LLMRequest, domain::LLMStreamChunk> {
 public:
  using StreamCallback =
      std::function<void(const domain::LLMStreamChunk&, bool)>;

  LLMService();

  LLMService(const tt::utils::tokenizers::Tokenizer* tokenizer,
             std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
             std::unique_ptr<tt::worker::WorkerManager> workerManager,
             std::unique_ptr<ReasoningParser> reasoningParser,
             std::unique_ptr<IToolCallParser> toolCallParser,
             std::unique_ptr<tt::ipc::QueueManager> queueManager = nullptr,
             size_t maxQueueSize = std::numeric_limits<size_t>::max());

  ~LLMService() override;

  LLMService(const LLMService&) = delete;
  LLMService& operator=(const LLMService&) = delete;

  void start() override;
  void stop() override;

  bool isModelReady() const override;

  void preProcess(domain::LLMRequest& request) const override;

  void postProcess(domain::LLMResponse& response) const override;

  void processStreamingRequest(
      domain::LLMRequest request,
      std::function<void(domain::LLMStreamChunk&, bool isFinal)> callback)
      override;

  void abortRequest(uint32_t taskId);

  tt::worker::WorkerManager* getWorkerManager() const {
    return workerManager.get();
  }

 protected:
  size_t currentQueueSize() const override;
  domain::LLMResponse processRequest(domain::LLMRequest request) override;

  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

  void streamingPostProcess(domain::LLMStreamChunk&) const override {}

 private:
  struct StreamCallbackEntry {
    std::function<void(domain::LLMStreamChunk&, bool)> callback;
    bool skip_special_tokens = true;
  };

  void startConsumers();
  void consumerLoopForWorker(size_t workerIdx);

  std::optional<StreamCallbackEntry> resolveCallback(uint32_t taskId,
                                                     bool isFinal);

  void init(const tt::utils::tokenizers::Tokenizer* tokenizer,
            std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
            std::unique_ptr<tt::worker::WorkerManager> workerManager,
            std::unique_ptr<ReasoningParser> reasoningParser,
            std::unique_ptr<IToolCallParser> toolCallParser,
            std::unique_ptr<tt::ipc::QueueManager> queueManager,
            size_t maxQueueSize);

  std::vector<std::thread> consumerThreads;

  utils::ConcurrentMap<uint32_t, StreamCallbackEntry> streamCallbacks;
  mutable utils::ConcurrentMap<uint32_t, tt::domain::tool_calls::ToolChoice>
      toolChoiceMap;
  utils::ConcurrentMap<uint32_t, bool> reasoningSuppressedMap;
  utils::ConcurrentMap<uint32_t, StructuredOutputParseState>
      structuredOutputStateMap;

  std::atomic<size_t> pendingTasks{0};
  std::atomic<bool> running{false};

  std::shared_ptr<tt::ipc::ITaskQueue> taskQueue;
  std::unique_ptr<tt::worker::WorkerManager> workerManager;
  std::unique_ptr<tt::ipc::QueueManager> queueManager;
  const tt::utils::tokenizers::Tokenizer* tokenizer;
  std::unordered_set<int64_t> stopTokenSet;
  std::unique_ptr<ReasoningParser> reasoningParser;
  std::unique_ptr<IToolCallParser> toolCallParser;
};

}  // namespace tt::services
