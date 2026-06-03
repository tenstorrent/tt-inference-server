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

#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "domain/tool_calls/tool_choice.hpp"
#include "ipc/interface/task_queue.hpp"
#include "ipc/queue_manager.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/reasoning_parser.hpp"
#include "services/request_pipeline.hpp"
#include "services/tool_call_parser.hpp"
#include "utils/concurrent_map.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::services {

using namespace tt::domain::llm;

class LLMService : public BaseStreamingService<LLMRequest, LLMStreamChunk> {
 public:
  using StreamCallback = std::function<void(const LLMStreamChunk&, bool)>;

  LLMService();

  LLMService(std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
             std::unique_ptr<tt::worker::WorkerManager> workerManager,
             std::unique_ptr<ReasoningParser> reasoningParser,
             std::unique_ptr<IToolCallParser> toolCallParser,
             std::unique_ptr<tt::ipc::QueueManager> queueManager,
             size_t maxQueueSize = std::numeric_limits<size_t>::max());

  ~LLMService() override;

  LLMService(const LLMService&) = delete;
  LLMService& operator=(const LLMService&) = delete;

  void start() override;
  void stop() override;

  bool isModelReady() const override;

  void preProcess(LLMRequest& request) const override;

  /** Cleanup after a non-streaming response is assembled (not part of the
   *  streaming base contract). */
  void postProcess(LLMResponse& response) const;

  void abortRequest(uint32_t taskId) override;

  ReasoningParser* getReasoningParser() const { return reasoningParser.get(); }

  tt::worker::WorkerManager* getWorkerManager() const {
    return workerManager.get();
  }

 protected:
  size_t currentQueueSize() const override;

  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

  void produceStream(
      LLMRequest request,
      std::function<void(LLMStreamChunk&, bool isFinal)> callback) override;

 private:
  struct StreamCallbackEntry {
    std::function<void(LLMStreamChunk&, bool)> callback;
    bool skip_special_tokens = true;
    // Mirror of LLMRequest::skip_text_decode; lets the consumer loop
    // skip decode / reasoning / tool-call parsing for token-id-only
    // transports.
    bool skip_text_decode = false;
  };

  void startConsumers();
  void consumerLoopForWorker(size_t workerIdx);

  std::optional<StreamCallbackEntry> resolveCallback(uint32_t taskId,
                                                     bool isFinal);

  void init(std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
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

  std::atomic<size_t> pendingTasks{0};
  std::atomic<bool> running{false};

  std::shared_ptr<tt::ipc::ITaskQueue> taskQueue;
  std::unique_ptr<tt::worker::WorkerManager> workerManager;
  std::unique_ptr<tt::ipc::QueueManager> queueManager;
  std::unordered_set<int64_t> stopTokenSet;
  std::unique_ptr<ReasoningParser> reasoningParser;
  std::unique_ptr<IToolCallParser> toolCallParser;
  std::unique_ptr<IToolCallParser> jsonToolCallParser;
};

}  // namespace tt::services
