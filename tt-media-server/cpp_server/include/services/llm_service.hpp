// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_set>
#include <vector>

#include "domain/llm_request.hpp"
#include "domain/llm_response.hpp"
#include "domain/tool_calls/tool_choice.hpp"
#include "ipc/queue_manager.hpp"
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
  ~LLMService() override;

  LLMService(const LLMService&) = delete;
  LLMService& operator=(const LLMService&) = delete;

  void start() override;
  void stop() override;

  bool isModelReady() const override;

  void preProcess(domain::LLMRequest& request) const override;

  /**
   * Run post-processing (reasoning strip, tool-call parsing) on a fully
   * accumulated response. Public wrapper around the protected postProcess
   * so non-streaming callers that bypass submitRequest (e.g. the controller
   * accumulating from streaming chunks) can still apply final processing.
   */
  void finalizeResponse(domain::LLMResponse& response) const {
    postProcess(response);
  }

  /**
   * Abort an in-flight request. Removes the streaming callback, decrements
   * pending_tasks_, invokes the callback with finish_reason="abort" to unblock
   * synchronous waiters, and broadcasts cancel to all worker queues.
   * Idempotent and thread-safe.
   */
  void abortRequest(uint32_t taskId);

  /** Borrowed pointer to the worker manager, used by main to wire the
   * worker metrics aggregator. Lifetime tied to this LLMService. */
  tt::worker::WorkerManager* getWorkerManager() const {
    return workerManager.get();
  }

 protected:
  void postProcess(domain::LLMResponse& response) const override;
  size_t currentQueueSize() const override;
  domain::LLMResponse processRequest(domain::LLMRequest request) override;

  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

  void streamingPostProcess(domain::LLMStreamChunk&) const override {}
  void processStreamingRequest(
      domain::LLMRequest request,
      std::function<void(domain::LLMStreamChunk&, bool isFinal)> callback)
      override;

 private:
  struct StreamCallbackEntry {
    std::function<void(domain::LLMStreamChunk&, bool)> callback;
    bool skip_special_tokens = true;
  };

  void startConsumers();
  void consumerLoopForWorker(size_t workerIdx);

  std::optional<StreamCallbackEntry> resolveCallback(uint32_t taskId,
                                                     bool isFinal);

  std::vector<std::thread> consumerThreads;

  utils::ConcurrentMap<uint32_t, StreamCallbackEntry> streamCallbacks;
  mutable utils::ConcurrentMap<uint32_t, tt::domain::tool_calls::ToolChoice>
      toolChoiceMap;

  std::atomic<size_t> pendingTasks{0};
  std::atomic<bool> running{false};

  std::unique_ptr<tt::ipc::QueueManager> queueManager;
  std::unique_ptr<tt::worker::WorkerManager> workerManager;
  const tt::utils::tokenizers::Tokenizer* tokenizer;
  std::unordered_set<int64_t> stopTokenSet;
  std::unique_ptr<ReasoningParser> reasoningParser;
  std::unique_ptr<IToolCallParser> toolCallParser;
};

}  // namespace tt::services
