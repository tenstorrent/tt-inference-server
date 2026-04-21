// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "domain/llm_request.hpp"
#include "domain/llm_response.hpp"
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
   * Abort an in-flight request. Removes the streaming callback, decrements
   * pending_tasks_, invokes the callback with finish_reason="abort" to unblock
   * synchronous waiters, and broadcasts cancel to all worker queues.
   * Idempotent and thread-safe.
   */
  void abortRequest(uint32_t taskId);

  /** Borrowed pointer to the worker manager, used by main to wire the
   * worker metrics aggregator. Lifetime tied to this LLMService. */
  tt::worker::WorkerManager* workerManager() const {
    return worker_manager_.get();
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

  std::vector<std::thread> consumer_threads_;

  utils::ConcurrentMap<uint32_t, StreamCallbackEntry> stream_callbacks_;

  std::atomic<size_t> pending_tasks_{0};
  std::atomic<bool> running_{false};

  std::unique_ptr<tt::ipc::QueueManager> queue_manager_;
  std::unique_ptr<tt::worker::WorkerManager> worker_manager_;
  const tt::utils::tokenizers::Tokenizer* tokenizer_;
  std::unordered_set<int64_t> stop_token_set_;
  std::unique_ptr<ReasoningParser> reasoning_parser_;
  std::unique_ptr<IToolCallParser> tool_call_parser_;
};

}  // namespace tt::services
