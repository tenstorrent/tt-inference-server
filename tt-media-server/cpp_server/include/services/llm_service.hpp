// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"
#include "ipc/queue_manager.hpp"
#include "services/base_service.hpp"
#include "services/reasoning_parser.hpp"
#include "services/streamable.hpp"
#include "utils/concurrent_map.hpp"
#include "utils/tokenizer.hpp"
#include "worker/worker_manager.hpp"

namespace tt::services {

class LLMService
    : public BaseService<domain::CompletionRequest, domain::CompletionResponse>,
      public Streamable<domain::CompletionRequest,
                        domain::StreamingChunkResponse> {
 public:
  using StreamCallback =
      std::function<void(const domain::StreamingChunkResponse&, bool)>;

  LLMService();
  ~LLMService() override;

  LLMService(const LLMService&) = delete;
  LLMService& operator=(const LLMService&) = delete;

  void start() override;
  void stop() override;

  bool isModelReady() const override;

  void preProcess(domain::CompletionRequest& request) const override;

  /**
   * Abort an in-flight request. Removes the streaming callback, decrements
   * pending_tasks_, invokes the callback with finish_reason="abort" to unblock
   * synchronous waiters, and broadcasts cancel to all worker queues.
   * Idempotent and thread-safe.
   */
  void abortRequest(uint32_t taskId);

 protected:
  void postProcess(domain::CompletionResponse& response) const override;
  size_t currentQueueSize() const override;
  domain::CompletionResponse processRequest(
      domain::CompletionRequest request) override;

  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

  void streamingPostProcess(domain::StreamingChunkResponse&) const override {}
  void processStreamingRequest(
      domain::CompletionRequest request,
      std::function<void(domain::StreamingChunkResponse&, bool isFinal)>
          callback) override;

 private:
  void startConsumers();

  void consumerLoopForWorker(size_t workerIdx);

  std::vector<std::thread> consumer_threads_;

  ConcurrentMap<uint32_t,
                std::function<void(domain::StreamingChunkResponse&, bool)>>
      stream_callbacks_;

  std::atomic<uint64_t> next_worker_{0};

  std::atomic<size_t> pending_tasks_{0};

  std::atomic<bool> running_{false};

  std::unique_ptr<tt::ipc::QueueManager> queue_manager_;
  std::unique_ptr<tt::worker::WorkerManager> worker_manager_;
  const tt::utils::Tokenizer* tokenizer_;
  std::unique_ptr<ReasoningParser> reasoning_parser_;
};

}  // namespace tt::services
