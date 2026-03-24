// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "config/types.hpp"
#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"
#include "domain/prefill_request.hpp"
#include "ipc/queue_manager.hpp"
#include "services/base_service.hpp"
#include "services/reasoning_parser.hpp"
#include "services/streamable.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/concurrent_map.hpp"
#include "utils/tokenizer.hpp"
#include "worker/worker_manager.hpp"

namespace tt::services {

class LLMService
    : public BaseService<domain::CompletionRequest, domain::CompletionResponse>,
      public Streamable<domain::CompletionRequest,
                        domain::StreamingChunkResponse> {
 public:
  using PrefillRequestCallback =
      std::function<bool(const domain::PrefillRequest&)>;

  LLMService();
  ~LLMService() override;

  LLMService(const LLMService&) = delete;
  LLMService& operator=(const LLMService&) = delete;

  void start() override;
  void stop() override;

  bool isModelReady() const override;

  using StreamCallback =
      std::function<void(domain::StreamingChunkResponse&, bool)>;
  std::optional<StreamCallback> detachStreamCallback(const std::string& taskId);
  void submitDecodeContinuation(domain::CompletionRequest request,
                                StreamCallback callback);

  void handleConnectionLost();

  void setPrefillRequestCallback(PrefillRequestCallback callback);

  std::shared_ptr<tt::sockets::InterServerService> getSocketService() const;

 protected:
  void preProcess(domain::CompletionRequest& request) const override;
  void postProcess(domain::CompletionResponse&) const override {}
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

  tt::config::LLMMode mode_;

  std::vector<std::thread> consumer_threads_;

  ConcurrentMap<std::string,
                std::function<void(domain::StreamingChunkResponse&, bool)>>
      stream_callbacks_;

  std::atomic<uint64_t> next_worker_{0};

  std::atomic<size_t> pending_tasks_{0};

  std::atomic<bool> running_{false};

  std::unique_ptr<tt::ipc::QueueManager> queue_manager_;
  std::unique_ptr<tt::worker::WorkerManager> worker_manager_;
  const tt::utils::Tokenizer* tokenizer_;
  std::shared_ptr<tt::sockets::InterServerService> socket_service_;
  std::unique_ptr<ReasoningParser> reasoning_parser_;

  PrefillRequestCallback prefill_request_callback_;
};

}  // namespace tt::services
