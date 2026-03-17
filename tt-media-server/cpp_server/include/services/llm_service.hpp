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
#include "services/streamable.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/concurrent_map.hpp"
#include "utils/tokenizer.hpp"
#include "worker/single_process_worker.hpp"

namespace tt::services {

worker::WorkerConfig makeWorkerConfigForProcess(int workerId);

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

  bool is_model_ready() const override;
  SystemStatus get_system_status() const override;

  using StreamCallback =
      std::function<void(domain::StreamingChunkResponse&, bool)>;
  std::optional<StreamCallback> detachStreamCallback(
      const std::string& taskId);
  void submitDecodeContinuation(domain::CompletionRequest request,
                               StreamCallback callback);

  void handleConnectionLost();

  void setPrefillRequestCallback(PrefillRequestCallback callback);

  std::shared_ptr<tt::sockets::InterServerService> getSocketService() const;

 protected:
  void pre_process(domain::CompletionRequest& request) const override;
  void post_process(domain::CompletionResponse& response) const override;
  size_t current_queue_size() const override;
  domain::CompletionResponse process_request(
      domain::CompletionRequest request) override;

  void streaming_post_process(domain::StreamingChunkResponse&) const override {}
  void process_streaming_request(
      domain::CompletionRequest request,
      std::function<void(domain::StreamingChunkResponse&, bool isFinal)>
          callback) override;

 private:
  void startWorkers();
  void startConsumers();

  void consumerLoopForWorker(size_t workerIdx);

  bool checkWorkerAlive(size_t workerIdx);

  tt::config::LLMMode mode;

  std::vector<std::unique_ptr<worker::SingleProcessWorker>> workers;
  size_t numWorkers;

  std::vector<std::thread> consumerThreads;

  ConcurrentMap<std::string,
                std::function<void(domain::StreamingChunkResponse&, bool)>>
      streamCallbacks;

  std::atomic<uint64_t> nextWorker{0};

  std::atomic<size_t> pendingTasks{0};

  std::atomic<bool> isReady{false};
  std::atomic<bool> running{false};

  std::string device = "cpu";

  std::unique_ptr<tt::ipc::QueueManager> queueManager;
  const tt::utils::Tokenizer* tokenizer;
  std::shared_ptr<tt::sockets::InterServerService> socketService;

  PrefillRequestCallback prefillRequestCallback;
};

}  // namespace tt::services
