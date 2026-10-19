// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"
#include "ipc/embedding_ipc.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/request_pipeline.hpp"

namespace tt::services {

/**
 * Embedding request pipeline over WorkerManager + Boost shared-memory IPC
 * queues.
 *
 * Parent thread validates requests, enforces maxQueueSize synchronously (429),
 * pushes EmbeddingIpcTask messages to the shared task queue, and completes
 * callbacks from per-worker result-queue consumer threads.
 */
class EmbeddingService : public BaseAsyncService<domain::EmbeddingRequest,
                                                 domain::EmbeddingResponse> {
 public:
  EmbeddingService();
  EmbeddingService(
      config::EmbeddingConfig config,
      std::unique_ptr<tt::worker::WorkerManager> workerManager,
      std::unique_ptr<tt::ipc::embedding::EmbeddingQueueSet> queueSet);
  ~EmbeddingService() override;

  EmbeddingService(const EmbeddingService&) = delete;
  EmbeddingService& operator=(const EmbeddingService&) = delete;

  void start() override;
  void stop() override;
  bool isModelReady() const override;

 protected:
  void produceResponseAsync(domain::EmbeddingRequest request,
                            Callback onComplete) override;
  size_t currentQueueSize() const override;
  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::services
