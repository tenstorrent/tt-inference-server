// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <functional>
#include <memory>

#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"
#include "services/request_pipeline.hpp"

namespace tt::services {

/**
 * Embedding request pipeline over forked worker processes. Async-only:
 * submitRequestAsync queues the request and a dispatch thread completes it.
 */
class EmbeddingService : public RequestPipeline<domain::EmbeddingRequest> {
 public:
  EmbeddingService();
  ~EmbeddingService() override;

  EmbeddingService(const EmbeddingService&) = delete;
  EmbeddingService& operator=(const EmbeddingService&) = delete;

  void start() override;
  void stop() override;
  bool isModelReady() const override;

  /** Enqueue and return; onComplete fires exactly once from a dispatch
   * thread. The queue-capacity check runs synchronously, so
   * QueueFullException (HTTP 429) propagates to the caller. */
  void submitRequestAsync(
      domain::EmbeddingRequest request,
      std::function<void(domain::EmbeddingResponse&&)> onComplete);

 protected:
  size_t currentQueueSize() const override;

  /** Per-worker liveness/readiness for the health endpoints. */
  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::services
