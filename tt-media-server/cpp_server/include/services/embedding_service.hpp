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
 * Service for handling embedding requests.
 *
 * Uses a multiprocess scheduler with EmbeddingRunner workers.
 * The request path is asynchronous only (submitRequestAsync): the request is
 * queued and the caller returns immediately; the worker dispatch thread
 * invokes the completion callback with the response. There is deliberately
 * no synchronous submitRequest — nothing consumes it, and a blocking API
 * would need a thread parked per in-flight request.
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

  void submitRequestAsync(
      domain::EmbeddingRequest request,
      std::function<void(domain::EmbeddingResponse&&)> onComplete);

 protected:
  size_t currentQueueSize() const override;

  /** Real per-worker liveness/readiness for /health and /tt-liveness; without
   * this the health endpoints report an empty worker list and external
   * harnesses see "0/0 workers ready". */
  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::services
