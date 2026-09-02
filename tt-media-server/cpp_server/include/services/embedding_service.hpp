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
 * The HTTP controller uses the asynchronous path (submitRequestAsync): the
 * request is queued and the caller returns immediately; the worker dispatch
 * thread invokes the completion callback with the response. The inherited
 * synchronous submitRequest still works via an adapter and blocks the caller.
 */
class EmbeddingService : public BaseSyncService<domain::EmbeddingRequest,
                                                domain::EmbeddingResponse> {
 public:
  EmbeddingService();
  ~EmbeddingService() override;

  EmbeddingService(const EmbeddingService&) = delete;
  EmbeddingService& operator=(const EmbeddingService&) = delete;

  void start() override;
  void stop() override;
  bool isModelReady() const override;

  /**
   * Enqueue the request and return immediately; a worker dispatch thread
   * invokes onComplete (exactly once) with the response, from that dispatch
   * thread. The queue-capacity check runs synchronously here, so
   * QueueFullException propagates to the caller (mapped to HTTP 429) and is
   * never reported through onComplete.
   */
  void submitRequestAsync(
      domain::EmbeddingRequest request,
      std::function<void(domain::EmbeddingResponse&&)> onComplete);

 protected:
  size_t currentQueueSize() const override;

  /** Real per-worker liveness/readiness for /health and /tt-liveness; without
   * this the health endpoints report an empty worker list and external
   * harnesses see "0/0 workers ready". */
  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

  domain::EmbeddingResponse produceResponse(
      domain::EmbeddingRequest request) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::services
