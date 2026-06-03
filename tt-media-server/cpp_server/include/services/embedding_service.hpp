// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"
#include "services/request_pipeline.hpp"

namespace tt::services {

/**
 * Service for handling embedding requests.
 *
 * Uses a multiprocess scheduler with EmbeddingRunner workers.
 * Synchronous: submit_request blocks until the embedding is computed.
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

 protected:
  size_t currentQueueSize() const override;

  domain::EmbeddingResponse produceResponse(
      domain::EmbeddingRequest request) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::services
