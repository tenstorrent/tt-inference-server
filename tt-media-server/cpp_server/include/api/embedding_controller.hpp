// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <drogon/HttpController.h>

#include <atomic>
#include <memory>

#include "services/embedding_service.hpp"

namespace tt::api {

/**
 * OpenAI-compatible embedding API controller.
 *
 * Endpoints:
 * - POST /v1/embeddings - Create embeddings
 * - GET /health - Health check
 * - GET /tt-liveness - Liveness check
 */
class EmbeddingController : public drogon::HttpController<EmbeddingController> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(EmbeddingController::createEmbedding, "/v1/embeddings",
                drogon::Post);
  METHOD_LIST_END

  EmbeddingController();
  ~EmbeddingController();

  /**
   * POST /v1/embeddings
   * Create embeddings for the provided input text.
   */
  void createEmbedding(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback);

 private:
  std::shared_ptr<services::EmbeddingService> service_;
  std::atomic<uint64_t> request_counter_{0};
};

}  // namespace tt::api
