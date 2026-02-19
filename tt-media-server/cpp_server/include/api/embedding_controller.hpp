// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <atomic>
#include <drogon/HttpController.h>

#include "services/embedding_service.hpp"

namespace tt::api {

using namespace std;

/**
 * OpenAI-compatible embedding API controller.
 *
 * Endpoints:
 * - POST /v1/embeddings - Create embeddings
 * - GET /health - Health check
 * - GET /ready - Readiness check
 */
class EmbeddingController : public drogon::HttpController<EmbeddingController> {
public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(EmbeddingController::create_embedding, "/v1/embeddings", drogon::Post);
    ADD_METHOD_TO(EmbeddingController::health, "/health", drogon::Get);
    ADD_METHOD_TO(EmbeddingController::ready, "/ready", drogon::Get);
    METHOD_LIST_END

    EmbeddingController();
    ~EmbeddingController();

    /**
     * POST /v1/embeddings
     * Create embeddings for the provided input text.
     */
    void create_embedding(
        const drogon::HttpRequestPtr& req,
        function<void(const drogon::HttpResponsePtr&)>&& callback);

    /**
     * GET /health
     * Basic health check.
     */
    void health(
        const drogon::HttpRequestPtr& req,
        function<void(const drogon::HttpResponsePtr&)>&& callback);

    /**
     * GET /ready
     * Detailed readiness check with system status.
     */
    void ready(
        const drogon::HttpRequestPtr& req,
        function<void(const drogon::HttpResponsePtr&)>&& callback);

private:
    shared_ptr<services::EmbeddingService> service_;
    atomic<uint64_t> request_counter_{0};

    string generate_task_id();
};

} // namespace tt::api
