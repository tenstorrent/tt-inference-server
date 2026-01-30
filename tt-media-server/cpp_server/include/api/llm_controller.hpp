// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <drogon/drogon.h>
#include <memory>

#include "services/llm_service.hpp"

namespace tt::api {

/**
 * LLM API Controller - OpenAI-compatible completions endpoint.
 * Similar to Python's open_ai_api/llm.py router.
 */
class LLMController : public drogon::HttpController<LLMController> {
public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(LLMController::completions, "/v1/completions", drogon::Post);
    ADD_METHOD_TO(LLMController::health, "/health", drogon::Get);
    ADD_METHOD_TO(LLMController::ready, "/ready", drogon::Get);
    METHOD_LIST_END

    LLMController();

    /**
     * POST /v1/completions
     * OpenAI-compatible text completion endpoint.
     */
    void completions(
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    /**
     * GET /health
     * Health check endpoint.
     */
    void health(
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    /**
     * GET /ready
     * Readiness check endpoint.
     */
    void ready(
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

private:
    std::shared_ptr<services::LLMService> service_;

    /**
     * Handle non-streaming completion request.
     */
    void handle_non_streaming(
        const domain::CompletionRequest& request,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    /**
     * Handle streaming completion request (SSE).
     */
    void handle_streaming(
        const domain::CompletionRequest& request,
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    /**
     * Generate a unique completion ID.
     */
    static std::string generate_completion_id();
};

} // namespace tt::api
