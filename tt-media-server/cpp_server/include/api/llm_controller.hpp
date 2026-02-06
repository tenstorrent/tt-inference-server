// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <drogon/drogon.h>
#include <json/json.h>
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
    ADD_METHOD_TO(LLMController::chat_completions, "/v1/chat/completions", drogon::Post);
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
     * POST /v1/chat/completions
     * OpenAI-compatible chat completions endpoint.
     */
    void chat_completions(
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

    void handle_non_streaming(
        const domain::CompletionRequest& request,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    void handle_streaming(
        domain::CompletionRequest request,
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    void handle_streaming_buffered(
        domain::CompletionRequest request,
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    void handle_chat_non_streaming(
        const domain::CompletionRequest& request,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    void handle_chat_streaming(
        domain::CompletionRequest request,
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    static std::string generate_completion_id();

    using ResponseFormatter = std::function<Json::Value(const domain::CompletionResponse&)>;
    void run_async_completion(
        const domain::CompletionRequest& request,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback,
        ResponseFormatter formatter);
};

} // namespace tt::api
