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
        std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;


    /**
     * POST /v1/chat/completions
     * OpenAI-compatible chat completions endpoint.
     */
    void chat_completions(
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

    /**
     * GET /health
     * Health check endpoint.
     */
    void health(
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

    /**
     * GET /ready
     * Readiness check endpoint.
     */
    void ready(
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

private:
    std::shared_ptr<services::LLMService> service_;

    /**
     * Handle streaming text completion request (SSE).
     */
    void handle_streaming(
        const domain::CompletionRequest& request,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

    /**
     * Handle streaming chat completion request (SSE).
     */
    void handle_chat_streaming(
        const domain::CompletionRequest& request,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

    /**
     * Handle non-streaming text completion: accumulates streamed tokens
     * into a single CompletionResponse JSON.
     */
    void handle_non_streaming(
        domain::CompletionRequest request,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

    /**
     * Handle non-streaming chat completion: accumulates streamed tokens
     * into a single ChatCompletionResponse JSON.
     */
    void handle_chat_non_streaming(
        domain::CompletionRequest request,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

    /**
     * Generate a unique completion ID (hex string).
     */
    static std::string generate_completion_id();

    /**
     * Build OpenAI-style error JSON (flat object/message/type/param/code).
     */
    static Json::Value error_json(const std::string& message, const std::string& type,
        const Json::Value& param = Json::nullValue, const Json::Value& code = Json::nullValue);
};

} // namespace tt::api
