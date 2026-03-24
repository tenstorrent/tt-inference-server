// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <drogon/drogon.h>
#include <json/json.h>

#include <memory>

#include "services/disaggregation_service.hpp"
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
  ADD_METHOD_TO(LLMController::chatCompletions, "/v1/chat/completions",
                drogon::Post);
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
  void chatCompletions(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

 private:
  std::shared_ptr<services::LLMService> service;
  std::shared_ptr<services::DisaggregationService> disaggregationService;

  /**
   * Handle streaming completion (SSE). When is_chat is true, emits
   * ChatCompletionStreamChunk objects; otherwise StreamingChunkResponse.
   * Automatically uses accumulated batching when enabled via config.
   */
  void handleStreaming(
      std::shared_ptr<domain::CompletionRequest> reqPtr,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback,
      bool isChat) const;

  /**
   * Generate a unique completion ID (hex string).
   */
  static std::string generateCompletionId();

  /**
   * Build OpenAI-style error JSON (flat object/message/type/param/code).
   */
  static Json::Value errorJson(const std::string& message,
                               const std::string& type,
                               const Json::Value& param = Json::nullValue,
                               const Json::Value& code = Json::nullValue);
};

}  // namespace tt::api
