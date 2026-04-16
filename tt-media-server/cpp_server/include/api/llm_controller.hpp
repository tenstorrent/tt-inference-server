// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <drogon/drogon.h>
#include <json/json.h>

#include <memory>

#include "services/disaggregation_service.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"

namespace tt::api {

/**
 * LLM API Controller - OpenAI-compatible chat completions endpoint.
 * Similar to Python's open_ai_api/llm.py router.
 */
class LLMController : public drogon::HttpController<LLMController> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(LLMController::chatCompletions, "/v1/chat/completions",
                drogon::Post);
  ADD_METHOD_TO(LLMController::createSession, "/v1/sessions", drogon::Post);
  ADD_METHOD_TO(LLMController::closeSession, "/v1/sessions/{session_id}",
                drogon::Delete);
  ADD_METHOD_TO(LLMController::getSlotId, "/v1/sessions/{session_id}/slot",
                drogon::Get);
  ADD_METHOD_TO(LLMController::models, "/v1/models",
                drogon::Get);
  METHOD_LIST_END

  LLMController();
  

  void models(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

  /**
   * POST /v1/chat/completions
   * OpenAI-compatible chat completions endpoint.
   */
  void chatCompletions(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

  /**
   * POST /v1/sessions
   * Create a new session with optional slot assignment.
   */
  void createSession(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

  /**
   * DELETE /v1/sessions/{session_id}
   * Close an existing session.
   */
  void closeSession(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback,
      const std::string& sessionId) const;

  /**
   * GET /v1/sessions/{session_id}/slot
   * Get the slot ID for a session.
   */
  void getSlotId(const drogon::HttpRequestPtr& req,
                 std::function<void(const drogon::HttpResponsePtr&)>&& callback,
                 const std::string& sessionId) const;

 private:
  std::shared_ptr<services::LLMService> service;
  std::shared_ptr<services::DisaggregationService> disaggregationService;
  std::shared_ptr<services::SessionManager> sessionManager;

  /**
   * Handle streaming chat completion (SSE). Emits ChatCompletionStreamChunk
   * objects. Automatically uses accumulated batching when enabled via config.
   */
  void handleStreaming(
      std::shared_ptr<domain::LLMRequest> reqPtr,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

  struct SessionInfo {
    bool validSessionFound = false;
  };

  enum class SessionErrorType {
    RATE_LIMIT,      // Returns 429 Too Many Requests
    ALLOCATION_FAIL  // Returns 503 Service Unavailable
  };

  struct SessionError {
    SessionErrorType type;
    std::string message;
  };

  /**
   * Validate/create session, assign slot, populate request fields.
   * Throws std::runtime_error if session creation fails.
   */
  void resolveSession(std::shared_ptr<domain::LLMRequest> req,
                      trantor::EventLoop* loop,
                      std::function<void(SessionInfo)> onResolved,
                      std::function<void(const SessionError&)> onError) const;

  /**
   * Determine if disaggregated prefill should be used for this request.
   */
  bool shouldDoPrefillOnDecode(const domain::LLMRequest& request,
                               bool validSessionFound) const;
};

}  // namespace tt::api
