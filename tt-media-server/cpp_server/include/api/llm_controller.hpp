// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>
#include <json/json.h>

#include <functional>
#include <memory>
#include <string>

#include "api/response_writer/non_stream_response_writer.hpp"
#include "api/response_writer/response_writer.hpp"
#include "api/stream_event_formatter.hpp"
#include "config/settings.hpp"
#include "domain/models_response.hpp"
#include "services/disaggregation_service.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"

namespace tt::sockets {
class InterServerService;
}

namespace tt::api {

/**
 * LLM API Controller - OpenAI-compatible chat completions, responses, and
 * session-management endpoints. Similar to Python's open_ai_api/llm.py router.
 */
class LLMController : public drogon::HttpController<LLMController> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(LLMController::chatCompletions, "/v1/chat/completions",
                drogon::Post);
  ADD_METHOD_TO(LLMController::responses, "/v1/responses", drogon::Post);
  ADD_METHOD_TO(LLMController::models, "/v1/models", drogon::Get);
  METHOD_LIST_END

  LLMController();

  void models(
      const drogon::HttpRequestPtr&,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
    domain::ModelsResponse response;
    response.data.push_back({toString(tt::config::model())});
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response.toJson());
    callback(resp);
  }

  /**
   * POST /v1/chat/completions
   * OpenAI-compatible chat completions endpoint.
   */
  void chatCompletions(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

  /**
   * POST /v1/responses
   * OpenAI-compatible responses endpoint.
   */
  void responses(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

 private:
  std::shared_ptr<services::LLMService> service;
  std::shared_ptr<services::DisaggregationService> disaggregationService;
  std::shared_ptr<services::SessionManager> sessionManager;
  std::shared_ptr<sockets::InterServerService> socketService;

  /**
   * Handle streaming responses (SSE). The provided `formatter` decides the
   * SSE wire format (chat.completion.chunk vs Responses API events). When
   * `formatter` is null, the writer falls back to ChatCompletionEventFormatter.
   */
  void handleStreaming(
      std::shared_ptr<LLMRequest> reqPtr,
      std::shared_ptr<StreamEventFormatter> formatter, bool includeUsage,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

  /**
   * Handle non-streaming responses. Drives the same Streamable producer as
   * handleStreaming and accumulates chunks into a single JSON body, so
   * disaggregated and prefill-on-decode routing is honored identically. The
   * `builder` converts the accumulated LLMResponse into the wire format
   * (chat-completion JSON by default; Responses API JSON for /v1/responses).
   */
  void handleNonStreaming(
      std::shared_ptr<LLMRequest> reqPtr,
      NonStreamResponseWriter::ResponseBuilder builder,
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
   * Validate/create session, mark it in-flight, and populate request fields.
   * cancelFn is stored atomically with the in-flight state so that a concurrent
   * closeSession always has a consistent view. Both streaming and non-streaming
   * paths pass a cancelFn; when closeSession fires mid-flight the client
   * receives finish_reason="abort" (partial response for non-streaming).
   */
  void resolveSession(std::shared_ptr<LLMRequest> req, trantor::EventLoop* loop,
                      std::function<void(SessionInfo)> onResolved,
                      std::function<void(const SessionError&)> onError,
                      std::function<void()> cancelFn = nullptr) const;

  /**
   * Determine if disaggregated prefill should be used for this request.
   */
  bool shouldDoPrefillOnDecode(const LLMRequest& request,
                               bool validSessionFound) const;

  /**
   * Submit the request to the appropriate streaming producer based on
   * llm_mode (REGULAR vs DECODE_ONLY) and the prefill-on-decode heuristic.
   * Caller must invoke service->preProcess(req) beforehand. Throws on
   * unsupported mode or queue/dispatch failures.
   */
  void dispatchGeneration(
      LLMRequest& request, bool validSessionFound,
      const std::function<void(const LLMStreamChunk&, bool)>& cb) const;

  /**
   * Translate a SessionError into a drogon HTTP error response.
   */
  static drogon::HttpResponsePtr makeSessionErrorResponse(
      const SessionError& err);

  /**
   * Build the ResponseWriterParams shared by both streaming and non-streaming
   * writers.
   */
  ResponseWriterParams makeWriterParams(const LLMRequest& request) const;

  /**
   * Build the streaming callback that pumps LLMStreamChunks into a
   * ResponseWriter. Common to both streaming and non-streaming code paths.
   */
  static std::function<void(const LLMStreamChunk&, bool)> makeStreamingCallback(
      std::shared_ptr<ResponseWriter> writer, domain::Session* session);
};

}  // namespace tt::api
