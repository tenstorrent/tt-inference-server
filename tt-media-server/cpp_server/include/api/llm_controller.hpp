// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>
#include <json/json.h>

#include <functional>
#include <memory>
#include <string>

#include "api/resolvers/resolved_session.hpp"
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

namespace tt::api::resolvers {
class ChatCompletionsResolver;
}  // namespace tt::api::resolvers

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
  std::shared_ptr<resolvers::ChatCompletionsResolver> chatResolver;

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
    std::optional<size_t> registrationHash;
  };

  /**
   * Apply a resolver decision onto the in-flight request. Mirrors the
   * field writes that `LLMController::resolveSession` used to perform
   * inline, kept here as a private helper so both streaming and
   * non-streaming paths stay in sync.
   */
  static void applyResolvedSession(LLMRequest& request,
                                   const resolvers::ResolvedSession& resolved);

  /**
   * Build the SessionInfo consumed by dispatchGeneration from the
   * resolver's decision. registrationHash is only forwarded on a
   * prefix-cache HIT; fresh allocations and the no-manager path leave it
   * unset, preserving the original (pre-resolver) controller behavior.
   */
  static SessionInfo makeSessionInfo(
      const resolvers::ResolvedSession& resolved);

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
      LLMRequest& request, SessionInfo sessionInfo,
      const std::function<void(const LLMStreamChunk&, bool)>& cb) const;

  /**
   * Translate a SessionError into a drogon HTTP error response.
   */
  static drogon::HttpResponsePtr makeSessionErrorResponse(
      const resolvers::SessionError& err);

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
