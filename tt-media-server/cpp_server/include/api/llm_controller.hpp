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
#include "services/llm_pipeline.hpp"
#include "services/llm_service.hpp"

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
  std::shared_ptr<services::LLMPipeline> pipeline;

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

  /**
   * Translate a pipeline session error into a drogon HTTP error response.
   * `traceId` is echoed back as `X-Request-Id` so failed requests stay
   * correlatable with logs.
   */
  static drogon::HttpResponsePtr makeSessionErrorResponse(
      const services::LLMPipeline::SessionError& err,
      const std::string& traceId);

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

  /**
   * Resolve the trace id for an inbound HTTP request: returns the inbound
   * `X-Request-Id` (or, if absent/empty, a freshly generated trace id).
   * Centralized so all entry points stay consistent.
   */
  static std::string resolveTraceId(const drogon::HttpRequestPtr& req);
};

}  // namespace tt::api
