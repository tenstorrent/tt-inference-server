// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "api/response_writer/response_writer.hpp"

namespace tt::api {

/**
 * Response writer that accumulates chunks into a single JSON response body.
 *
 * The non-streaming counterpart to StreamingResponseWriter: the controller
 * drives the same streaming producer (LLMService::submitStreamingRequest or
 * the disaggregation service) and forwards every chunk here instead of out to
 * SSE. Tool-call streaming deltas are accumulated into a single `tool_calls`
 * array. On the final chunk the writer builds the assembled `LLMResponse`,
 * hands it to a caller-supplied builder for JSON encoding, releases the
 * session in-flight slot, and invokes the http callback exactly once.
 *
 * The default builder produces an OpenAI chat-completion JSON body. The
 * Responses API endpoint passes a builder that produces a `ResponsesResponse`
 * body instead, so both endpoints can reuse the same accumulator.
 */
class NonStreamResponseWriter : public ResponseWriter {
 public:
  using HttpCallback = std::function<void(const drogon::HttpResponsePtr&)>;
  /** Build the JSON body to return to the client. */
  using ResponseBuilder = std::function<std::string(const LLMResponse&)>;

  /** Default factory: builds an OpenAI chat-completion JSON body. */
  static std::shared_ptr<NonStreamResponseWriter> create(
      ResponseWriterParams params, HttpCallback httpCallback);

  /** Strategy factory: caller supplies the JSON body builder. */
  static std::shared_ptr<NonStreamResponseWriter> create(
      ResponseWriterParams params, HttpCallback httpCallback,
      ResponseBuilder builder);

  void handleTokenChunk(const LLMStreamChunk& chunk) override;
  void finalize() override;

  /**
   * Send an OpenAI-compatible error response and tear down. Used for failures
   * detected after the writer is created but before the response would normally
   * be built (e.g. dispatch throws).
   */
  void sendError(drogon::HttpStatusCode status, const std::string& message,
                 const std::string& type);

 private:
  NonStreamResponseWriter(ResponseWriterParams params,
                          HttpCallback httpCallback, ResponseBuilder builder);

  // Accumulated tool call data from streaming deltas
  struct AccumulatedToolCall {
    std::string id;
    std::string name;
    std::ostringstream arguments;
  };

  HttpCallback httpCallback;
  ResponseBuilder builder;

  std::ostringstream accumulatedAnswer;
  std::ostringstream accumulatedReasoning;
  std::vector<AccumulatedToolCall> accumulatedToolCalls;
  std::string finishReason = "stop";
};

}  // namespace tt::api
