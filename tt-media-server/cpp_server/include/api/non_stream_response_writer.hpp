// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>

#include <functional>
#include <memory>
#include <string>

#include "api/response_writer.hpp"

namespace tt::api {

/**
 * Response writer that accumulates chunks into a single ChatCompletionResponse.
 *
 * The non-streaming counterpart to StreamingResponseWriter: the controller
 * drives the same Streamable producer (LLMService::submitStreamingRequest or
 * the disaggregation service) and forwards every chunk here instead of out to
 * SSE. On the final chunk we build the full ChatCompletionResponse, run the
 * service's postProcess (reasoning strip + tool-call parsing — non-streaming
 * only), release the session in-flight slot, and invoke the http callback
 * exactly once.
 */
class NonStreamResponseWriter : public ResponseWriter {
 public:
  using HttpCallback = std::function<void(const drogon::HttpResponsePtr&)>;

  static std::shared_ptr<NonStreamResponseWriter> create(
      ResponseWriterParams params, HttpCallback httpCallback);

  void handleTokenChunk(const domain::LLMStreamChunk& chunk) override;
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
                          HttpCallback httpCallback);

  HttpCallback httpCallback;

  std::string accumulatedAnswer;
  std::string accumulatedReasoning;
  std::string finishReason = "stop";
};

}  // namespace tt::api
