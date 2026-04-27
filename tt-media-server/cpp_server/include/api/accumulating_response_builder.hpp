// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "domain/llm_response.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"

namespace tt::api {

struct AccumulatingResponseParams {
  std::string completionId;
  std::string model;
  int64_t created;
  int promptTokensCount;
  std::optional<std::string> sessionId;
  uint32_t taskId;
  std::shared_ptr<services::LLMService> service;
  std::shared_ptr<services::SessionManager> sessionManager;
};

/**
 * Accumulates streaming LLMStreamChunks into a single ChatCompletionResponse.
 *
 * The non-streaming counterpart to SseStreamWriter: the controller drives the
 * same Streamable producer (LLMService::submitStreamingRequest or the
 * disaggregation service) and forwards every chunk here instead of out to SSE.
 * On the final chunk we build the full ChatCompletionResponse, run the
 * service's postProcess (reasoning strip + tool-call parsing — non-streaming
 * only), release the session in-flight slot, and invoke the http callback
 * exactly once.
 */
class AccumulatingResponseBuilder
    : public std::enable_shared_from_this<AccumulatingResponseBuilder> {
 public:
  using HttpCallback = std::function<void(const drogon::HttpResponsePtr&)>;

  static std::shared_ptr<AccumulatingResponseBuilder> create(
      AccumulatingResponseParams params, HttpCallback httpCallback);

  AccumulatingResponseBuilder(const AccumulatingResponseBuilder&) = delete;
  AccumulatingResponseBuilder& operator=(const AccumulatingResponseBuilder&) =
      delete;

  void handleTokenChunk(const domain::LLMStreamChunk& chunk);
  void finalize();
  void sendError(drogon::HttpStatusCode status, const std::string& message,
                 const std::string& type);

  bool isDone() const { return done.load(); }

 private:
  AccumulatingResponseBuilder(AccumulatingResponseParams params,
                              HttpCallback httpCallback);

  domain::CompletionUsage buildFinalUsage() const;
  void releaseInFlight();

  AccumulatingResponseParams params;
  HttpCallback httpCallback;

  std::string accumulatedAnswer;
  std::string accumulatedReasoning;
  int completionTokens = 0;
  std::string finishReason = "stop";

  std::chrono::high_resolution_clock::time_point startTime =
      std::chrono::high_resolution_clock::now();
  std::optional<std::chrono::high_resolution_clock::time_point> firstTokenTime;
  std::optional<std::chrono::high_resolution_clock::time_point> secondTokenTime;

  std::atomic<bool> done{false};
};

}  // namespace tt::api
