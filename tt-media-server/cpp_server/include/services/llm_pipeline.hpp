// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"

namespace trantor {
class EventLoop;
}

namespace tt::domain {
class Session;
}

namespace tt::services {
class DisaggregationService;
class LLMService;
class SessionManager;
}  // namespace tt::services

namespace tt::sockets {
class InterServerService;
struct PrefillRequestMessage;
struct PrefillResultMessage;
}

namespace tt::services {

/**
 * Front-end-agnostic glue around the LLM stack: prefix-cache routing, session
 * management, and dispatch to either the in-process LLMService or the
 * disaggregated prefill path.
 *
 * Both `LLMController` (HTTP /v1/chat/completions) and
 * `tt::dynamo::DynamoEndpoint` (TCP `generate`) drive the same pipeline so
 * Dynamo requests benefit from the same session/prefix-cache reuse and
 * disaggregation routing as HTTP traffic.
 *
 * Pipeline is stateless aside from its dependency pointers; constructing
 * multiple instances that share the same dependencies is harmless.
 */
class LLMPipeline {
 public:
  enum class SessionErrorType {
    RATE_LIMIT,      // Maps to HTTP 429.
    ALLOCATION_FAIL  // Maps to HTTP 503.
  };

  struct SessionError {
    SessionErrorType type;
    std::string message;
  };

  struct SessionInfo {
    bool validSessionFound = false;
    std::vector<uint64_t> registrationHashes;
  };

  using StreamCallback =
      std::function<void(const tt::domain::llm::LLMStreamChunk&, bool)>;
  using StreamCallbackFactory = std::function<StreamCallback(
      SessionInfo, std::shared_ptr<tt::domain::Session>)>;

  struct GenerationHandlers {
    std::function<void(SessionInfo)> onSessionResolved;
    std::function<void()> onPreProcessed;
    std::function<void()> onDispatchSucceeded;
    std::function<void(const std::exception&,
                       std::shared_ptr<tt::domain::Session>)>
        onPreProcessError;
    std::function<void(const std::exception&,
                       std::shared_ptr<tt::domain::Session>)>
        onDispatchError;
    std::function<void(const SessionError&)> onSessionError;
  };

  LLMPipeline(std::shared_ptr<LLMService> service,
              std::shared_ptr<SessionManager> sessionManager,
              std::shared_ptr<DisaggregationService> disaggregationService,
              std::shared_ptr<sockets::InterServerService> socketService);

  /**
   * Resolve (or create) a session for `req`, populating sessionId / slotId /
   * continuation / prompt fields, and computing the prefix-cache routing.
   *
   * Branches on the request shape:
   *   - When `req->messages` is non-empty (HTTP path) the chat-message hasher
   *     drives lookup; the delta prompt is a rendered string.
   *   - When `req->messages` is empty and `req->prompt` holds a token id
   *     vector (Dynamo path) the token hasher drives lookup; the delta
   *     prompt is a token id vector.
   *
   * `onResolved` runs on `loop`. `onError` is invoked when session allocation
   * is rejected (rate limit) or fails. `cancelFn` is stored alongside the
   * in-flight slot so a concurrent session close cancels the request cleanly.
   */
  void resolveSession(std::shared_ptr<tt::domain::llm::LLMRequest> req,
                      trantor::EventLoop* loop,
                      std::function<void(SessionInfo)> onResolved,
                      std::function<void(const SessionError&)> onError,
                      std::function<void()> cancelFn = nullptr) const;

  /**
   * Shared HTTP/Dynamo orchestration: resolve, preprocess, build the frontend
   * stream callback, and dispatch generation.
   */
  void runStreamingRequest(std::shared_ptr<tt::domain::llm::LLMRequest> req,
                           trantor::EventLoop* loop,
                           StreamCallbackFactory makeStreamCallback,
                           GenerationHandlers handlers,
                           std::function<void()> cancelFn = nullptr) const;

  /**
   * Submit `request` to the appropriate streaming producer based on
   * `LLM_MODE` (REGULAR vs DECODE_ONLY) and the prefill-on-decode heuristic.
   * Prefer `runStreamingRequest` from frontend code; direct callers must
   * preprocess first or arrange an equivalent skip upstream.
   * Throws on unsupported mode or queue/dispatch failures.
   */
  void dispatchGeneration(
      tt::domain::llm::LLMRequest& request, SessionInfo sessionInfo,
      const std::function<void(const tt::domain::llm::LLMStreamChunk&, bool)>&
          cb) const;

  void handlePrefillRequest(
      const tt::sockets::PrefillRequestMessage& message,
      std::function<void(const tt::sockets::PrefillResultMessage&)> onResult)
      const;
  void handlePrefillResult(
      const tt::sockets::PrefillResultMessage& message,
      const std::function<void(const tt::domain::llm::LLMStreamChunk&, bool)>&
          cb) const;

  void abortRequest(uint32_t taskId) const;

  std::shared_ptr<LLMService> service() const { return service_; }
  std::shared_ptr<SessionManager> sessionManager() const {
    return sessionManager_;
  }

 private:
  bool shouldDoPrefillOnDecode(
      const tt::domain::llm::LLMRequest& request) const;

  // Decide, given the uncached delta size, whether prefill runs locally on the
  // decode device (true) or is sent to the prefill server (false). Shared by
  // session resolution and dispatch so the slot-copy decision and the actual
  // prefill routing stay consistent.
  bool willPrefillOnDecode(const tt::domain::llm::LLMRequest& request,
                           size_t deltaTokens) const;

  std::shared_ptr<LLMService> service_;
  std::shared_ptr<SessionManager> sessionManager_;
  std::shared_ptr<DisaggregationService> disaggregationService_;
  std::shared_ptr<sockets::InterServerService> socketService_;
};

}  // namespace tt::services
