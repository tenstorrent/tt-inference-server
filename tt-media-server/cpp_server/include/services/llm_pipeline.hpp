// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"

namespace trantor {
class EventLoop;
}

namespace tt::services {
class DisaggregationService;
class LLMService;
class SessionManager;
}  // namespace tt::services

namespace tt::sockets {
class InterServerService;
}

namespace tt::services {

/**
 * Front-end-agnostic glue around the LLM stack: prefix-cache routing, session
 * management, and dispatch to either the in-process LLMService or the
 * disaggregated prefill path.
 *
 * Both `LLMController` (HTTP /v1/chat/completions, /v1/responses) and
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
   * Submit `request` to the appropriate streaming producer based on
   * `LLM_MODE` (REGULAR vs DECODE_ONLY) and the prefill-on-decode heuristic.
   * Caller must invoke `service->preProcess(req)` (or set `skipPreProcess`
   * upstream) before calling this. Throws on unsupported mode or queue/dispatch
   * failures.
   */
  void dispatchGeneration(
      tt::domain::llm::LLMRequest& request, SessionInfo sessionInfo,
      const std::function<void(const tt::domain::llm::LLMStreamChunk&, bool)>&
          cb) const;

  void abortRequest(uint32_t taskId) const;

  std::shared_ptr<LLMService> service() const { return service_; }
  std::shared_ptr<SessionManager> sessionManager() const {
    return sessionManager_;
  }

 private:
  bool shouldDoPrefillOnDecode(const tt::domain::llm::LLMRequest& request,
                               bool validSessionFound) const;

  std::shared_ptr<LLMService> service_;
  std::shared_ptr<SessionManager> sessionManager_;
  std::shared_ptr<DisaggregationService> disaggregationService_;
  std::shared_ptr<sockets::InterServerService> socketService_;
};

}  // namespace tt::services
