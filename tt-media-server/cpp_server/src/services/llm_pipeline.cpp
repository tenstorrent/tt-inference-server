// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/llm_pipeline.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "config/settings.hpp"
#include "metrics/metrics.hpp"
#include "services/disaggregation_service.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::services {

LLMPipeline::LLMPipeline(
    std::shared_ptr<LLMService> service,
    std::shared_ptr<SessionManager> sessionManager,
    std::shared_ptr<DisaggregationService> disaggregationService,
    std::shared_ptr<sockets::InterServerService> socketService)
    : service_(std::move(service)),
      sessionManager_(std::move(sessionManager)),
      disaggregationService_(std::move(disaggregationService)),
      socketService_(std::move(socketService)) {}

namespace {

/**
 * Compute prefix-cache routing for the request, picking the message-level or
 * token-level hasher based on what the caller filled in.
 */
tt::utils::PrefixCachingInfo computeRoutingInfo(
    const tt::domain::llm::LLMRequest& req) {
  if (!req.messages.empty()) {
    return tt::utils::computePrefixCachingInfo(req.messages);
  }
  if (auto* tokens = std::get_if<std::vector<int>>(&req.prompt)) {
    return tt::utils::computePrefixCachingInfoFromTokens(*tokens);
  }
  return {};  // No messages and prompt is a string; skip prefix-cache lookup.
}

/**
 * Apply the resolved delta prompt to the request. The hasher returns either
 * a rendered string (message path) or a token id vector (Dynamo path); both
 * variants slot directly into `req.prompt`.
 */
void applyDeltaPrompt(tt::domain::llm::LLMRequest& req,
                      tt::utils::PrefixCachingInfo& info) {
  if (auto* s = std::get_if<std::string>(&info.deltaPrompt)) {
    req.prompt_tokens_count = static_cast<int>(
        tt::utils::tokenizers::activeTokenizer().encode(*s).size());
    req.prompt = std::move(*s);
  } else {
    auto& tok = std::get<std::vector<int>>(info.deltaPrompt);
    req.prompt_tokens_count = static_cast<int>(tok.size());
    req.prompt = std::move(tok);
  }
}

}  // namespace

void LLMPipeline::resolveSession(
    std::shared_ptr<tt::domain::llm::LLMRequest> req,
    trantor::EventLoop* loop, std::function<void(SessionInfo)> onResolved,
    std::function<void(const SessionError&)> onError,
    std::function<void()> cancelFn) const {
  SessionInfo info;

  if (!sessionManager_) {
    TT_LOG_WARN("[LLMPipeline] SessionManager not available");
    onResolved(info);
    return;
  }

  auto routingInfo = computeRoutingInfo(*req);
  TT_LOG_DEBUG(
      "[LLMPipeline] Routing: hasPriorTurn={}, lookupHash={}, "
      "registrationHash={}",
      routingInfo.hasPriorTurn,
      routingInfo.lookupHash.has_value()
          ? std::to_string(*routingInfo.lookupHash)
          : "none",
      routingInfo.registrationHash);

  // Layer 1: Prefix-cache routing. Requires a prior turn and a lookup hash.
  if (routingInfo.hasPriorTurn && routingInfo.lookupHash.has_value()) {
    try {
      auto acquired = sessionManager_->tryAcquireByPrefixHash(
          *routingInfo.lookupHash, cancelFn);

      if (acquired.has_value()) {
        tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(true);
        TT_LOG_DEBUG(
            "[LLMPipeline] Prefix cache HIT: hash={}, sessionId={}, "
            "slotId={}",
            *routingInfo.lookupHash, acquired->sessionId, acquired->slotId);
        req->slotId = acquired->slotId;
        req->session = sessionManager_->getSession(acquired->sessionId);
        req->continuation = true;
        applyDeltaPrompt(*req, routingInfo);
        sessionManager_->registerPrefixHash(acquired->sessionId,
                                            routingInfo.registrationHash);
        info.validSessionFound = true;
        info.registrationHash = routingInfo.registrationHash;
        onResolved(info);
        return;
      }

      tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(false);
      TT_LOG_DEBUG(
          "[LLMPipeline] Prefix cache MISS: hash={}, allocating new session",
          *routingInfo.lookupHash);
    } catch (const services::SessionInFlightException& e) {
      TT_LOG_WARN("[LLMPipeline] All sessions busy for hash={}: {}",
                  *routingInfo.lookupHash, e.what());
      onError({SessionErrorType::RATE_LIMIT, e.what()});
      return;
    }
  }

  // Layer 2: Allocate a new session. Async — onCompletion runs on `loop`.
  sessionManager_->createSession(
      [req, routingInfo, onResolved, cancelFn = std::move(cancelFn),
       mgr = sessionManager_](const tt::domain::Session& session) mutable {
        req->sessionId = session.getSessionId();
        req->slotId =
            mgr->acquireInFlight(session.getSessionId(), std::move(cancelFn));
        req->session = mgr->getSession(session.getSessionId());
        req->continuation = false;
        mgr->registerPrefixHash(session.getSessionId(),
                                routingInfo.registrationHash);
        TT_LOG_INFO(
            "[LLMPipeline] New session: sessionId={}, slotId={}, "
            "registered under hash={}",
            session.getSessionId(),
            req->slotId.has_value() ? std::to_string(*req->slotId) : "none",
            routingInfo.registrationHash);

        SessionInfo info;
        info.registrationHash = routingInfo.registrationHash;
        onResolved(info);
      },
      [onError](std::string_view err) {
        onError({SessionErrorType::ALLOCATION_FAIL, std::string(err)});
      },
      loop, routingInfo.registrationHash);
}

void LLMPipeline::dispatchGeneration(
    tt::domain::llm::LLMRequest& request, SessionInfo sessionInfo,
    const std::function<void(const tt::domain::llm::LLMStreamChunk&, bool)>& cb)
    const {
  const auto mode = tt::config::llmMode();
  if (mode == tt::config::LLMMode::REGULAR) {
    service_->submitStreamingRequest(request, cb, /*skipPreProcess=*/true);
    return;
  }

  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    if (shouldDoPrefillOnDecode(request, sessionInfo.validSessionFound)) {
      TT_LOG_DEBUG("[LLMPipeline] Using prefill on decode for sessionId: {}",
                   request.sessionId.value_or("none"));
      service_->submitStreamingRequest(request, cb, /*skipPreProcess=*/true);
    } else {
      TT_LOG_DEBUG(
          "[LLMPipeline] Using disaggregated prefill for request with "
          "sessionId: {}",
          request.sessionId.value_or("none"));
      disaggregationService_->handleStreamingRequest(
          request, sessionInfo.registrationHash.value_or(0), cb);
    }
    return;
  }

  throw std::runtime_error(
      "LLM Mode must be regular or decode only for chat completions");
}

bool LLMPipeline::shouldDoPrefillOnDecode(
    const tt::domain::llm::LLMRequest& request, bool validSessionFound) const {
  const bool socketReady = socketService_ && socketService_->isConnected();
  if (!socketReady) {
    TT_LOG_WARN(
        "[LLMPipeline] Prefill server not connected; falling back to "
        "prefill on decode for taskId={}",
        request.task_id);
    return true;
  }

  if (request.disaggregation_override.has_value()) {
    const bool forceDisagg = *request.disaggregation_override;
    TT_LOG_INFO(
        "[LLMPipeline] Honoring disaggregation override={} for taskId={}",
        forceDisagg, request.task_id);
    return !forceDisagg;
  }

  if (validSessionFound) {
    return true;
  }

  const size_t maxTokens = tt::config::maxTokensToPrefillOnDecode();
  const size_t promptTokens = static_cast<size_t>(request.prompt_tokens_count);

  return promptTokens < maxTokens;
}

}  // namespace tt::services
