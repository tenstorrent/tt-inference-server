// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/resolvers/chat_completions_resolver.hpp"

#include <string>
#include <utility>

#include "api/resolvers/prefix_caching.hpp"
#include "domain/session.hpp"
#include "metrics/metrics.hpp"
#include "services/session_manager.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::api::resolvers {

ChatCompletionsResolver::ChatCompletionsResolver(
    std::shared_ptr<services::SessionManager> manager)
    : sessionManager(std::move(manager)) {}

void ChatCompletionsResolver::resolve(
    std::shared_ptr<domain::llm::LLMRequest> request, trantor::EventLoop* loop,
    std::function<void()> onDone,
    std::function<void(const SessionError&)> onError,
    std::function<void()> cancelFn) const {
  if (!sessionManager) {
    TT_LOG_WARN("[ChatCompletionsResolver] SessionManager not available");
    onDone();
    return;
  }

  auto routingInfo = computePrefixCachingInfo(request->messages);
  TT_LOG_DEBUG(
      "[ChatCompletionsResolver] Routing: hasPriorTurn={}, lookupHash={}, "
      "registrationHash={}",
      routingInfo.hasPriorTurn,
      routingInfo.lookupHash.has_value()
          ? std::to_string(*routingInfo.lookupHash)
          : "none",
      routingInfo.registrationHash);

  // Layer 1: Prefix-cache routing. Requires a prior [assistant, user] pair.
  if (routingInfo.hasPriorTurn && routingInfo.lookupHash.has_value()) {
    try {
      auto acquired = sessionManager->tryAcquireByPrefixHash(
          *routingInfo.lookupHash, cancelFn);

      if (acquired.has_value()) {
        tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(true);
        TT_LOG_DEBUG(
            "[ChatCompletionsResolver] Prefix cache HIT: hash={}, "
            "sessionId={}, slotId={}",
            *routingInfo.lookupHash, acquired->sessionId, acquired->slotId);

        request->sessionId = acquired->sessionId;
        request->slotId = acquired->slotId;
        request->session = sessionManager->getSession(acquired->sessionId);
        request->continuation = true;
        request->registrationHash = routingInfo.registrationHash;
        request->prompt = routingInfo.deltaPrompt;
        request->prompt_tokens_count =
            static_cast<int>(tt::utils::tokenizers::activeTokenizer()
                                 .encode(routingInfo.deltaPrompt)
                                 .size());
        sessionManager->registerPrefixHash(acquired->sessionId,
                                           routingInfo.registrationHash);

        onDone();
        return;
      }

      tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(false);
      TT_LOG_DEBUG(
          "[ChatCompletionsResolver] Prefix cache MISS: hash={}, allocating "
          "new session",
          *routingInfo.lookupHash);
    } catch (const services::SessionInFlightException& e) {
      TT_LOG_WARN("[ChatCompletionsResolver] All sessions busy for hash={}: {}",
                  *routingInfo.lookupHash, e.what());
      onError({SessionErrorType::RATE_LIMIT, e.what()});
      return;
    }
  }

  // Layer 2: Allocate a new session. createSession is async — `onCompletion`
  // and `onError` are queued on `loop` by the SessionManager.
  // Fresh allocations leave `request->registrationHash` at 0 (the
  // disaggregation service treats unhashed requests as fresh prefix).
  sessionManager->createSession(
      [request = std::move(request),
       registrationHash = routingInfo.registrationHash,
       cancelFn = std::move(cancelFn), mgr = sessionManager,
       onDone = std::move(onDone)](const tt::domain::Session& session) mutable {
        request->sessionId = session.getSessionId();
        request->slotId =
            mgr->acquireInFlight(session.getSessionId(), std::move(cancelFn));
        request->session = mgr->getSession(session.getSessionId());
        request->continuation = false;
        mgr->registerPrefixHash(session.getSessionId(), registrationHash);

        TT_LOG_INFO(
            "[ChatCompletionsResolver] New session: sessionId={}, slotId={}, "
            "registered under hash={}",
            session.getSessionId(),
            request->slotId.has_value() ? std::to_string(*request->slotId)
                                        : "none",
            registrationHash);

        onDone();
      },
      [onError = std::move(onError)](std::string_view err) {
        onError({SessionErrorType::ALLOCATION_FAIL, std::string(err)});
      },
      loop, routingInfo.registrationHash);
}

}  // namespace tt::api::resolvers
