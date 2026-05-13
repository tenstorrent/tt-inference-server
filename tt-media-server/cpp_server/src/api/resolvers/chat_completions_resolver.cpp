// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/resolvers/chat_completions_resolver.hpp"

#include <string>
#include <utility>

#include "domain/session.hpp"
#include "metrics/metrics.hpp"
#include "services/session_manager.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::api::resolvers {

ChatCompletionsResolver::ChatCompletionsResolver(
    std::shared_ptr<services::SessionManager> manager)
    : sessionManager(std::move(manager)) {}

void ChatCompletionsResolver::resolve(
    const std::vector<domain::llm::ChatMessage>& messages,
    trantor::EventLoop* loop, std::function<void(ResolvedSession)> onDone,
    std::function<void(const SessionError&)> onError,
    std::function<void()> cancelFn) const {
  if (!sessionManager) {
    TT_LOG_WARN("[ChatCompletionsResolver] SessionManager not available");
    onDone(ResolvedSession{});
    return;
  }

  auto routingInfo = tt::utils::computePrefixCachingInfo(messages);
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
        sessionManager->registerPrefixHash(acquired->sessionId,
                                           routingInfo.registrationHash);

        ResolvedSession resolved;
        resolved.sessionId = acquired->sessionId;
        resolved.slotId = acquired->slotId;
        resolved.session = sessionManager->getSession(acquired->sessionId);
        resolved.isFresh = false;
        resolved.registrationHash = routingInfo.registrationHash;
        resolved.prompt = routingInfo.deltaPrompt;
        resolved.promptTokensCount =
            static_cast<int>(tt::utils::tokenizers::activeTokenizer()
                                 .encode(routingInfo.deltaPrompt)
                                 .size());
        onDone(std::move(resolved));
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
  sessionManager->createSession(
      [registrationHash = routingInfo.registrationHash,
       cancelFn = std::move(cancelFn), mgr = sessionManager,
       onDone = std::move(onDone)](const tt::domain::Session& session) mutable {
        ResolvedSession resolved;
        resolved.sessionId = session.getSessionId();
        resolved.slotId =
            mgr->acquireInFlight(session.getSessionId(), std::move(cancelFn));
        resolved.session = mgr->getSession(session.getSessionId());
        resolved.isFresh = true;
        resolved.registrationHash = registrationHash;

        mgr->registerPrefixHash(session.getSessionId(), registrationHash);
        TT_LOG_INFO(
            "[ChatCompletionsResolver] New session: sessionId={}, slotId={}, "
            "registered under hash={}",
            session.getSessionId(),
            resolved.slotId.has_value() ? std::to_string(*resolved.slotId)
                                        : "none",
            registrationHash);

        onDone(std::move(resolved));
      },
      [onError = std::move(onError)](std::string_view err) {
        onError({SessionErrorType::ALLOCATION_FAIL, std::string(err)});
      },
      loop, routingInfo.registrationHash);
}

}  // namespace tt::api::resolvers
