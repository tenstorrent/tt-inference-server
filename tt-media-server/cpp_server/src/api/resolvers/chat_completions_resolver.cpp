// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/resolvers/chat_completions_resolver.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define XXH_INLINE_ALL
#include "domain/llm/chat_message.hpp"
#include "domain/session.hpp"
#include "metrics/metrics.hpp"
#include "services/session_manager.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"
#include "xxhash.h"

namespace tt::api::resolvers {

namespace {

using domain::llm::ChatMessage;

// Drop "tool" and (legacy) "function" turns: they're deterministic
// outputs of the preceding assistant's tool_calls and shouldn't
// participate in prefix identity. System / developer messages stay
// intact since they belong to the stable prefix.
std::vector<ChatMessage> stripToolMessages(
    const std::vector<ChatMessage>& messages) {
  std::vector<ChatMessage> result;
  result.reserve(messages.size());
  for (const auto& msg : messages) {
    if (msg.role != "tool" && msg.role != "function") {
      result.push_back(msg);
    }
  }
  return result;
}

// Returns the prefix that would be used to LOOK UP a continuing
// session: everything except the trailing [assistant, user] pair (with
// tool/function turns stripped). nullopt means "no prior turn -- skip
// lookup, go straight to allocate".
std::optional<std::vector<ChatMessage>> extractPriorTurnPrefix(
    const std::vector<ChatMessage>& messages) {
  if (messages.empty() || messages.back().role != "user") return std::nullopt;
  auto turns = stripToolMessages(messages);
  if (turns.size() < 2) return std::nullopt;
  if (turns[turns.size() - 2].role != "assistant") return std::nullopt;

  std::vector<ChatMessage> prior(turns.begin(), turns.end() - 2);
  if (prior.empty()) return std::nullopt;
  return prior;
}

// Routing decision drawn purely from the message list -- no tokenizer.
struct PrefixRouting {
  std::optional<uint64_t> lookupHash;  // matches a registered prior-turn hash
  uint64_t registrationHash = 0;  // next-turn lookup key (current full conv)
  bool hasPriorTurn = false;
};

PrefixRouting computePrefixRouting(const std::vector<ChatMessage>& messages) {
  PrefixRouting r;
  auto turns = stripToolMessages(messages);
  r.registrationHash = ChatCompletionsResolver::hashMessages(turns);

  if (auto prior = extractPriorTurnPrefix(messages); prior.has_value()) {
    r.hasPriorTurn = true;
    r.lookupHash = ChatCompletionsResolver::hashMessages(*prior);
  }
  return r;
}

// Render the last user turn standalone, with addGenerationPrompt=true.
// This is the delta sent to the model when the slot's KV cache already
// holds the prior-turn prefix. Tokenizer usage lives here, in the
// resolver, not in the prefix-routing helpers above.
std::string renderLastUserTurn(const std::vector<ChatMessage>& messages) {
  auto it =
      std::find_if(messages.rbegin(), messages.rend(),
                   [](const ChatMessage& msg) { return msg.role == "user"; });
  if (it == messages.rend()) return std::string{};
  return tt::utils::tokenizers::activeTokenizer().applyChatTemplate({*it},
                                                                    true);
}

}  // namespace

uint64_t ChatCompletionsResolver::hashMessages(
    const std::vector<ChatMessage>& messages) {
  if (messages.empty()) return 0;

  XXH64_state_t* state = XXH64_createState();
  XXH64_reset(state, 0);
  // Null separators between role/content and between messages prevent
  // (role, content) boundary aliasing -- e.g., role="us"/content="er.."
  // vs role="user"/content=".." must not collide.
  const char nul = '\0';
  for (const auto& m : messages) {
    XXH64_update(state, m.role.data(), m.role.size());
    XXH64_update(state, &nul, 1);
    XXH64_update(state, m.content.data(), m.content.size());
    XXH64_update(state, &nul, 1);
  }
  uint64_t hash = XXH64_digest(state);
  XXH64_freeState(state);
  return hash;
}

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

  auto routing = computePrefixRouting(request->messages);
  TT_LOG_DEBUG(
      "[ChatCompletionsResolver] Routing: hasPriorTurn={}, lookupHash={}, "
      "registrationHash={}",
      routing.hasPriorTurn,
      routing.lookupHash.has_value() ? std::to_string(*routing.lookupHash)
                                     : "none",
      routing.registrationHash);

  // Layer 1: Prefix-cache routing. Requires a prior [assistant, user] pair.
  if (routing.hasPriorTurn && routing.lookupHash.has_value()) {
    try {
      auto acquired =
          sessionManager->tryAcquireByPrefixHash(*routing.lookupHash, cancelFn);

      if (acquired.has_value()) {
        tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(true);
        TT_LOG_DEBUG(
            "[ChatCompletionsResolver] Prefix cache HIT: hash={}, "
            "sessionId={}, slotId={}",
            *routing.lookupHash, acquired->sessionId, acquired->slotId);

        std::string deltaPrompt = renderLastUserTurn(request->messages);

        request->sessionId = acquired->sessionId;
        request->slotId = acquired->slotId;
        request->session = sessionManager->getSession(acquired->sessionId);
        request->continuation = true;
        request->registrationHash = routing.registrationHash;
        request->prompt_tokens_count =
            static_cast<int>(tt::utils::tokenizers::activeTokenizer()
                                 .encode(deltaPrompt)
                                 .size());
        request->prompt = std::move(deltaPrompt);
        sessionManager->registerPrefixHash(acquired->sessionId,
                                           routing.registrationHash);

        onDone();
        return;
      }

      tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(false);
      TT_LOG_DEBUG(
          "[ChatCompletionsResolver] Prefix cache MISS: hash={}, allocating "
          "new session",
          *routing.lookupHash);
    } catch (const services::SessionInFlightException& e) {
      TT_LOG_WARN("[ChatCompletionsResolver] All sessions busy for hash={}: {}",
                  *routing.lookupHash, e.what());
      onError({SessionErrorType::RATE_LIMIT, e.what()});
      return;
    }
  }

  // Layer 2: Allocate a new session. createSession is async -- callbacks
  // are queued on `loop` by the SessionManager. Fresh allocations leave
  // `request->registrationHash` at 0 (the disaggregation service treats
  // unhashed requests as fresh prefix).
  sessionManager->createSession(
      [request = std::move(request),
       registrationHash = routing.registrationHash,
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
      loop, routing.registrationHash);
}

}  // namespace tt::api::resolvers
