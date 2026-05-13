// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <trantor/net/EventLoop.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "api/resolvers/session_error.hpp"
#include "api/resolvers/session_resolver.hpp"
#include "domain/llm/chat_message.hpp"
#include "domain/llm/llm_request.hpp"

namespace tt::services {
class SessionManager;
}  // namespace tt::services

namespace tt::api::resolvers {

/**
 * Routes /v1/chat/completions requests to an existing prefix-cache slot
 * when possible, falling back to a fresh allocation otherwise.
 *
 * The resolver inspects the conversation messages, derives a lookup hash
 * for the prior turn, and asks the SessionManager to atomically acquire
 * the matching slot in-flight. On a miss it allocates a new session
 * asynchronously via the SessionManager's IPC path.
 *
 * On success the resolver mutates the request in place with the
 * resolved session, slot, continuation flag, `registrationHash`, and
 * (on a HIT) delta prompt + token count. The `onDone` callback then
 * signals "request is ready, dispatch generation".
 *
 * Dispatch:
 *   - Prefix-cache HIT path is synchronous: `onDone` fires on the
 *     calling thread before `resolve()` returns.
 *   - Allocation path is asynchronous: `onDone` / `onError` are queued
 *     on `loop` by the SessionManager.
 *   - RATE_LIMIT (all candidate sessions in-flight) is reported
 *     synchronously via `onError`.
 *   - When no SessionManager is wired up, the request is left untouched
 *     and `onDone` fires synchronously.
 *
 * `cancelFn` is bound atomically with the in-flight state so a
 * concurrent SessionManager::closeSession aborts the request.
 */
class ChatCompletionsResolver : public SessionResolver {
 public:
  explicit ChatCompletionsResolver(
      std::shared_ptr<services::SessionManager> manager);

  void resolve(std::shared_ptr<domain::llm::LLMRequest> request,
               trantor::EventLoop* loop,
               std::function<void(services::SlotLease)> onDone,
               std::function<void(const SessionError&)> onError,
               std::function<void()> cancelFn) const override;

  /**
   * Stable 64-bit identity hash of a chat-message list. Computed
   * structurally over (role, content) pairs -- no tokenizer involvement
   * -- so the value is reproducible by any caller that needs to talk to
   * the SessionManager's prefix-hash registry (currently: this resolver
   * and its tests). Returns 0 for an empty list.
   */
  static uint64_t hashMessages(
      const std::vector<domain::llm::ChatMessage>& messages);

 private:
  std::shared_ptr<services::SessionManager> sessionManager;
};

}  // namespace tt::api::resolvers
