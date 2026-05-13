// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <trantor/net/EventLoop.h>

#include <functional>
#include <memory>
#include <vector>

#include "api/resolvers/resolved_session.hpp"
#include "domain/llm/chat_message.hpp"

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
 * The resolver does NOT mutate the LLMRequest; the controller copies
 * the fields from ResolvedSession onto the request after `onDone` fires.
 *
 * Dispatch:
 *   - Prefix-cache HIT path is synchronous: `onDone` fires on the calling
 *     thread before `resolve()` returns.
 *   - Allocation path is asynchronous: `onDone` / `onError` are queued on
 *     `loop` by the SessionManager.
 *   - RATE_LIMIT (all candidate sessions in-flight) is reported
 *     synchronously via `onError`.
 *   - When no SessionManager is wired up, `onDone` fires synchronously
 *     with a default-constructed ResolvedSession.
 *
 * `cancelFn` is bound atomically with the in-flight state so that a
 * concurrent SessionManager::closeSession aborts the request.
 */
class ChatCompletionsResolver {
 public:
  explicit ChatCompletionsResolver(
      std::shared_ptr<services::SessionManager> manager);

  void resolve(const std::vector<domain::llm::ChatMessage>& messages,
               trantor::EventLoop* loop,
               std::function<void(ResolvedSession)> onDone,
               std::function<void(const SessionError&)> onError,
               std::function<void()> cancelFn) const;

 private:
  std::shared_ptr<services::SessionManager> sessionManager;
};

}  // namespace tt::api::resolvers
