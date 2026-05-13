// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "api/resolvers/session_resolver.hpp"

namespace tt::services {
class SessionManager;
}  // namespace tt::services

namespace tt::api::resolvers {

/**
 * Session resolver for /v1/responses.
 *
 * NOTE: The /v1/responses endpoint does not implement prefix caching
 * yet -- the server doesn't store enough per-response state to attempt
 * a slot lookup, so every request goes through fresh allocation. When
 * `previous_response_id` support lands, this resolver becomes the
 * place where a prior response is mapped to a warmed slot. Until then
 * the implementation just calls `SessionManager::createSession`.
 */
class ResponsesResolver : public SessionResolver {
 public:
  explicit ResponsesResolver(
      std::shared_ptr<services::SessionManager> manager);

  void resolve(std::shared_ptr<domain::llm::LLMRequest> request,
               trantor::EventLoop* loop, std::function<void()> onDone,
               std::function<void(const SessionError&)> onError,
               std::function<void()> cancelFn) const override;

 private:
  std::shared_ptr<services::SessionManager> sessionManager;
};

}  // namespace tt::api::resolvers
