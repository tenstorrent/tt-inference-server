// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/resolvers/responses_resolver.hpp"

#include <string>
#include <utility>

#include "domain/session.hpp"
#include "services/session_manager.hpp"
#include "utils/logger.hpp"

namespace tt::api::resolvers {

ResponsesResolver::ResponsesResolver(
    std::shared_ptr<services::SessionManager> manager)
    : sessionManager(std::move(manager)) {}

void ResponsesResolver::resolve(
    std::shared_ptr<domain::llm::LLMRequest> request,
    trantor::EventLoop* loop, std::function<void()> onDone,
    std::function<void(const SessionError&)> onError,
    std::function<void()> cancelFn) const {
  if (!sessionManager) {
    TT_LOG_WARN("[ResponsesResolver] SessionManager not available");
    onDone();
    return;
  }

  // /v1/responses has no prefix-cache routing today: the server doesn't
  // persist enough per-response state to map a `previous_response_id`
  // back to a warmed slot, so a lookup would always miss. Skip Layer 1
  // entirely and always allocate. When previous_response_id support
  // arrives, the lookup branch lands here -- mirroring the structure of
  // ChatCompletionsResolver.
  sessionManager->createSession(
      [request = std::move(request), cancelFn = std::move(cancelFn),
       mgr = sessionManager,
       onDone = std::move(onDone)](const tt::domain::Session& session) mutable {
        request->sessionId = session.getSessionId();
        request->slotId =
            mgr->acquireInFlight(session.getSessionId(), std::move(cancelFn));
        request->session = mgr->getSession(session.getSessionId());
        request->continuation = false;
        // registrationHash stays at 0: there is no prefix-cache key to
        // register this session under, and the disaggregation service
        // treats 0 as "fresh prefix, no slot reuse".

        TT_LOG_INFO(
            "[ResponsesResolver] New session: sessionId={}, slotId={}",
            session.getSessionId(),
            request->slotId.has_value() ? std::to_string(*request->slotId)
                                        : "none");

        onDone();
      },
      [onError = std::move(onError)](std::string_view err) {
        onError({SessionErrorType::ALLOCATION_FAIL, std::string(err)});
      },
      loop, /*registrationHash=*/0);
}

}  // namespace tt::api::resolvers
