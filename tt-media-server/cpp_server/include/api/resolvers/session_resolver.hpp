// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <trantor/net/EventLoop.h>

#include <functional>
#include <memory>

#include "api/resolvers/session_error.hpp"
#include "domain/llm/llm_request.hpp"
#include "services/slot_lease.hpp"

namespace tt::api::resolvers {

/**
 * Strategy interface for binding an incoming LLM request to a session
 * slot. Implementations mutate the request with the resolved session id
 * / slot id / continuation flag / registration hash, then pass an
 * acquired SlotLease back via `onDone`. The lease is RAII -- the
 * caller's destruction releases the in-flight grant, so the request
 * pipeline never needs to manually clear in-flight state.
 *
 * Errors flow through `onError`; the SessionError type distinguishes
 * RATE_LIMIT (429) from ALLOCATION_FAIL (503) at the HTTP boundary.
 *
 * Each endpoint owns a concrete resolver tuned to its semantics
 * (prefix-cache routing for chat completions, fresh allocation for
 * responses, etc.) and the LLMController dispatches through the base
 * interface so both endpoints share the streaming / non-streaming
 * scaffolding.
 */
class SessionResolver {
 public:
  virtual ~SessionResolver() = default;

  virtual void resolve(std::shared_ptr<domain::llm::LLMRequest> request,
                       trantor::EventLoop* loop,
                       std::function<void(services::SlotLease)> onDone,
                       std::function<void(const SessionError&)> onError,
                       std::function<void()> cancelFn) const = 0;
};

}  // namespace tt::api::resolvers
