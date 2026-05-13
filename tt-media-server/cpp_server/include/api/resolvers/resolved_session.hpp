// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace tt::domain {
class Session;
}  // namespace tt::domain

namespace tt::api::resolvers {

/**
 * Classification of a session-resolution failure. Maps 1:1 to an HTTP
 * status code at the controller boundary.
 */
enum class SessionErrorType {
  RATE_LIMIT,      // → HTTP 429 (no slot available right now)
  ALLOCATION_FAIL  // → HTTP 503 (memory/IPC layer refused or timed out)
};

struct SessionError {
  SessionErrorType type;
  std::string message;
};

/**
 * Routing decision produced by a session resolver.
 *
 * The controller copies the populated fields onto the in-flight
 * `LLMRequest` before dispatching generation. A default-constructed
 * `ResolvedSession` means "no session resolution was performed" (e.g.
 * the SessionManager is unavailable): controller leaves the request
 * untouched and the downstream pipeline runs without a session.
 */
struct ResolvedSession {
  // Session identity. nullopt when no session was attached to the request.
  std::optional<std::string> sessionId;
  std::optional<uint32_t> slotId;

  // Raw pointer back into the SessionManager's owning map. Borrowed: the
  // SessionManager owns the lifetime. Removed in a follow-up PR once the
  // SlotLease replaces in-flight bookkeeping.
  tt::domain::Session* session = nullptr;

  // True if the resolver allocated a brand-new slot. False on a
  // prefix-cache HIT against an existing session.
  bool isFresh = true;

  // Hash of the full current conversation (the resolver registers this
  // under the session so the next turn's lookup can find it). 0 means
  // "no prefix routing was performed" — currently only the no-manager
  // path.
  uint64_t registrationHash = 0;

  // Set only on a prefix-cache HIT: the rendered last-user turn that
  // should be sent to the model (the KV cache already holds the prefix).
  // Empty on fresh allocations — the controller keeps the full prompt
  // already on the request.
  std::string prompt;
  int promptTokensCount = 0;
};

}  // namespace tt::api::resolvers
