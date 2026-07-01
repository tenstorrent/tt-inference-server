// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "domain/sentinel_values.hpp"
#include "domain/session.hpp"

namespace tt::services {

class SessionRateLimitException : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class SessionInFlightException : public SessionRateLimitException {
 public:
  SessionInFlightException()
      : SessionRateLimitException(
            "Session already has a request in flight. Multiple concurrent "
            "requests per session are not supported.") {}
};

enum class MarkInFlightOutcome {
  Marked,
  Busy,
  Stale,
  NotFound,
};

struct MarkInFlightResult {
  MarkInFlightOutcome outcome = MarkInFlightOutcome::NotFound;
  uint32_t slotId = domain::INVALID_SLOT_ID;
};

class SessionLease {
 public:
  virtual ~SessionLease() = default;

  // Atomically validate (when expected* are set), mark the session in-flight,
  // and register cancelFn. cancelFn is moved only on Marked; unchanged
  // otherwise.
  virtual MarkInFlightResult tryMarkInFlight(
      const std::string& sessionId, std::function<void()>& cancelFn,
      std::optional<uint64_t> expectedKeyHash = std::nullopt,
      const std::string* expectedResponseId = nullptr) = 0;

  virtual std::shared_ptr<domain::Session> getSession(
      const std::string& sessionId) = 0;

  virtual std::optional<uint64_t> getSessionHash(
      const std::string& sessionId) const = 0;

  virtual bool setSessionHash(const std::string& sessionId,
                              uint64_t keyHash) = 0;

  virtual bool setSessionResponseId(const std::string& sessionId,
                                    const std::string& responseId) = 0;

  virtual void shrinkResidentPrefixToMatchedTokens(const std::string& sessionId,
                                                   uint32_t matchedTokens) = 0;

  virtual void unlockSlot(uint32_t slotId) = 0;
};

}  // namespace tt::services
