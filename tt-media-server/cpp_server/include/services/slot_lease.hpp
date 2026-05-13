// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "domain/slot_types.hpp"

namespace tt::services {

class SessionManager;

/**
 * RAII handle for the in-flight grant on a session slot. Constructed by
 * the session resolver after a successful acquire (prefix-cache HIT) or
 * allocation (MISS); the request pipeline owns it until the response is
 * fully delivered, error-ed, or aborted, at which point destruction
 * releases the slot back to the manager.
 *
 * Lifecycle:
 *   - Empty by default; equivalent to "no slot bound to this request".
 *   - acquired -> moved through resolver onDone callback into the
 *     response writer.
 *   - release() is idempotent and noexcept; the destructor calls it.
 *
 * Threading: SessionManager::releaseInFlight is thread-safe (it just
 * mutates the session under the manager's existing lock), so the lease
 * can be destroyed on any thread.
 *
 * The lease deliberately mirrors no state from the session itself
 * beyond identity -- it stores the slot id only so callers that already
 * have a lease don't need a second `getSlotIdBySessionId` round-trip.
 */
class SlotLease {
 public:
  SlotLease() = default;
  SlotLease(SessionManager* manager, std::string sessionId, uint32_t slotId)
      : manager(manager),
        sessionIdValue(std::move(sessionId)),
        slotIdValue(slotId) {}

  ~SlotLease() { release(); }

  SlotLease(const SlotLease&) = delete;
  SlotLease& operator=(const SlotLease&) = delete;

  SlotLease(SlotLease&& other) noexcept
      : manager(std::exchange(other.manager, nullptr)),
        sessionIdValue(std::move(other.sessionIdValue)),
        slotIdValue(std::exchange(other.slotIdValue, domain::INVALID_SLOT_ID)) {
  }

  SlotLease& operator=(SlotLease&& other) noexcept {
    if (this != &other) {
      release();
      manager = std::exchange(other.manager, nullptr);
      sessionIdValue = std::move(other.sessionIdValue);
      slotIdValue = std::exchange(other.slotIdValue, domain::INVALID_SLOT_ID);
    }
    return *this;
  }

  // Release the in-flight grant. Idempotent: subsequent calls are no-ops.
  // Safe when the session has been closed concurrently (closeSession
  // already pulled the in-flight state via takeCancelFn).
  void release() noexcept;

  bool empty() const noexcept { return manager == nullptr; }
  const std::string& sessionId() const noexcept { return sessionIdValue; }
  uint32_t slotId() const noexcept { return slotIdValue; }

 private:
  SessionManager* manager = nullptr;
  std::string sessionIdValue;
  uint32_t slotIdValue = domain::INVALID_SLOT_ID;
};

}  // namespace tt::services
