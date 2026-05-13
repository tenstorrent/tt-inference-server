// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/slot_lease.hpp"

#include "services/session_manager.hpp"

namespace tt::services {

void SlotLease::release() noexcept {
  if (manager == nullptr) return;
  // releaseInFlight is no-op when the session is missing, so this is
  // safe even if closeSession raced us.
  manager->releaseInFlight(sessionIdValue);
  manager = nullptr;
}

}  // namespace tt::services
