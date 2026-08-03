// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>

#include "domain/sentinel_values.hpp"

namespace tt::domain {

enum class MarkInFlightOutcome {
  Marked,
  Busy,
  Stale,
  NotFound,
};

struct MarkInFlightResult {
  MarkInFlightOutcome outcome = MarkInFlightOutcome::NotFound;
  uint32_t slotId = INVALID_SLOT_ID;
};

}  // namespace tt::domain
