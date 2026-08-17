// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <limits>

namespace tt::domain {

// Sentinel for an unassigned KV-cache slot. Matches tt_llm_engine INVALID_SLOT
// and pipeline wire-format conventions (UINT32_MAX).
constexpr uint32_t INVALID_SLOT_ID = std::numeric_limits<uint32_t>::max();

}  // namespace tt::domain
