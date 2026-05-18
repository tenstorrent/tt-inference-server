// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tt::gateway {

// Reason a prefill was (not) selected; used for metrics labels.
enum class SelectionReason : uint8_t {
  EQUALITY_MATCH,  // AffinityCache hit on registration_hash
  LEAST_INFLIGHT,
  ROUND_ROBIN,  // cold-start tie-breaker
  NO_PEERS_AVAILABLE,
};

// Snapshot used for one selection call. Value type → pure-function selector.
struct PrefillSnapshot {
  std::string server_id;
  bool healthy = false;
  bool accepting_tasks = true;
  uint32_t in_flight = 0;
  uint32_t max_in_flight = 0;  // 0 = unlimited
};

struct SelectionResult {
  std::optional<std::string> server_id;
  SelectionReason reason = SelectionReason::NO_PEERS_AVAILABLE;
};

const char* reasonLabel(SelectionReason reason);

// Choose a prefill. Order: sticky-by-hash → least-inflight → round-robin.
// `round_robin_cursor` is caller-owned so the selector stays pure.
SelectionResult selectPrefill(const std::vector<PrefillSnapshot>& prefills,
                              size_t registration_hash,
                              const std::optional<std::string>& sticky_target,
                              size_t& round_robin_cursor);

}  // namespace tt::gateway
