// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tt::gateway {

// Snapshot used for one selection call. Value type → pure-function selector.
struct PrefillSnapshot {
  std::string server_id;
  bool healthy = false;
  bool accepting_tasks = true;
  uint32_t in_flight = 0;
  uint32_t max_in_flight = 0;  // 0 = unlimited
  size_t cached_blocks = 0;
  std::chrono::steady_clock::time_point last_heartbeat{};
};

enum class PrefillRoutingReason {
  PrefixMatch,
  StickyFallback,
  LeastInflight,
  RoundRobin,
  NoEligiblePrefill,
};

struct PrefillSelection {
  std::optional<std::string> server_id;
  PrefillRoutingReason reason = PrefillRoutingReason::NoEligiblePrefill;
};

struct PrefillEligibilitySummary {
  size_t total = 0;
  size_t healthy = 0;
  size_t accepting = 0;
  size_t capacity_available = 0;
};

PrefillEligibilitySummary summarizePrefillEligibility(
    const std::vector<PrefillSnapshot>& prefills);

std::string_view routingReasonName(PrefillRoutingReason reason);

PrefillSelection selectPrefill(const std::vector<PrefillSnapshot>& prefills,
                               size_t registration_hash,
                               const std::optional<std::string>& sticky_target,
                               size_t& round_robin_cursor);

}  // namespace tt::gateway
