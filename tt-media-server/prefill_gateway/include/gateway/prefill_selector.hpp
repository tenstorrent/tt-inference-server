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
  std::string serverId;
  bool healthy = false;
  bool acceptingTasks = true;
  uint32_t inFlight = 0;
  uint32_t maxInFlight = 0;  // 0 = unlimited
  size_t cachedBlocks = 0;
  size_t prefixMatchDepth = 0;
  std::chrono::steady_clock::time_point lastHeartbeat{};

  bool isEligible() const {
    if (!healthy) return false;
    if (!acceptingTasks) return false;
    if (maxInFlight > 0 && inFlight >= maxInFlight) return false;
    return true;
  }
};

enum class PrefillRoutingReason {
  PrefixMatch,
  LeastInflight,
  RoundRobin,
  NoEligiblePrefill,
};

struct PrefillSelection {
  std::optional<std::string> serverId;
  PrefillRoutingReason reason = PrefillRoutingReason::NoEligiblePrefill;
  size_t prefixMatchDepth = 0;
};

struct PrefillEligibilitySummary {
  size_t total = 0;
  size_t healthy = 0;
  size_t accepting = 0;
  size_t capacityAvailable = 0;
};

PrefillEligibilitySummary summarizePrefillEligibility(
    const std::vector<PrefillSnapshot>& prefills);

std::string_view routingReasonName(PrefillRoutingReason reason);

PrefillSelection selectPrefill(const std::vector<PrefillSnapshot>& prefills,
                               size_t& roundRobinCursor);

}  // namespace tt::gateway
