// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_selector.hpp"

#include <algorithm>
#include <limits>

namespace tt::gateway {

namespace {

bool isEligible(const PrefillSnapshot& p) {
  if (!p.healthy) return false;
  if (p.max_in_flight > 0 && p.in_flight >= p.max_in_flight) return false;
  return true;
}

const PrefillSnapshot* findById(const std::vector<PrefillSnapshot>& prefills,
                                const std::string& serverId) {
  auto it = std::find_if(
      prefills.begin(), prefills.end(),
      [&](const PrefillSnapshot& p) { return p.server_id == serverId; });
  return it == prefills.end() ? nullptr : &*it;
}

}  // namespace

std::optional<std::string> selectPrefill(
    const std::vector<PrefillSnapshot>& prefills, size_t registrationHash,
    const std::optional<std::string>& stickyTarget, size_t& roundRobinCursor) {
  if (registrationHash != 0 && stickyTarget.has_value()) {
    const PrefillSnapshot* hit = findById(prefills, *stickyTarget);
    if (hit && isEligible(*hit)) {
      return *stickyTarget;
    }
  }

  std::vector<const PrefillSnapshot*> eligible;
  eligible.reserve(prefills.size());
  for (const auto& p : prefills) {
    if (isEligible(p)) eligible.push_back(&p);
  }

  if (eligible.empty()) {
    return std::nullopt;
  }

  uint32_t minInFlight = std::numeric_limits<uint32_t>::max();
  for (const auto* p : eligible) {
    minInFlight = std::min(minInFlight, p->in_flight);
  }

  std::vector<const PrefillSnapshot*> leastLoaded;
  leastLoaded.reserve(eligible.size());
  for (const auto* p : eligible) {
    if (p->in_flight == minInFlight) leastLoaded.push_back(p);
  }

  if (leastLoaded.size() == 1) {
    return leastLoaded.front()->server_id;
  }

  const size_t pickIndex = roundRobinCursor % leastLoaded.size();
  ++roundRobinCursor;
  return leastLoaded[pickIndex]->server_id;
}

}  // namespace tt::gateway
