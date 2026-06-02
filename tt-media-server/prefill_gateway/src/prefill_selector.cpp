// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_selector.hpp"

#include <algorithm>

namespace tt::gateway {

namespace {

bool isEligible(const PrefillSnapshot& p) {
  if (!p.healthy) return false;
  if (!p.accepting_tasks) return false;
  if (p.max_in_flight > 0 && p.in_flight >= p.max_in_flight) return false;
  return true;
}

const PrefillSnapshot* findById(const std::vector<PrefillSnapshot>& prefills,
                                const std::string& serverId) {
  auto it = std::ranges::find_if(prefills, [&](const PrefillSnapshot& p) {
    return p.server_id == serverId;
  });
  return it == prefills.end() ? nullptr : &*it;
}

}  // namespace

PrefillEligibilitySummary summarizePrefillEligibility(
    const std::vector<PrefillSnapshot>& prefills) {
  auto healthy = [](const PrefillSnapshot& prefill) { return prefill.healthy; };
  auto accepting = [](const PrefillSnapshot& prefill) {
    return prefill.healthy && prefill.accepting_tasks;
  };
  auto capacityAvailable = [](const PrefillSnapshot& prefill) {
    return prefill.healthy && prefill.accepting_tasks &&
           (prefill.max_in_flight == 0 ||
            prefill.in_flight < prefill.max_in_flight);
  };

  PrefillEligibilitySummary summary;
  summary.total = prefills.size();
  summary.healthy = std::ranges::count_if(prefills, healthy);
  summary.accepting = std::ranges::count_if(prefills, accepting);
  summary.capacity_available =
      std::ranges::count_if(prefills, capacityAvailable);
  return summary;
}

std::string_view routingReasonName(PrefillRoutingReason reason) {
  switch (reason) {
    case PrefillRoutingReason::PrefixMatch:
      return "prefix_match";
    case PrefillRoutingReason::StickyFallback:
      return "sticky_fallback";
    case PrefillRoutingReason::LeastInflight:
      return "least_inflight";
    case PrefillRoutingReason::RoundRobin:
      return "round_robin";
    case PrefillRoutingReason::NoEligiblePrefill:
      return "no_eligible_prefill";
  }
  return "unknown";
}

PrefillSelection selectPrefill(const std::vector<PrefillSnapshot>& prefills,
                               size_t registrationHash,
                               const std::optional<std::string>& stickyTarget,
                               size_t& roundRobinCursor) {
  const bool hasStickyHint = registrationHash != 0 && stickyTarget.has_value();
  if (hasStickyHint) {
    const PrefillSnapshot* hit = findById(prefills, *stickyTarget);
    if (hit && isEligible(*hit)) {
      return {*stickyTarget, PrefillRoutingReason::PrefixMatch};
    }
  }

  std::vector<const PrefillSnapshot*> eligible;
  eligible.reserve(prefills.size());
  for (const auto& p : prefills) {
    if (isEligible(p)) eligible.push_back(&p);
  }

  if (eligible.empty()) {
    return {std::nullopt, PrefillRoutingReason::NoEligiblePrefill};
  }

  const auto minInFlight =
      (*std::ranges::min_element(eligible, {}, [](const PrefillSnapshot* p) {
        return p->in_flight;
      }))->in_flight;

  std::vector<const PrefillSnapshot*> leastLoaded;
  leastLoaded.reserve(eligible.size());
  for (const auto* p : eligible) {
    if (p->in_flight == minInFlight) leastLoaded.push_back(p);
  }

  if (leastLoaded.size() == 1) {
    return {leastLoaded.front()->server_id,
            hasStickyHint ? PrefillRoutingReason::StickyFallback
                          : PrefillRoutingReason::LeastInflight};
  }

  const size_t pickIndex = roundRobinCursor % leastLoaded.size();
  ++roundRobinCursor;
  return {leastLoaded[pickIndex]->server_id,
          hasStickyHint ? PrefillRoutingReason::StickyFallback
                        : PrefillRoutingReason::RoundRobin};
}

}  // namespace tt::gateway
