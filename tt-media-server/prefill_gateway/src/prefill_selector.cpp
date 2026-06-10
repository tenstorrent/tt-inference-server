// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_selector.hpp"

#include <algorithm>

#include "utils/prefix_match.hpp"

namespace tt::gateway {

namespace {

bool isEligible(const PrefillSnapshot& p) {
  if (!p.healthy) return false;
  if (!p.accepting_tasks) return false;
  if (p.max_in_flight > 0 && p.in_flight >= p.max_in_flight) return false;
  return true;
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
    case PrefillRoutingReason::LeastInflight:
      return "least_inflight";
    case PrefillRoutingReason::RoundRobin:
      return "round_robin";
    case PrefillRoutingReason::NoEligiblePrefill:
      return "no_eligible_prefill";
  }
  return "unknown";
}

PrefillSelection selectPrefill(
    const std::vector<PrefillSnapshot>& prefills,
    const std::vector<uint64_t>& registrationHashes, size_t& roundRobinCursor) {
  std::vector<const PrefillSnapshot*> eligible;
  eligible.reserve(prefills.size());
  for (const auto& p : prefills) {
    if (isEligible(p)) eligible.push_back(&p);
  }

  if (eligible.empty()) {
    return {std::nullopt, PrefillRoutingReason::NoEligiblePrefill};
  }

  size_t bestPrefixDepth = 0;
  std::vector<const PrefillSnapshot*> prefixMatches;
  if (!registrationHashes.empty()) {
    for (const auto* p : eligible) {
      const size_t depth = tt::utils::countMatchingPrefixDepth(
          registrationHashes, [p](uint64_t hash) {
            return p->cached_block_hashes.contains(hash);
          });
      if (depth == 0) {
        continue;
      }
      if (depth > bestPrefixDepth) {
        bestPrefixDepth = depth;
        prefixMatches.clear();
      }
      if (depth == bestPrefixDepth) {
        prefixMatches.push_back(p);
      }
    }
  }

  const std::vector<const PrefillSnapshot*>& candidates =
      bestPrefixDepth > 0 ? prefixMatches : eligible;

  const auto minInFlight =
      (*std::ranges::min_element(candidates, {}, [](const PrefillSnapshot* p) {
        return p->in_flight;
      }))->in_flight;

  std::vector<const PrefillSnapshot*> leastLoaded;
  leastLoaded.reserve(candidates.size());
  for (const auto* p : candidates) {
    if (p->in_flight == minInFlight) leastLoaded.push_back(p);
  }

  if (leastLoaded.size() == 1) {
    return {leastLoaded.front()->server_id,
            bestPrefixDepth > 0 ? PrefillRoutingReason::PrefixMatch
                                : PrefillRoutingReason::LeastInflight,
            bestPrefixDepth};
  }

  const size_t pickIndex = roundRobinCursor % leastLoaded.size();
  ++roundRobinCursor;
  return {leastLoaded[pickIndex]->server_id,
          bestPrefixDepth > 0 ? PrefillRoutingReason::PrefixMatch
                              : PrefillRoutingReason::RoundRobin,
          bestPrefixDepth};
}

}  // namespace tt::gateway
