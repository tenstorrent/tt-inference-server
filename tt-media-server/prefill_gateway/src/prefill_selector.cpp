// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_selector.hpp"

#include <algorithm>

namespace tt::gateway {

namespace {

struct Candidate {
  const PrefillSnapshot* prefill = nullptr;
  size_t prefix_match_depth = 0;

  bool isBetterThan(const Candidate& other) const {
    if (prefix_match_depth != other.prefix_match_depth) {
      return prefix_match_depth > other.prefix_match_depth;
    }
    return prefill->in_flight < other.prefill->in_flight;
  }

  bool isTiedWith(const Candidate& other) const {
    return prefix_match_depth == other.prefix_match_depth &&
           prefill->in_flight == other.prefill->in_flight;
  }
};

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

PrefillSelection selectPrefill(const std::vector<PrefillSnapshot>& prefills,
                               size_t& roundRobinCursor) {
  std::vector<Candidate> bestCandidates;
  bestCandidates.reserve(prefills.size());

  for (const auto& p : prefills) {
    if (!p.isEligible()) {
      continue;
    }

    Candidate candidate{&p, p.prefix_match_depth};
    if (bestCandidates.empty() ||
        candidate.isBetterThan(bestCandidates.front())) {
      bestCandidates = {candidate};
      continue;
    }
    if (candidate.isTiedWith(bestCandidates.front())) {
      bestCandidates.push_back(candidate);
    }
  }

  if (bestCandidates.empty()) {
    return {std::nullopt, PrefillRoutingReason::NoEligiblePrefill};
  }

  const bool hasPrefixMatch = bestCandidates.front().prefix_match_depth > 0;
  if (bestCandidates.size() == 1) {
    const auto& selected = bestCandidates.front();
    return {selected.prefill->server_id,
            hasPrefixMatch ? PrefillRoutingReason::PrefixMatch
                           : PrefillRoutingReason::LeastInflight,
            selected.prefix_match_depth};
  }

  const size_t pickIndex = roundRobinCursor % bestCandidates.size();
  ++roundRobinCursor;
  const auto& selected = bestCandidates[pickIndex];
  return {selected.prefill->server_id,
          hasPrefixMatch ? PrefillRoutingReason::PrefixMatch
                         : PrefillRoutingReason::RoundRobin,
          selected.prefix_match_depth};
}

}  // namespace tt::gateway
