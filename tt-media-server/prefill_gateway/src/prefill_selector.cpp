// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_selector.hpp"

#include <algorithm>

namespace tt::gateway {

namespace {

struct Candidate {
  const PrefillSnapshot* prefill = nullptr;
  size_t prefixMatchDepth = 0;

  bool isBetterThan(const Candidate& other) const {
    if (prefixMatchDepth != other.prefixMatchDepth) {
      return prefixMatchDepth > other.prefixMatchDepth;
    }
    return prefill->inFlight < other.prefill->inFlight;
  }

  bool isTiedWith(const Candidate& other) const {
    return prefixMatchDepth == other.prefixMatchDepth &&
           prefill->inFlight == other.prefill->inFlight;
  }
};

}  // namespace

PrefillEligibilitySummary summarizePrefillEligibility(
    const std::vector<PrefillSnapshot>& prefills) {
  auto healthy = [](const PrefillSnapshot& prefill) { return prefill.healthy; };
  auto accepting = [](const PrefillSnapshot& prefill) {
    return prefill.healthy && prefill.acceptingTasks;
  };
  auto capacityAvailable = [](const PrefillSnapshot& prefill) {
    return prefill.healthy && prefill.acceptingTasks &&
           (prefill.maxInFlight == 0 || prefill.inFlight < prefill.maxInFlight);
  };

  PrefillEligibilitySummary summary;
  summary.total = prefills.size();
  summary.healthy = std::ranges::count_if(prefills, healthy);
  summary.accepting = std::ranges::count_if(prefills, accepting);
  summary.capacityAvailable =
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

    Candidate candidate{&p, p.prefixMatchDepth};
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

  const bool hasPrefixMatch = bestCandidates.front().prefixMatchDepth > 0;
  if (bestCandidates.size() == 1) {
    const auto& selected = bestCandidates.front();
    return {selected.prefill->serverId,
            hasPrefixMatch ? PrefillRoutingReason::PrefixMatch
                           : PrefillRoutingReason::LeastInflight,
            selected.prefixMatchDepth};
  }

  const size_t pickIndex = roundRobinCursor % bestCandidates.size();
  ++roundRobinCursor;
  const auto& selected = bestCandidates[pickIndex];
  return {selected.prefill->serverId,
          hasPrefixMatch ? PrefillRoutingReason::PrefixMatch
                         : PrefillRoutingReason::RoundRobin,
          selected.prefixMatchDepth};
}

}  // namespace tt::gateway
