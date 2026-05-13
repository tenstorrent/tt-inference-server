// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_selector.hpp"

#include <algorithm>
#include <limits>

namespace tt::gateway {

namespace {

bool isEligible(const PrefillSnapshot& p) {
  if (!p.healthy) return false;
  if (!p.accepting_tasks) return false;
  if (p.max_in_flight > 0 && p.in_flight >= p.max_in_flight) return false;
  return true;
}

const PrefillSnapshot* findById(const std::vector<PrefillSnapshot>& prefills,
                                const std::string& server_id) {
  auto it = std::find_if(prefills.begin(), prefills.end(),
                         [&](const PrefillSnapshot& p) {
                           return p.server_id == server_id;
                         });
  return it == prefills.end() ? nullptr : &*it;
}

}  // namespace

const char* reasonLabel(SelectionReason reason) {
  switch (reason) {
    case SelectionReason::PREFIX_MATCH:
      return "prefix_match";
    case SelectionReason::EQUALITY_MATCH:
      return "equality_match";
    case SelectionReason::LEAST_INFLIGHT:
      return "least_inflight";
    case SelectionReason::ROUND_ROBIN:
      return "round_robin";
    case SelectionReason::NO_PEERS_AVAILABLE:
      return "no_peers_available";
  }
  return "unknown";
}

SelectionResult selectPrefill(const std::vector<PrefillSnapshot>& prefills,
                              size_t registration_hash,
                              const std::optional<std::string>& sticky_target,
                              size_t& round_robin_cursor) {
  if (registration_hash != 0 && sticky_target.has_value()) {
    const PrefillSnapshot* hit = findById(prefills, *sticky_target);
    if (hit && isEligible(*hit)) {
      return {*sticky_target, SelectionReason::EQUALITY_MATCH};
    }
  }

  std::vector<const PrefillSnapshot*> eligible;
  eligible.reserve(prefills.size());
  for (const auto& p : prefills) {
    if (isEligible(p)) eligible.push_back(&p);
  }

  if (eligible.empty()) {
    return {std::nullopt, SelectionReason::NO_PEERS_AVAILABLE};
  }

  uint32_t min_in_flight = std::numeric_limits<uint32_t>::max();
  for (const auto* p : eligible) {
    min_in_flight = std::min(min_in_flight, p->in_flight);
  }

  std::vector<const PrefillSnapshot*> least_loaded;
  least_loaded.reserve(eligible.size());
  for (const auto* p : eligible) {
    if (p->in_flight == min_in_flight) least_loaded.push_back(p);
  }

  if (least_loaded.size() == 1) {
    return {least_loaded.front()->server_id, SelectionReason::LEAST_INFLIGHT};
  }

  const size_t pick_index = round_robin_cursor % least_loaded.size();
  ++round_robin_cursor;
  return {least_loaded[pick_index]->server_id, SelectionReason::ROUND_ROBIN};
}

}  // namespace tt::gateway
