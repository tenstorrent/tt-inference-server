// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tt::gateway {

/**
 * @brief Reason a prefill server was selected (or no selection was made).
 *
 * Stable string-mappable values for metrics and structured logs:
 * gateway_routing_decisions_total{reason="..."}.
 */
enum class SelectionReason : uint8_t {
  PREFIX_MATCH,           // v1: longest cached-prefix hit
  EQUALITY_MATCH,         // v0: routing-table hit on single registration_hash
  LEAST_INFLIGHT,         // no prefix match, picked least-loaded healthy peer
  ROUND_ROBIN,            // cold-start fallback; all equal
  NO_PEERS_AVAILABLE,     // every peer down or accepting_tasks=false
};

/**
 * @brief Snapshot of a single prefill used for one selection call.
 *
 * Built by the Dispatcher from PrefillRegistry state. Keeping it a value
 * type lets the selector run as a pure function in unit tests with no fakes.
 */
struct PrefillSnapshot {
  std::string server_id;
  bool healthy = false;          // socket alive, registration completed
  bool accepting_tasks = true;   // from LoadBalanceMessage
  uint32_t in_flight = 0;        // gateway-tracked dispatch count
  uint32_t max_in_flight = 0;    // from PrefillRegistration; 0 = unlimited
};

/**
 * @brief Outcome of a single selection call.
 */
struct SelectionResult {
  std::optional<std::string> server_id;
  SelectionReason reason = SelectionReason::NO_PEERS_AVAILABLE;
};

/**
 * @brief Convert SelectionReason to its metric / structured-log label.
 */
const char* reasonLabel(SelectionReason reason);

/**
 * @brief Pure selector: choose a prefill for a request.
 *
 * v0 logic:
 *   1. If `registration_hash != 0` and `sticky_target` is set, and that
 *      peer is healthy & not overloaded → EQUALITY_MATCH.
 *   2. Else pick the healthy + accepting peer with the lowest in_flight →
 *      LEAST_INFLIGHT (or ROUND_ROBIN among ties using `round_robin_cursor`).
 *   3. Else no peers available → NO_PEERS_AVAILABLE.
 *
 * Round-robin cursor is passed by reference so the caller (Dispatcher) owns
 * the state and the selector stays pure.
 *
 * Future v1 adds a longest-prefix-match step ahead of equality match once
 * chained block hashes (issue #3467) land.
 *
 * @param prefills           All known prefills, including down / overloaded ones.
 * @param registration_hash  Hash from PrefillRequestMessage (0 = no hint).
 * @param sticky_target      AffinityCache lookup result, if any.
 * @param round_robin_cursor In/out: caller-owned cursor for tie-breaking.
 */
SelectionResult selectPrefill(const std::vector<PrefillSnapshot>& prefills,
                              size_t registration_hash,
                              const std::optional<std::string>& sticky_target,
                              size_t& round_robin_cursor);

}  // namespace tt::gateway
