// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tt::gateway {

// Snapshot used for one selection call. Value type → pure-function selector.
struct PrefillSnapshot {
  std::string server_id;
  bool healthy = false;
  uint32_t in_flight = 0;
  uint32_t max_in_flight = 0;  // 0 = unlimited
};

// Choose a prefill. Order: sticky-by-hash → least-inflight → round-robin.
// `round_robin_cursor` is caller-owned so the selector stays pure.
std::optional<std::string> selectPrefill(
    const std::vector<PrefillSnapshot>& prefills, size_t registration_hash,
    const std::optional<std::string>& sticky_target,
    size_t& round_robin_cursor);

}  // namespace tt::gateway
