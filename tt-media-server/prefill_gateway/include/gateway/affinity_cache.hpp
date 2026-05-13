// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace tt::gateway {

/**
 * @brief Remembers which prefill last handled a given registration_hash.
 *
 * "Cache" semantics on purpose: entries are written on every result,
 * invalidated when the assigned prefill drops, and may be missing at any
 * moment (cold start, eviction, gateway restart). Selector treats a miss
 * as "no affinity hint" and falls back to least-in-flight / round-robin.
 *
 * v0: equality match on a single `size_t` hash carried in
 * PrefillRequestMessage. v1 replaces this with the per-prefill block-cache
 * view held by PrefillRegistry once chained block hashes (issue #3467) land.
 *
 * Concurrency: protected by a single mutex; gateway throughput is bounded
 * by the prefill peers (seconds per request), so contention here is not a
 * concern in v0.
 */
class AffinityCache {
 public:
  AffinityCache() = default;
  AffinityCache(const AffinityCache&) = delete;
  AffinityCache& operator=(const AffinityCache&) = delete;

  /**
   * @brief Look up the server_id last seen handling `hash`.
   * @return std::nullopt if the hash is unknown or the entry was evicted.
   */
  std::optional<std::string> lookup(size_t hash) const;

  /**
   * @brief Record that `server_id` handled `hash` (most recent wins).
   *
   * No-op when hash is 0 (sentinel for "no hint").
   */
  void record(size_t hash, const std::string& server_id);

  /**
   * @brief Remove every entry that points at `server_id`.
   *
   * Called when a prefill drops; the gateway must invalidate stale affinity
   * before serving the next request.
   */
  void evictPrefill(const std::string& server_id);

  /**
   * @brief Drop a single entry (e.g., when a task fails after dispatch).
   */
  void evictHash(size_t hash);

  size_t size() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<size_t, std::string> hash_to_server_;
};

}  // namespace tt::gateway
