// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace tt::gateway {

// Maps registration_hash -> prefill that last handled it. Cache semantics:
// best-effort, may be missing at any time; selector treats misses as no hint.
class AffinityCache {
 public:
  AffinityCache() = default;
  AffinityCache(const AffinityCache&) = delete;
  AffinityCache& operator=(const AffinityCache&) = delete;

  std::optional<std::string> lookup(size_t hash) const;

  // Most-recent-wins. No-op when hash == 0.
  void record(size_t hash, const std::string& server_id);

  // Remove all entries pointing at `server_id` (called on prefill drop).
  void evictPrefill(const std::string& server_id);

 private:
  mutable std::mutex mutex_;
  std::unordered_map<size_t, std::string> hash_to_server_;
};

}  // namespace tt::gateway
