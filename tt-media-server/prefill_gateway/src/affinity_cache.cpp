// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/affinity_cache.hpp"

namespace tt::gateway {

std::optional<std::string> AffinityCache::lookup(size_t hash) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = hash_to_server_.find(hash);
  if (it == hash_to_server_.end()) return std::nullopt;
  return it->second;
}

void AffinityCache::record(size_t hash, const std::string& serverId) {
  if (hash == 0) return;
  std::lock_guard<std::mutex> lock(mutex_);
  hash_to_server_[hash] = serverId;
}

void AffinityCache::evictPrefill(const std::string& serverId) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::erase_if(hash_to_server_,
                [&](const auto& kv) { return kv.second == serverId; });
}

size_t AffinityCache::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return hash_to_server_.size();
}

}  // namespace tt::gateway
