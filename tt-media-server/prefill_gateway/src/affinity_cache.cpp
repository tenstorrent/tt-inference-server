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

void AffinityCache::record(size_t hash, const std::string& server_id) {
  if (hash == 0) return;
  std::lock_guard<std::mutex> lock(mutex_);
  hash_to_server_[hash] = server_id;
}

void AffinityCache::evictPrefill(const std::string& server_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = hash_to_server_.begin(); it != hash_to_server_.end();) {
    if (it->second == server_id) {
      it = hash_to_server_.erase(it);
    } else {
      ++it;
    }
  }
}

void AffinityCache::evictHash(size_t hash) {
  std::lock_guard<std::mutex> lock(mutex_);
  hash_to_server_.erase(hash);
}

size_t AffinityCache::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return hash_to_server_.size();
}

}  // namespace tt::gateway
