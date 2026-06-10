// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_registry.hpp"

#include <utility>

namespace tt::gateway {

void PrefillRegistry::removeCachedBlockFromIndex(uint64_t blockHash,
                                                 const std::string& serverId) {
  auto indexIt = cache_block_index_.find(blockHash);
  if (indexIt == cache_block_index_.end()) {
    return;
  }
  indexIt->second.erase(serverId);
  if (indexIt->second.empty()) {
    cache_block_index_.erase(indexIt);
  }
}

PrefillSnapshot PrefillRegistry::makeSnapshot(const PrefillPeer& peer,
                                              size_t prefixMatchDepth) {
  PrefillSnapshot snap;
  snap.server_id = peer.server_id;
  snap.healthy = peer.healthy;
  snap.accepting_tasks = peer.accepting_tasks;
  snap.in_flight = peer.in_flight;
  snap.max_in_flight = peer.max_in_flight;
  snap.cached_blocks = peer.cached_blocks.size();
  snap.prefix_match_depth = prefixMatchDepth;
  snap.last_heartbeat = peer.last_heartbeat;
  return snap;
}

void PrefillRegistry::preRegister(const std::string& serverId,
                                  tt::sockets::SocketManager* manager) {
  std::lock_guard<std::mutex> lock(mutex_);
  PrefillPeer peer;
  peer.server_id = serverId;
  peer.socket_manager = manager;
  // healthy stays false until markRegistered() arrives.
  prefills_.emplace(serverId, std::move(peer));
}

bool PrefillRegistry::markRegistered(const std::string& serverId,
                                     uint32_t maxInFlight) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(serverId);
  if (it == prefills_.end()) return false;
  it->second.healthy = true;
  it->second.max_in_flight = maxInFlight;
  it->second.last_heartbeat = std::chrono::steady_clock::now();
  return true;
}

void PrefillRegistry::markDown(const std::string& serverId) {
  PrefillStateCallback downCb;
  bool wasKnown = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = prefills_.find(serverId);
    if (it != prefills_.end()) {
      wasKnown = true;
      it->second.healthy = false;
      for (const uint64_t blockHash : it->second.cached_blocks) {
        removeCachedBlockFromIndex(blockHash, serverId);
      }
      it->second.cached_blocks.clear();
      // Keep in_flight unchanged: dispatcher's onPrefillDown will fail those
      // tasks and decrement counts via the normal path.
      downCb = on_prefill_down_;
    }
  }
  if (wasKnown && downCb) downCb(serverId);
}

void PrefillRegistry::setAcceptingTasks(const std::string& serverId,
                                        bool acceptingTasks) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(serverId);
  if (it == prefills_.end()) return;
  it->second.accepting_tasks = acceptingTasks;
}

void PrefillRegistry::incrementInflight(const std::string& serverId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(serverId);
  if (it == prefills_.end()) return;
  ++it->second.in_flight;
}

void PrefillRegistry::decrementInflight(const std::string& serverId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(serverId);
  if (it == prefills_.end()) return;
  if (it->second.in_flight > 0) --it->second.in_flight;
}

void PrefillRegistry::addCachedBlocks(
    const std::string& serverId, const std::vector<uint64_t>& blockHashes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(serverId);
  if (it == prefills_.end()) return;
  for (const uint64_t blockHash : blockHashes) {
    if (it->second.cached_blocks.insert(blockHash).second) {
      cache_block_index_[blockHash].insert(serverId);
    }
  }
}

void PrefillRegistry::evictCachedBlocks(
    const std::string& serverId, const std::vector<uint64_t>& blockHashes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(serverId);
  if (it == prefills_.end()) return;
  for (uint64_t h : blockHashes) {
    if (it->second.cached_blocks.erase(h) > 0) {
      removeCachedBlockFromIndex(h, serverId);
    }
  }
}

std::vector<PrefillSnapshot> PrefillRegistry::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<PrefillSnapshot> out;
  out.reserve(prefills_.size());
  for (const auto& [_, peer] : prefills_) {
    out.push_back(makeSnapshot(peer, /*prefixMatchDepth=*/0));
  }
  return out;
}

std::vector<PrefillSnapshot> PrefillRegistry::routingSnapshot(
    const std::vector<uint64_t>& registrationHashes) const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::unordered_map<std::string, size_t> prefixDepthByServer;
  std::unordered_set<std::string> candidates;

  for (size_t depth = 0; depth < registrationHashes.size(); ++depth) {
    auto indexIt = cache_block_index_.find(registrationHashes[depth]);
    if (indexIt == cache_block_index_.end()) {
      break;
    }

    if (depth == 0) {
      candidates = indexIt->second;
    } else {
      std::erase_if(candidates, [&](const std::string& serverId) {
        return !indexIt->second.contains(serverId);
      });
    }

    if (candidates.empty()) {
      break;
    }

    for (const auto& serverId : candidates) {
      prefixDepthByServer[serverId] = depth + 1;
    }
  }

  std::vector<PrefillSnapshot> out;
  out.reserve(prefills_.size());
  for (const auto& [serverId, peer] : prefills_) {
    const auto depthIt = prefixDepthByServer.find(serverId);
    const size_t prefixDepth =
        depthIt == prefixDepthByServer.end() ? 0 : depthIt->second;
    out.push_back(makeSnapshot(peer, prefixDepth));
  }
  return out;
}

tt::sockets::SocketManager* PrefillRegistry::getSocketManager(
    const std::string& serverId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(serverId);
  if (it == prefills_.end()) return nullptr;
  return it->second.socket_manager;
}

void PrefillRegistry::setOnPrefillDown(PrefillStateCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  on_prefill_down_ = std::move(callback);
}

}  // namespace tt::gateway
