// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_registry.hpp"

#include <utility>

namespace tt::gateway {

void PrefillRegistry::preRegister(const std::string& serverId,
                                  tt::sockets::SocketManager* manager) {
  std::lock_guard<std::mutex> lock(mutex_);
  PrefillPeer peer;
  peer.server_id = serverId;
  peer.socket_manager = manager;
  // healthy stays false until markRegistered() arrives. accepting_tasks
  // stays true so a freshly-registered prefill is eligible immediately.
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
      // Keep in_flight unchanged: dispatcher's onPrefillDown will fail those
      // tasks and decrement counts via the normal path.
      downCb = on_prefill_down_;
    }
  }
  if (wasKnown && downCb) downCb(serverId);
}

void PrefillRegistry::updateLoadInfo(const std::string& serverId,
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
  it->second.cached_blocks.insert(blockHashes.begin(), blockHashes.end());
}

void PrefillRegistry::evictCachedBlocks(
    const std::string& serverId, const std::vector<uint64_t>& blockHashes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(serverId);
  if (it == prefills_.end()) return;
  for (uint64_t h : blockHashes) {
    it->second.cached_blocks.erase(h);
  }
}

std::vector<PrefillSnapshot> PrefillRegistry::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<PrefillSnapshot> out;
  out.reserve(prefills_.size());
  for (const auto& [_, peer] : prefills_) {
    PrefillSnapshot snap;
    snap.server_id = peer.server_id;
    snap.healthy = peer.healthy;
    snap.accepting_tasks = peer.accepting_tasks;
    snap.in_flight = peer.in_flight;
    snap.max_in_flight = peer.max_in_flight;
    out.push_back(std::move(snap));
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
