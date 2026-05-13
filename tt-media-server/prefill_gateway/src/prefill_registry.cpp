// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_registry.hpp"

#include <utility>

namespace tt::gateway {

void PrefillRegistry::preRegister(const std::string& server_id,
                                  tt::sockets::SocketManager* manager) {
  std::lock_guard<std::mutex> lock(mutex_);
  PrefillPeer peer;
  peer.server_id = server_id;
  peer.socket_manager = manager;
  // healthy stays false until markRegistered() arrives. accepting_tasks
  // stays true so a freshly-registered prefill is eligible immediately.
  prefills_.emplace(server_id, std::move(peer));
}

bool PrefillRegistry::markRegistered(const std::string& server_id,
                                     uint32_t max_in_flight) {
  PrefillStateCallback up_cb;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = prefills_.find(server_id);
    if (it == prefills_.end()) return false;
    it->second.healthy = true;
    it->second.max_in_flight = max_in_flight;
    it->second.last_heartbeat = std::chrono::steady_clock::now();
    up_cb = on_prefill_up_;
  }
  if (up_cb) up_cb(server_id);
  return true;
}

void PrefillRegistry::markDown(const std::string& server_id) {
  PrefillStateCallback down_cb;
  bool was_known = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = prefills_.find(server_id);
    if (it != prefills_.end()) {
      was_known = true;
      it->second.healthy = false;
      // Keep in_flight unchanged: dispatcher's onPrefillDown will fail those
      // tasks and decrement counts via the normal path.
      down_cb = on_prefill_down_;
    }
  }
  if (was_known && down_cb) down_cb(server_id);
}

void PrefillRegistry::updateLoadInfo(const std::string& server_id,
                                     bool accepting_tasks) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(server_id);
  if (it == prefills_.end()) return;
  it->second.accepting_tasks = accepting_tasks;
}

void PrefillRegistry::incrementInflight(const std::string& server_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(server_id);
  if (it == prefills_.end()) return;
  ++it->second.in_flight;
}

void PrefillRegistry::decrementInflight(const std::string& server_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(server_id);
  if (it == prefills_.end()) return;
  if (it->second.in_flight > 0) --it->second.in_flight;
}

void PrefillRegistry::addCachedBlocks(
    const std::string& server_id,
    const std::vector<uint64_t>& block_hashes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(server_id);
  if (it == prefills_.end()) return;
  it->second.cached_blocks.insert(block_hashes.begin(), block_hashes.end());
}

void PrefillRegistry::evictCachedBlocks(
    const std::string& server_id,
    const std::vector<uint64_t>& block_hashes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(server_id);
  if (it == prefills_.end()) return;
  for (uint64_t h : block_hashes) {
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
    const std::string& server_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = prefills_.find(server_id);
  if (it == prefills_.end()) return nullptr;
  return it->second.socket_manager;
}

std::vector<std::string> PrefillRegistry::healthyPrefillIds() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> out;
  out.reserve(prefills_.size());
  for (const auto& [id, peer] : prefills_) {
    if (peer.healthy) out.push_back(id);
  }
  return out;
}

void PrefillRegistry::setOnPrefillUp(PrefillStateCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  on_prefill_up_ = std::move(callback);
}

void PrefillRegistry::setOnPrefillDown(PrefillStateCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  on_prefill_down_ = std::move(callback);
}

}  // namespace tt::gateway
