// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/prefill_registry.hpp"

#include <utility>

namespace tt::gateway {

void PrefillRegistry::addCachedBlock(PrefillPeer& peer, uint64_t blockHash) {
  if (peer.cachedBlocks.insert(blockHash).second) {
    cacheBlockIndex[blockHash].insert(peer.serverId);
  }
}

void PrefillRegistry::clearCachedBlocks(PrefillPeer& peer) {
  for (const uint64_t blockHash : peer.cachedBlocks) {
    removeCachedBlockFromIndex(blockHash, peer.serverId);
  }
  peer.cachedBlocks.clear();
}

void PrefillRegistry::removeCachedBlockFromIndex(uint64_t blockHash,
                                                 const std::string& serverId) {
  auto indexIt = cacheBlockIndex.find(blockHash);
  if (indexIt == cacheBlockIndex.end()) {
    return;
  }
  indexIt->second.erase(serverId);
  if (indexIt->second.empty()) {
    cacheBlockIndex.erase(indexIt);
  }
}

PrefillSnapshot PrefillRegistry::makeSnapshot(const PrefillPeer& peer,
                                              size_t prefixMatchDepth) {
  PrefillSnapshot snap;
  snap.serverId = peer.serverId;
  snap.healthy = peer.healthy;
  snap.acceptingTasks = peer.acceptingTasks;
  snap.inFlight = peer.inFlight;
  snap.maxInFlight = peer.maxInFlight;
  snap.cachedBlocks = peer.cachedBlocks.size();
  snap.prefixMatchDepth = prefixMatchDepth;
  snap.lastHeartbeat = peer.lastHeartbeat;
  return snap;
}

void PrefillRegistry::preRegister(const std::string& serverId,
                                  tt::sockets::SocketManager* manager) {
  std::lock_guard<std::mutex> lock(mutex);
  // New entries stay unhealthy until markRegistered() arrives. Existing entries
  // keep their runtime/cache state but refresh the non-owning socket pointer.
  auto it = prefills.try_emplace(serverId).first;
  it->second.serverId = serverId;
  it->second.socketManager = manager;
}

bool PrefillRegistry::markRegistered(const std::string& serverId,
                                     uint32_t maxInFlight) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = prefills.find(serverId);
  if (it == prefills.end()) return false;
  it->second.healthy = true;
  it->second.maxInFlight = maxInFlight;
  it->second.lastHeartbeat = std::chrono::steady_clock::now();
  return true;
}

void PrefillRegistry::markDown(const std::string& serverId) {
  PrefillStateCallback downCb;
  bool wasKnown = false;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = prefills.find(serverId);
    if (it != prefills.end()) {
      wasKnown = true;
      it->second.healthy = false;
      clearCachedBlocks(it->second);
      // Keep inFlight unchanged: dispatcher's onPrefillDown will fail those
      // tasks and decrement counts via the normal path.
      downCb = onPrefillDown;
    }
  }
  if (wasKnown && downCb) downCb(serverId);
}

void PrefillRegistry::setAcceptingTasks(const std::string& serverId,
                                        bool acceptingTasks) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = prefills.find(serverId);
  if (it == prefills.end()) return;
  it->second.acceptingTasks = acceptingTasks;
}

void PrefillRegistry::incrementInflight(const std::string& serverId) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = prefills.find(serverId);
  if (it == prefills.end()) return;
  ++it->second.inFlight;
}

void PrefillRegistry::decrementInflight(const std::string& serverId) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = prefills.find(serverId);
  if (it == prefills.end()) return;
  if (it->second.inFlight > 0) --it->second.inFlight;
}

void PrefillRegistry::addCachedBlocks(
    const std::string& serverId, const std::vector<uint64_t>& blockHashes) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = prefills.find(serverId);
  if (it == prefills.end()) return;
  for (const uint64_t blockHash : blockHashes) {
    addCachedBlock(it->second, blockHash);
  }
}

std::vector<PrefillSnapshot> PrefillRegistry::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex);
  std::vector<PrefillSnapshot> out;
  out.reserve(prefills.size());
  for (const auto& [_, peer] : prefills) {
    out.push_back(makeSnapshot(peer, /*prefixMatchDepth=*/0));
  }
  return out;
}

std::vector<PrefillSnapshot> PrefillRegistry::routingSnapshot(
    const std::vector<uint64_t>& registrationHashes) const {
  std::lock_guard<std::mutex> lock(mutex);

  std::unordered_map<std::string, size_t> prefixDepthByServer;
  std::unordered_set<std::string> candidates;

  for (size_t depth = 0; depth < registrationHashes.size(); ++depth) {
    auto indexIt = cacheBlockIndex.find(registrationHashes[depth]);
    if (indexIt == cacheBlockIndex.end()) {
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
  out.reserve(prefills.size());
  for (const auto& [serverId, peer] : prefills) {
    const auto depthIt = prefixDepthByServer.find(serverId);
    const size_t prefixDepth =
        depthIt == prefixDepthByServer.end() ? 0 : depthIt->second;
    out.push_back(makeSnapshot(peer, prefixDepth));
  }
  return out;
}

tt::sockets::SocketManager* PrefillRegistry::getSocketManager(
    const std::string& serverId) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = prefills.find(serverId);
  if (it == prefills.end()) return nullptr;
  return it->second.socketManager;
}

void PrefillRegistry::setOnPrefillDown(PrefillStateCallback callback) {
  std::lock_guard<std::mutex> lock(mutex);
  onPrefillDown = std::move(callback);
}

}  // namespace tt::gateway
