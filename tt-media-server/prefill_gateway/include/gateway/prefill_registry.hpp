// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gateway/prefill_selector.hpp"

namespace tt::sockets {
class SocketManager;  // forward; lifetime owned outside the registry
}

namespace tt::gateway {

// Thread-safe registry of prefill nodes, keyed by stable `serverId`.
class PrefillRegistry {
 public:
  using PrefillStateCallback =
      std::function<void(const std::string& serverId)>;

  PrefillRegistry() = default;
  PrefillRegistry(const PrefillRegistry&) = delete;
  PrefillRegistry& operator=(const PrefillRegistry&) = delete;

  // Insert a prefill slot (CLIENT-mode socket already created, not yet
  // registered).
  void preRegister(const std::string& serverId,
                   tt::sockets::SocketManager* manager);

  // Mark a prefill ready (PrefillRegistrationMessage received). Returns false
  // if no slot exists for serverId.
  bool markRegistered(const std::string& serverId, uint32_t maxInFlight);

  // Mark a prefill down; caller should re-route in-flight tasks.
  void markDown(const std::string& serverId);

  void setAcceptingTasks(const std::string& serverId, bool acceptingTasks);

  void incrementInflight(const std::string& serverId);
  void decrementInflight(const std::string& serverId);  // saturates at 0

  void addCachedBlocks(const std::string& serverId,
                       const std::vector<uint64_t>& blockHashes);

  std::vector<PrefillSnapshot> snapshot() const;
  std::vector<PrefillSnapshot> routingSnapshot(
      const std::vector<uint64_t>& registrationHashes) const;

  // Non-owning. Valid until the next markDown() for `serverId`.
  tt::sockets::SocketManager* getSocketManager(const std::string& serverId);

  void setOnPrefillDown(PrefillStateCallback callback);

 private:
  // Per-prefill runtime state. `socketManager` is non-owning; tests pass null.
  struct PrefillPeer {
    std::string serverId;
    tt::sockets::SocketManager* socketManager = nullptr;

    bool healthy = false;
    bool acceptingTasks = true;
    uint32_t inFlight = 0;
    uint32_t maxInFlight = 0;

    std::chrono::steady_clock::time_point lastHeartbeat{};
    std::unordered_set<uint64_t> cachedBlocks;
  };

  using ServerIdSet = std::unordered_set<std::string>;
  using CacheBlockIndex = std::unordered_map<uint64_t, ServerIdSet>;

  void addCachedBlock(PrefillPeer& peer, uint64_t blockHash);
  void clearCachedBlocks(PrefillPeer& peer);
  void removeCachedBlockFromIndex(uint64_t blockHash,
                                  const std::string& serverId);
  static PrefillSnapshot makeSnapshot(const PrefillPeer& peer,
                                      size_t prefixMatchDepth);

  mutable std::mutex mutex;
  std::unordered_map<std::string, PrefillPeer> prefills;
  // Inverted cache index used for request-time longest-prefix matching.
  CacheBlockIndex cacheBlockIndex;

  PrefillStateCallback onPrefillDown;
};

}  // namespace tt::gateway
