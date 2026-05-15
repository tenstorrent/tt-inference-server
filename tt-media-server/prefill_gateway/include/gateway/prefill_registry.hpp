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

// Per-prefill runtime state. `socket_manager` is non-owning; tests pass null.
struct PrefillPeer {
  std::string server_id;
  tt::sockets::SocketManager* socket_manager = nullptr;

  bool healthy = false;
  bool accepting_tasks = true;
  uint32_t in_flight = 0;
  uint32_t max_in_flight = 0;

  std::chrono::steady_clock::time_point last_heartbeat{};

  // Block hashes cached on this prefill. Set-based for v0.
  std::unordered_set<uint64_t> cached_blocks;
};

// Thread-safe registry of prefill nodes, keyed by stable `server_id`.
class PrefillRegistry {
 public:
  using PrefillStateCallback =
      std::function<void(const std::string& server_id)>;

  PrefillRegistry() = default;
  PrefillRegistry(const PrefillRegistry&) = delete;
  PrefillRegistry& operator=(const PrefillRegistry&) = delete;

  // Insert a prefill slot (CLIENT-mode socket already created, not yet
  // registered).
  void preRegister(const std::string& server_id,
                   tt::sockets::SocketManager* manager);

  // Mark a prefill ready (PrefillRegistrationMessage received). Returns false
  // if no slot exists for server_id.
  bool markRegistered(const std::string& server_id, uint32_t max_in_flight);

  // Mark a prefill down; caller should re-route in-flight tasks.
  void markDown(const std::string& server_id);

  void updateLoadInfo(const std::string& server_id, bool accepting_tasks);

  void incrementInflight(const std::string& server_id);
  void decrementInflight(const std::string& server_id);  // saturates at 0

  void addCachedBlocks(const std::string& server_id,
                       const std::vector<uint64_t>& block_hashes);
  void evictCachedBlocks(const std::string& server_id,
                         const std::vector<uint64_t>& block_hashes);

  std::vector<PrefillSnapshot> snapshot() const;

  // Non-owning. Valid until the next markDown() for `server_id`.
  tt::sockets::SocketManager* getSocketManager(const std::string& server_id);

  std::vector<std::string> healthyPrefillIds() const;

  void setOnPrefillUp(PrefillStateCallback callback);
  void setOnPrefillDown(PrefillStateCallback callback);

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, PrefillPeer> prefills_;

  PrefillStateCallback on_prefill_up_;
  PrefillStateCallback on_prefill_down_;
};

}  // namespace tt::gateway
