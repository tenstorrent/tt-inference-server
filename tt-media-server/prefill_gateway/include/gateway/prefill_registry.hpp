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

/**
 * @brief Per-prefill runtime state held by the gateway.
 *
 * The socket_manager pointer is the gateway's outbound channel to that
 * specific prefill (gateway in CLIENT mode). The registry holds it as a
 * non-owning pointer; lifetime is managed by the gateway's main() (typically
 * a vector<unique_ptr<SocketManager>>). This keeps the registry testable
 * without spinning up real sockets — tests pass nullptr.
 */
struct PrefillPeer {
  std::string server_id;
  tt::sockets::SocketManager* socket_manager = nullptr;  // non-owning

  bool healthy = false;
  bool accepting_tasks = true;
  uint32_t in_flight = 0;
  uint32_t max_in_flight = 0;

  std::chrono::steady_clock::time_point last_heartbeat{};

  // Per-prefill cache view. Block hashes that this prefill has cached.
  // Set-based for v0; switch to a trie if we measure a need.
  std::unordered_set<uint64_t> cached_blocks;
};

/**
 * @brief Thread-safe registry of prefill nodes.
 *
 * Stores prefill peers only — the gateway's single decode endpoint is held
 * separately by the Dispatcher (decode_facing_). Identity is the prefill's
 * stable `server_id`, not its host/port (addresses can change on reconnect).
 *
 * Responsibilities:
 *   - Track prefills keyed by stable server_id.
 *   - Snapshot prefill state for the selector (avoids holding the lock
 *     during dispatch).
 *   - Maintain in-flight counts as the dispatcher hands tasks in and out.
 *   - Apply cache add/evict deltas from PrefillCacheBlocks* messages.
 *   - Notify callers on prefill-up / prefill-down so the dispatcher can
 *     react (e.g., fail tasks routed to a dropped prefill).
 *
 * Not responsible for socket lifecycle beyond holding the SocketManager
 * instance; the caller wires up handlers and starts the manager.
 */
class PrefillRegistry {
 public:
  using PrefillStateCallback =
      std::function<void(const std::string& server_id)>;

  PrefillRegistry() = default;
  PrefillRegistry(const PrefillRegistry&) = delete;
  PrefillRegistry& operator=(const PrefillRegistry&) = delete;

  /**
   * @brief Insert a prefill slot before the SocketManager is started.
   *
   * Pre-registration is by static config in v1 (host:port list); registration
   * completion happens later via markRegistered() on receipt of
   * PrefillRegistrationMessage. `manager` is non-owning; nullptr is allowed
   * in tests where socket I/O is not exercised.
   */
  void preRegister(const std::string& server_id,
                   tt::sockets::SocketManager* manager);

  /**
   * @brief Mark a prefill as registered (PrefillRegistrationMessage received).
   * @return false if no slot exists for server_id.
   */
  bool markRegistered(const std::string& server_id, uint32_t max_in_flight);

  /**
   * @brief Mark a prefill as down (socket closed / unrecoverable error).
   *
   * Caller is expected to subsequently fail/re-route its in-flight tasks.
   */
  void markDown(const std::string& server_id);

  /**
   * @brief Update load-balance fields from LoadBalanceMessage.
   */
  void updateLoadInfo(const std::string& server_id, bool accepting_tasks);

  /**
   * @brief Increment a prefill's in-flight count when a task is dispatched.
   */
  void incrementInflight(const std::string& server_id);

  /**
   * @brief Decrement on completion or failure. Saturates at 0.
   */
  void decrementInflight(const std::string& server_id);

  /**
   * @brief Apply cache-block deltas from prefill notifications.
   */
  void addCachedBlocks(const std::string& server_id,
                       const std::vector<uint64_t>& block_hashes);
  void evictCachedBlocks(const std::string& server_id,
                         const std::vector<uint64_t>& block_hashes);

  /**
   * @brief Snapshot all prefills for the selector.
   */
  std::vector<PrefillSnapshot> snapshot() const;

  /**
   * @brief Borrow the socket_manager for a specific prefill (for sending).
   *
   * Returns nullptr if the prefill is unknown. The pointer remains valid as
   * long as the slot exists; callers must not retain it past the next
   * markDown() for that prefill.
   */
  tt::sockets::SocketManager* getSocketManager(const std::string& server_id);

  /**
   * @brief Snapshot the IDs of healthy prefills (for failover / iteration).
   */
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
