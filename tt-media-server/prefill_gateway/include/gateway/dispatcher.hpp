// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

#include "sockets/socket_messages.hpp"

namespace tt::gateway {

class PrefillRegistry;
class AffinityCache;

/**
 * @brief Glues prefills + selector + affinity cache into the request lifecycle.
 *
 * Flow (v1):
 *   decode --PrefillRequest--> Dispatcher
 *     Dispatcher consults the selector + AffinityCache to pick a prefill.
 *     If a prefill is picked:
 *       1. inflight++ on the prefill
 *       2. send PrefillAssignment to decode
 *       3. send PrefillRequest to chosen prefill
 *     If no prefill is available:
 *       send PrefillResult{error=true} back to decode immediately.
 *
 *   prefill --PrefillResult--> Dispatcher
 *     Dispatcher records affinity (hash -> server_id), inflight--, and
 *     forwards the result to decode unchanged.
 *
 *   prefill --PrefillCacheBlocksAdded/Evicted--> Dispatcher
 *     Dispatcher applies the delta to PrefillRegistry's per-prefill cache
 *     view.
 *
 *   Prefill-down callback from PrefillRegistry:
 *     Fail every in-flight task assigned to that prefill, evict its
 *     affinity-cache entries.
 *
 * Sockets are injected as Senders (function objects), not held directly,
 * so unit tests can capture sends without spinning up real sockets. Real
 * deployments wire Senders that call into SocketManager.
 */
class Dispatcher {
 public:
  /**
   * @brief Callable hooks for outbound traffic.
   *
   * Each returns true on successful send (delivered to the OS socket layer;
   * not necessarily ACK'd by the peer).
   */
  struct Senders {
    std::function<bool(const std::string& prefill_server_id,
                       const tt::sockets::PrefillRequestMessage&)>
        sendRequestToPrefill;
    std::function<bool(const tt::sockets::PrefillAssignmentMessage&)>
        sendAssignmentToDecode;
    std::function<bool(const tt::sockets::PrefillResultMessage&)>
        sendResultToDecode;
  };

  Dispatcher(PrefillRegistry& registry, AffinityCache& affinity_cache,
             Senders senders);

  Dispatcher(const Dispatcher&) = delete;
  Dispatcher& operator=(const Dispatcher&) = delete;

  /**
   * @brief Handle an incoming PrefillRequest from decode.
   */
  void onPrefillRequest(const tt::sockets::PrefillRequestMessage& msg);

  /**
   * @brief Handle a PrefillResult arriving from any prefill.
   * @param from_server_id  The prefill the result arrived on.
   */
  void onPrefillResult(const std::string& from_server_id,
                       const tt::sockets::PrefillResultMessage& msg);

  /**
   * @brief Handle PrefillCacheBlocksAdded from a prefill.
   */
  void onCacheBlocksAdded(
      const tt::sockets::PrefillCacheBlocksAddedMessage& msg);

  /**
   * @brief Handle PrefillCacheBlocksEvicted from a prefill.
   */
  void onCacheBlocksEvicted(
      const tt::sockets::PrefillCacheBlocksEvictedMessage& msg);

  /**
   * @brief PrefillRegistry hook: invoked when a prefill goes down. Fails
   * every in-flight task that was assigned to that prefill.
   */
  void onPrefillDown(const std::string& server_id);

 private:
  // Reply to decode with an error result when no prefill can take the task,
  // or when an assigned prefill dies mid-flight.
  void failTaskToDecode(uint32_t task_id, const std::string& reason);

  PrefillRegistry& registry_;
  AffinityCache& affinity_cache_;
  Senders senders_;

  std::mutex inflight_mutex_;
  // task_id -> server_id, lets us decrement the right prefill on result and
  // fail the right tasks on prefill-down.
  std::unordered_map<uint32_t, std::string> in_flight_task_to_prefill_;
  // task_id -> registration_hash, kept so onPrefillResult can record the
  // affinity without depending on the original request surviving.
  std::unordered_map<uint32_t, size_t> in_flight_task_to_hash_;
  size_t round_robin_cursor_ = 0;
};

}  // namespace tt::gateway
