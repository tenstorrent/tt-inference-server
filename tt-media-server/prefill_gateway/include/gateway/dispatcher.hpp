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
 * Sockets are injected as Senders (function objects) so unit tests can run
 * without real sockets.
 */
class Dispatcher {
 public:
  // Outbound hooks; each returns true on successful socket-layer send.
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

  void onPrefillRequest(const tt::sockets::PrefillRequestMessage& msg);

  // `from_server_id` is the prefill the result arrived on.
  void onPrefillResult(const std::string& from_server_id,
                       const tt::sockets::PrefillResultMessage& msg);

  void onCacheBlocksAdded(
      const tt::sockets::PrefillCacheBlocksAddedMessage& msg);
  void onCacheBlocksEvicted(
      const tt::sockets::PrefillCacheBlocksEvictedMessage& msg);

  // Fails all in-flight tasks assigned to `server_id`.
  void onPrefillDown(const std::string& server_id);

 private:
  void failTaskToDecode(uint32_t task_id, const std::string& reason);

  PrefillRegistry& registry_;
  AffinityCache& affinity_cache_;
  Senders senders_;

  // Tracked per in-flight task so the dispatcher can route results back, roll
  // back inflight counters on send failure, and fail orphaned tasks when a
  // prefill drops.
  struct InFlightEntry {
    std::string prefill_id;
    size_t registration_hash = 0;
  };

  std::mutex inflight_mutex_;
  std::unordered_map<uint32_t, InFlightEntry> in_flight_;
  size_t round_robin_cursor_ = 0;
};

}  // namespace tt::gateway
