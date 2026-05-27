// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
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
  using Clock = std::chrono::steady_clock;

  // Outbound hooks; each returns true on successful socket-layer send.
  struct Senders {
    std::function<bool(const std::string& prefill_server_id,
                       const tt::sockets::PrefillRequestMessage&)>
        sendRequestToPrefill;
    std::function<bool(const std::string& prefill_server_id,
                       const tt::sockets::CancelPrefillMessage&)>
        sendCancelToPrefill;
    std::function<bool(const tt::sockets::PrefillAssignmentMessage&)>
        sendAssignmentToDecode;
    std::function<bool(const tt::sockets::PrefillResultMessage&)>
        sendResultToDecode;
  };

  struct Options {
    std::chrono::milliseconds request_timeout;
    std::chrono::milliseconds timeout_window;
    std::chrono::milliseconds timeout_cooldown;
    uint32_t timeout_threshold;
  };

  Dispatcher(PrefillRegistry& registry, AffinityCache& affinity_cache,
             Senders senders);
  Dispatcher(PrefillRegistry& registry, AffinityCache& affinity_cache,
             Senders senders, Options options);

  Dispatcher(const Dispatcher&) = delete;
  Dispatcher& operator=(const Dispatcher&) = delete;

  void onPrefillRequest(const tt::sockets::PrefillRequestMessage& msg);
  void onPrefillCancel(const tt::sockets::CancelPrefillMessage& msg);

  // `from_server_id` is the prefill the result arrived on.
  void onPrefillResult(const std::string& from_server_id,
                       const tt::sockets::PrefillResultMessage& msg);

  void onCacheBlocksAdded(
      const tt::sockets::PrefillCacheBlocksAddedMessage& msg);
  void onCacheBlocksEvicted(
      const tt::sockets::PrefillCacheBlocksEvictedMessage& msg);

  // Fails all in-flight tasks assigned to `server_id`.
  void onPrefillDown(const std::string& server_id);

  // Fails requests that have been in-flight longer than `request_timeout`.
  void onRequestTimeouts(Clock::time_point now = Clock::now());

 private:
  void failTaskToDecode(uint32_t task_id, const std::string& reason);

  PrefillRegistry& registry_;
  AffinityCache& affinity_cache_;
  Senders senders_;
  Options options_;

  struct InFlightEntry {
    std::string prefill_id;
    size_t registration_hash = 0;
    Clock::time_point started_at;
  };

  std::mutex inflight_mutex_;
  std::unordered_map<uint32_t, InFlightEntry> in_flight_;
  std::mutex timeout_state_mutex_;
  std::unordered_map<std::string, std::deque<Clock::time_point>>
      prefill_timeout_history_;
  std::unordered_map<std::string, Clock::time_point> prefill_blocked_until_;
  size_t round_robin_cursor_ = 0;
};

}  // namespace tt::gateway
