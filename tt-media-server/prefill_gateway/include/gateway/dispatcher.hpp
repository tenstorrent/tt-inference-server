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

/**
 * @brief Glues prefills + selector into the request lifecycle.
 *
 * Sockets are injected as Senders (function objects) so unit tests can run
 * without real sockets.
 */
class Dispatcher {
 public:
  using Clock = std::chrono::steady_clock;

  // Outbound hooks; each returns true on successful socket-layer send.
  struct Senders {
    std::function<bool(const std::string& prefillServerId,
                       const tt::sockets::PrefillRequestMessage&)>
        sendRequestToPrefill;
    std::function<bool(const std::string& prefillServerId,
                       const tt::sockets::CancelPrefillMessage&)>
        sendCancelToPrefill;
    std::function<bool(const tt::sockets::PrefillResultMessage&)>
        sendResultToDecode;
    std::function<bool(const tt::sockets::SlotReservationRequestMessage&)>
        sendSlotReservationToDecode;
    std::function<bool(const std::string& prefillServerId,
                       const tt::sockets::SlotReservationResponseMessage&)>
        sendSlotReservationToPrefill;
  };

  struct Options {
    std::chrono::milliseconds requestTimeout;
    std::chrono::milliseconds timeoutWindow;
    std::chrono::milliseconds timeoutCooldown;
    uint32_t timeoutThreshold;
  };

  Dispatcher(PrefillRegistry& registry, Senders senders);
  Dispatcher(PrefillRegistry& registry, Senders senders, Options options);

  Dispatcher(const Dispatcher&) = delete;
  Dispatcher& operator=(const Dispatcher&) = delete;

  void onPrefillRequest(const tt::sockets::PrefillRequestMessage& msg);
  void onPrefillCancel(const tt::sockets::CancelPrefillMessage& msg);

  // `fromServerId` is the prefill the result arrived on.
  void onPrefillResult(const std::string& fromServerId,
                       const tt::sockets::PrefillResultMessage& msg);

  void onCacheBlocksAdded(
      const tt::sockets::PrefillCacheBlocksAddedMessage& msg);

  void onSlotReservationRequest(
      const std::string& fromPrefillServerId,
      const tt::sockets::SlotReservationRequestMessage& msg);

  void onSlotReservationResponse(
      const tt::sockets::SlotReservationResponseMessage& msg);

  // Fails all in-flight tasks assigned to `serverId`.
  void onPrefillDown(const std::string& serverId);

  // Fails requests that have been in-flight longer than `requestTimeout`.
  void onRequestTimeouts(Clock::time_point now = Clock::now());

 private:
  struct InFlightEntry {
    std::string prefillId;
    Clock::time_point startedAt;
  };

  void failTaskToDecode(uint32_t taskId, const std::string& reason,
                        const InFlightEntry* entry = nullptr);

  PrefillRegistry& registry;
  Senders senders;
  Options options;

  std::mutex inflightMutex;
  std::unordered_map<uint32_t, InFlightEntry> inFlight;
  std::mutex slotReservationMutex;
  std::unordered_map<uint32_t, std::string> slotReservationRoutes;
  std::mutex timeoutStateMutex;
  std::unordered_map<std::string, std::deque<Clock::time_point>>
      prefillTimeoutHistory;
  std::unordered_map<std::string, Clock::time_point> prefillBlockedUntil;
  size_t roundRobinCursor = 0;
};

}  // namespace tt::gateway
