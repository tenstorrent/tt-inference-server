// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <trantor/net/EventLoop.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include "domain/session.hpp"
#include "ipc/boost_ipc_queue.hpp"
#include "utils/concurrent_map.hpp"
#include "utils/concurrent_queue.hpp"

namespace tt::services {

// Base exception for session errors that should return 429 (rate limit)
class SessionRateLimitException : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class SessionInFlightException : public SessionRateLimitException {
 public:
  SessionInFlightException()
      : SessionRateLimitException(
            "Session already has a request in flight. Multiple concurrent "
            "requests per session are not supported.") {}
};

enum class CloseSessionResult {
  SUCCESS,
  NOT_FOUND,
  IN_FLIGHT,  // session exists but has an active request; dealloc deferred
};

class SessionManager {
 public:
  SessionManager();
  ~SessionManager();

  SessionManager(const SessionManager&) = delete;
  SessionManager& operator=(const SessionManager&) = delete;

  void createSession(
      std::function<void(const tt::domain::Session&)> onCompletion,
      std::function<void(std::string_view errorMessage)> onError,
      trantor::EventLoop* eventLoop, const std::string& requestPrompt);

  CloseSessionResult closeSession(const size_t sessionId);
  bool assignSlotId(const size_t& sessionId, uint32_t slotId);
  uint32_t getSlotIdBySessionId(const size_t& sessionId) const;
  uint32_t acquireSessionSlot(const size_t& sessionId);
  std::optional<domain::Session> getSession(const size_t& sessionId) const;
  size_t getActiveSessionCount() const;

  void setSessionInFlight(const size_t& sessionId, bool inFlight);

  /**
   * Try to find a session whose registered prefix hash matches, and
   * atomically mark it in-flight so no concurrent request can steal it.
   *
   * Returns:
   *   slotId  — session found and successfully locked; caller owns the
   *             in-flight state and MUST call setSessionInFlight(slotId,
   *             false) when the request completes (success or error).
   *   nullopt — no session registered under this hash. Caller should fall
   *             back to createSession.
   *
   * Throws:
   *   SessionInFlightException — all sessions under this hash are already
   *                              serving other requests. Controller maps
   *                              this to HTTP 429.
   */
  std::optional<uint32_t> tryAcquireByPrefixHash(uint64_t prefixHash);

  /**
   * Route future lookups of `prefixHash` to this slot. This registers the
   * slot under the given hash so the next turn's lookup can find it.
   *
   * If the slot was previously registered under a different hash, it is
   * removed from that hash's list and added to the new hash's list.
   *
   * Concurrency: safe to call while the slot is in-flight; readers see
   * either the old or new hash atomically.
   */
  void registerPrefixHash(uint32_t slotId, uint64_t prefixHash);

 private:
  struct PendingAllocation {
    tt::domain::Session session;
    std::function<void(const tt::domain::Session&)> onCompletion;
    std::function<void(std::string_view errorMessage)> onError;
    trantor::EventLoop* eventLoop = nullptr;
    int attemptsRemaining = 0;
    std::chrono::steady_clock::time_point retryAt{};

    PendingAllocation() = default;

    PendingAllocation(
        const tt::domain::Session& session,
        std::function<void(const tt::domain::Session&)> onCompletion,
        std::function<void(std::string_view errorMessage)> onError,
        trantor::EventLoop* eventLoop, int attemptsRemaining)
        : session(session),
          onCompletion(onCompletion),
          onError(onError),
          eventLoop(eventLoop),
          attemptsRemaining(attemptsRemaining) {}
  };

  struct DeferredDealloc {
    size_t sessionId;
    uint32_t slotId;
  };

  void sendAsyncAllocationRequest(PendingAllocation& pendingAllocation);
  void evictOldSessions();
  void sendDeallocRequest(const size_t& sessionId, uint32_t slotId);
  void readerLoop();
  void retryFailedAllocations();
  void retryFailedDeallocs();
  void handleMemoryResult(const domain::ManageMemoryResult& result);

  mutable utils::ConcurrentMap<size_t, std::list<domain::Session>> sessions;

  std::unique_ptr<ipc::MemoryRequestQueue> memoryRequestQueue;
  std::unique_ptr<ipc::MemoryResultQueue> memoryResultQueue;

  utils::ConcurrentMap<uint32_t, PendingAllocation> pendingAllocationsMap;
  utils::ConcurrentQueue<PendingAllocation> pendingAllocationsRetryQueue;
  utils::ConcurrentQueue<DeferredDealloc> deferredDeallocQueue;
  std::atomic<bool> stopped{false};
  std::atomic<bool> evictionInProgress{false};
  std::thread drainThread;
};

}  // namespace tt::services
