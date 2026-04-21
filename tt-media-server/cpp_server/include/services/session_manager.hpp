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
};

class SessionManager {
 public:
  // Result of tryAcquireByPrefixHash: the session's UUID and pre-assigned slot.
  struct AcquiredSession {
    std::string sessionId;
    uint32_t slotId;
  };

  SessionManager();
  ~SessionManager();

  SessionManager(const SessionManager&) = delete;
  SessionManager& operator=(const SessionManager&) = delete;

  void createSession(
      std::function<void(const tt::domain::Session&)> onCompletion,
      std::function<void(std::string_view errorMessage)> onError,
      trantor::EventLoop* eventLoop, const std::string& requestPrompt,
      size_t initialHash = 0, std::optional<uint32_t> slotId = std::nullopt);

  CloseSessionResult closeSession(const std::string& sessionId);
  bool assignSlotId(const std::string& sessionId, uint32_t slotId);
  uint32_t getSlotIdBySessionId(const std::string& sessionId) const;
  uint32_t acquireSessionSlot(const std::string& sessionId);
  std::optional<domain::Session> getSession(const std::string& sessionId) const;
  size_t getActiveSessionCount() const;

  void releaseInFlight(const std::string& sessionId);

  // Register a callback to be invoked immediately if closeSession is called
  // while the session has an in-flight request. The callback should abort the
  // active request. It is cleared automatically once the session is released.
  void setSessionAbortCallback(const std::string& sessionId,
                               std::function<void()> onAbort);

  /**
   * Try to find a session whose registered prefix hash matches, and
   * atomically mark it in-flight so no concurrent request can steal it.
   *
   * Returns:
   *   AcquiredSession — session found and successfully locked; contains both
   *                     slotId and sessionId (UUID). Caller owns the in-flight
   *                     state and MUST call releaseInFlight(sessionId) when
   *                     the request completes (success or error).
   *   nullopt         — no session registered under this hash. Caller should
   *                     fall back to createSession.
   *
   * Throws:
   *   SessionInFlightException — all sessions under this hash are already
   *                              serving other requests. Controller maps
   *                              this to HTTP 429.
   */
  std::optional<AcquiredSession> tryAcquireByPrefixHash(uint64_t prefixHash);

  /**
   * Route future lookups of `prefixHash` to this session. This registers the
   * session under the given hash so the next turn's lookup can find it.
   *
   * If the session was previously registered under a different hash, it is
   * removed from that hash's list and added to the new hash's list.
   *
   * Concurrency: safe to call while the session is in-flight; readers see
   * either the old or new hash atomically.
   */
  void registerPrefixHash(const std::string& sessionId, uint64_t prefixHash);

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
    std::string sessionId;
    uint32_t slotId;
  };

  void sendAsyncAllocationRequest(PendingAllocation& pendingAllocation);
  void evictOldSessions();
  void sendDeallocRequest(const std::string& sessionId, uint32_t slotId);
  void finalizeSessionClose(const std::string& sessionId,
                            const domain::Session& session);
  void readerLoop();
  void retryFailedAllocations();
  void retryFailedDeallocs();
  void handleMemoryResult(const domain::ManageMemoryResult& result);
  void updateSessionCountMetric();

  mutable utils::ConcurrentMap<size_t, std::list<domain::Session>> sessions;
  utils::ConcurrentMap<std::string, std::function<void()>> abortCallbacks_;

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
