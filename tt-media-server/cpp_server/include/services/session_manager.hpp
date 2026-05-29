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
#include <vector>

#include "domain/session.hpp"
#include "ipc/boost/boost_memory_queue.hpp"
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
    uint32_t numberOfMatchedTokens = 0;
  };

  SessionManager();
  ~SessionManager();

  SessionManager(const SessionManager&) = delete;
  SessionManager& operator=(const SessionManager&) = delete;

  void createSession(
      std::function<void(const tt::domain::Session&)> onCompletion,
      std::function<void(std::string_view errorMessage)> onError,
      trantor::EventLoop* eventLoop,
      std::vector<uint64_t> initialBlockHashes = {},
      std::optional<uint32_t> slotId = std::nullopt);

  CloseSessionResult closeSession(const std::string& sessionId);
  bool assignSlotId(const std::string& sessionId, uint32_t slotId);
  uint32_t getSlotIdBySessionId(const std::string& sessionId) const;

  // Marks the session in-flight and registers the cancel function atomically.
  // The cancel function is invoked if closeSession is called while in-flight.
  // Returns the assigned slot ID (INVALID_SLOT_ID if not yet allocated).
  uint32_t acquireInFlight(const std::string& sessionId,
                           std::function<void()> cancelFn);

  domain::Session* getSession(const std::string& sessionId);
  size_t getActiveSessionCount() const;

  /**
   * Try to find a session whose registered prefix hash matches one of the
   * provided block hashes. Searches from the longest prefix (last hash) to
   * the shortest (first hash) to maximize KV cache reuse. Atomically marks
   * the session in-flight and registers the cancel function.
   *
   * @param blockHashes  Per-block prefix hashes (index 0 = first block).
   * @param cancelFn     Cancel function registered on the acquired session.
   *
   * Returns:
   *   AcquiredSession — session found; contains sessionId, slotId, and
   *                     numberOfMatchedTokens. Caller owns the in-flight
   *                     state and MUST release when the request completes.
   *   nullopt         — no session registered under any hash. Caller should
   *                     fall back to createSession.
   *
   * Throws:
   *   SessionInFlightException — all candidate sessions are already
   *                              serving other requests (maps to HTTP 429).
   */
  std::optional<AcquiredSession> tryAcquireByPrefixHash(
      const std::vector<uint64_t>& blockHashes, std::function<void()> cancelFn);

  /**
   * Route future lookups to this session by registering the given block hashes.
   * blockHashes[0] becomes the key in prefixIndex; blockHashes[1:] are stored
   * as remainingHashes in the entry. If an entry with identical remainingHashes
   * already exists, the session is added to that entry; otherwise a new entry
   * is created.
   *
   * If the session was previously registered under a different key hash, it is
   * removed from that hash's index entry first.
   */
  void registerPrefixHash(const std::string& sessionId,
                          const std::vector<uint64_t>& blockHashes);

 private:
  struct PendingAllocation {
    tt::domain::Session session;
    std::function<void(const tt::domain::Session&)> onCompletion;
    std::function<void(std::string_view errorMessage)> onError;
    trantor::EventLoop* eventLoop = nullptr;
    int attemptsRemaining = 0;
    std::chrono::steady_clock::time_point retryAt{};
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

  void removeFromPrefixIndex(const std::string& sessionId, uint64_t prefixHash);

  mutable utils::ConcurrentMap<std::string, domain::Session> sessions;

  // An entry in the prefix index: a group of sessions sharing the same prefix
  // path, together with the remaining block hashes that follow (used for deeper
  // prefix matching / numberOfMatchedTokens calculation).
  struct PrefixIndexEntry {
    std::list<std::string> sessionIds;    // sessions registered here
    std::list<uint64_t> remainingHashes;  // subsequent block hashes
  };

  // Secondary index: block hash -> entries (each with different remaining
  // hashes pointing to different sessions/slots).
  // Used by tryAcquireByPrefixHash / registerPrefixHash for prefix caching.
  utils::ConcurrentMap<uint64_t, std::vector<PrefixIndexEntry>> prefixIndex;

  std::unique_ptr<ipc::boost::MemoryRequestQueue> memoryRequestQueue;
  std::unique_ptr<ipc::boost::MemoryResultQueue> memoryResultQueue;

  utils::ConcurrentMap<uint32_t, PendingAllocation> pendingAllocationsMap;
  utils::ConcurrentQueue<PendingAllocation> pendingAllocationsRetryQueue;
  utils::ConcurrentQueue<DeferredDealloc> deferredDeallocQueue;
  std::atomic<bool> stopped{false};
  std::atomic<bool> evictionInProgress{false};
  std::thread drainThread;
};

}  // namespace tt::services
