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
    // Prompt tokens already cached in the slot (prior turn's full prompt
    // length). Used by the response-id path to prefill only tokens[len:].
    size_t cachedPromptLen = 0;
  };

  SessionManager();
  ~SessionManager();

  SessionManager(const SessionManager&) = delete;
  SessionManager& operator=(const SessionManager&) = delete;

  void createSession(
      std::function<void(const tt::domain::Session&)> onCompletion,
      std::function<void(std::string_view errorMessage)> onError,
      trantor::EventLoop* eventLoop, size_t initialHash = 0,
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
   * Try to find a session whose registered prefix hash matches, atomically
   * mark it in-flight, and register the cancel function — all under the same
   * lock so no concurrent request can steal it.
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
  std::optional<AcquiredSession> tryAcquireByPrefixHash(
      uint64_t prefixHash, std::function<void()> cancelFn);

  /**
   * Route future lookups of `prefixHash` to this session. This registers the
   * session under the given hash so the next turn's lookup can find it.
   *
   * If the session was previously registered under a different hash, it is
   * removed from that hash's index entry and added to the new hash's index
   * entry.
   */
  void registerPrefixHash(const std::string& sessionId, uint64_t prefixHash);

  /**
   * Response-id continuation lookup. Parallel to tryAcquireByPrefixHash but
   * keyed on the OpenAI Responses API `previous_response_id` supplied by the
   * frontend instead of a content hash. Atomically marks the matching session
   * in-flight and registers the cancel function under the same lock.
   *
   * Returns:
   *   AcquiredSession — session found under `previousResponseId` and locked.
   *   nullopt         — no session registered under this id (or id empty).
   *                     Caller should fall back to createSession.
   *
   * Throws:
   *   SessionInFlightException — a session is registered under this id but is
   *                              already serving another request (HTTP 429).
   */
  std::optional<AcquiredSession> tryAcquireByResponseId(
      const std::string& previousResponseId, std::function<void()> cancelFn);

  /**
   * Route future lookups of `responseId` to this session, and record the
   * prompt-token length now committed to the slot's KV cache so the next turn
   * can prefill only the delta. Mirrors registerPrefixHash: if the session
   * was previously registered under a different response id, it is moved to
   * the new id's index entry.
   *
   * `cachedPromptLen` is the full prompt token count of the turn just
   * dispatched (after this turn the slot holds that prefix); pass 0 to leave
   * the recorded length unchanged.
   */
  void registerResponseId(const std::string& sessionId,
                          const std::string& responseId,
                          size_t cachedPromptLen);

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

  // Prefix index helpers: maintain prefixIndex alongside the sessions map.
  void addToPrefixIndex(const std::string& sessionId, uint64_t prefixHash);
  void removeFromPrefixIndex(const std::string& sessionId, uint64_t prefixHash);

  // Response-id index helpers: maintain responseIdIndex alongside the sessions
  // map. Mirror the prefix-index helpers above.
  void addToResponseIdIndex(const std::string& sessionId,
                            const std::string& responseId);
  void removeFromResponseIdIndex(const std::string& sessionId,
                                 const std::string& responseId);

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

  // Secondary index: previous_response_id -> sessions registered under it.
  // Parallel to prefixIndex but keyed on the Responses API response id instead
  // of a content hash; no block hashing is involved. Used by
  // tryAcquireByResponseId / registerResponseId.
  utils::ConcurrentMap<std::string, std::list<std::string>> responseIdIndex;

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
