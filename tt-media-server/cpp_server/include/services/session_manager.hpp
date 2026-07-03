// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <trantor/net/EventLoop.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "ipc/boost/boost_memory_queue.hpp"
#include "utils/concurrent_map.hpp"
#include "utils/concurrent_queue.hpp"
#include "utils/conversation_hasher.hpp"

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

}  // namespace tt::services

#include "services/prefix_cache_router.hpp"

namespace tt::services {

enum class CloseSessionResult {
  SUCCESS,
  NOT_FOUND,
};

class SessionManager {
 public:
  using Candidate = domain::prefix_cache::Candidate;
  using AcquiredSession = PrefixCacheAcquireResult;

  SessionManager();
  ~SessionManager();

  SessionManager(const SessionManager&) = delete;
  SessionManager& operator=(const SessionManager&) = delete;

  void createSession(
      std::function<void(const tt::domain::Session&)> onCompletion,
      std::function<void(std::string_view errorMessage)> onError,
      trantor::EventLoop* eventLoop,
      std::vector<utils::BlockHashInfo> initialBlockInfos = {},
      std::optional<uint32_t> slotId = std::nullopt,
      std::optional<uint32_t> slotIdToCopyFrom = std::nullopt);

  CloseSessionResult closeSession(const std::string& sessionId);
  bool assignSlotId(const std::string& sessionId, uint32_t slotId);
  uint32_t getSlotIdBySessionId(const std::string& sessionId) const;
  uint32_t getCommittedBlocks(const std::string& sessionId) const;

  // Marks the session in-flight and registers the cancel function atomically.
  // The cancel function is invoked if closeSession is called while in-flight.
  // Returns the assigned slot ID (INVALID_SLOT_ID if not yet allocated).
  uint32_t acquireInFlight(const std::string& sessionId,
                           std::function<void()> cancelFn);

  // Atomically transitions the session from IN_FLIGHT back to IDLE.
  // Thread-safe: holds the ConcurrentMap lock during the state transition.
  void releaseInFlight(const std::string& sessionId);

  std::shared_ptr<domain::Session> getSession(const std::string& sessionId);
  size_t getActiveSessionCount() const;

  MarkInFlightResult tryMarkInFlight(
      const std::string& sessionId, std::function<void()>& cancelFn,
      std::optional<uint64_t> expectedKeyHash = std::nullopt,
      const std::string* expectedResponseId = nullptr);
  std::optional<uint64_t> getSessionHash(const std::string& sessionId) const;
  bool setSessionHash(const std::string& sessionId, uint64_t keyHash);
  bool setSessionResponseId(const std::string& sessionId,
                            const std::string& responseId);
  void unlockSlot(uint32_t slotId);

  // Lock a slot to prevent eviction.
  void lockSlot(uint32_t slotId);

  /**
   * Try to find a session whose registered prefix hash matches one of the
   * provided block infos. Searches from the longest prefix (last hash) to
   * the shortest (first hash) to maximize KV cache reuse. Atomically marks
   * the session in-flight and registers the cancel function.
   *
   * @param blockInfos  Per-block hash and think count info (index 0 = first).
   * @param cancelFn    Cancel function registered on the acquired session.
   *
   * Returns:
   *   AcquiredSession — session found; contains sessionId, slotId,
   *                     numberOfMatchedTokens, and accumulatedThinkTokens.
   *                     Caller owns the in-flight state and MUST release
   *                     when the request completes.
   *   nullopt         — no session registered under any hash. Caller should
   *                     fall back to createSession.
   *
   * Throws:
   *   SessionInFlightException — all candidate sessions are already
   *                              serving other requests (maps to HTTP 429).
   */
  std::optional<AcquiredSession> tryAcquireByPrefixHash(
      const std::vector<utils::BlockHashInfo>& blockInfos,
      std::function<void()> cancelFn);

  /**
   * Route future lookups to this session by registering the given block infos.
   * blockInfos[0].hash becomes the key in prefixIndex; blockInfos[1:] are
   * stored as remainingBlocks in the entry. If an entry with identical
   * remaining hashes already exists, the session is added to that entry;
   * otherwise a new entry is created.
   *
   * If the session was previously registered under a different key hash, it is
   * removed from that hash's index entry first.
   */
  void registerPrefixHash(const std::string& sessionId,
                          const std::vector<utils::BlockHashInfo>& blockInfos);

  /**
   * Response-id continuation lookup.
   * Atomically marks the matching session in-flight and registers the cancel
   * function under the same lock.
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
   * First-time registration: associate a brand-new session with a response id.
   */
  void registerResponseId(const std::string& sessionId,
                          const std::string& responseId);

  /**
   * Update the response-id index entry. Looks up the session currently
   * registered under `previousResponseId`, removes that entry, and inserts a
   * new entry under `responseId`. No-op when either id is empty.
   */
  void updateResponseId(const std::string& previousResponseId,
                        const std::string& responseId);

  /**
   * Compute how many tokens of `blockInfos` are already cached for `sessionId`
   * in the prefix index. Used after response-id acquisition to derive the
   * delta. Returns {matchedTokens, accumulatedThinkTokens}.
   */
  std::pair<uint32_t, uint32_t> computeMatchedTokens(
      const std::string& sessionId,
      const std::vector<utils::BlockHashInfo>& blockInfos);

  /**
   * Reset accumulatedThinkTokens to 0 on all prefix index entries that contain
   * the given session. Called when prefill-on-decode overrides thinking tokens
   * so that future lookups report zero cached think tokens for this session.
   */
  void clearSessionBlockThinkTokens(const std::string& sessionId);

  /**
   * Mark the leading `residentBlocks` blocks of `sessionId`'s prefix as
   * resident (KV computed and safe to copy from). Called when a prefill
   * completes. Sets the count outright — the whole prompt is resident at that
   * point. See Session::committedBlocks.
   */
  void setResidentPrefixBlocks(const std::string& sessionId,
                               uint32_t residentBlocks);

  /**
   * Eagerly shrink `sessionId`'s resident-prefix count to the block count that
   * corresponds to `matchedTokens` (the common prefix this turn shares with the
   * cached session). On a divergent "rewind" turn this drops the now-stale tail
   * before the slot overwrites it; on a pure extension it is a no-op. Called on
   * a prefix-cache / response-id continuation HIT.
   */
  void shrinkResidentPrefixToMatchedTokens(const std::string& sessionId,
                                           uint32_t matchedTokens);

 private:
  struct PendingAllocation {
    tt::domain::Session session;
    std::function<void(const tt::domain::Session&)> onCompletion;
    std::function<void(std::string_view errorMessage)> onError;
    trantor::EventLoop* eventLoop = nullptr;
    int attemptsRemaining = 0;
    std::chrono::steady_clock::time_point retryAt{};
    std::optional<uint32_t> slotIdToCopyFrom;
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
  // Wrap a Session into the map's shared_ptr value and inject its release hook.
  void insertSession(const domain::Session& session);
  void readerLoop();
  void retryFailedAllocations();
  void retryFailedDeallocs();
  void handleMemoryResult(const domain::ManageMemoryResult& result);
  void updateSessionCountMetric();

  mutable std::unique_ptr<PrefixCacheRouter> prefixCacheRouter;

  mutable utils::ConcurrentMap<std::string, std::shared_ptr<domain::Session>>
      sessions;

  std::unique_ptr<ipc::boost::MemoryRequestQueue> memoryRequestQueue;
  std::unique_ptr<ipc::boost::MemoryResultQueue> memoryResultQueue;

  utils::ConcurrentMap<uint32_t, PendingAllocation> pendingAllocationsMap;
  utils::ConcurrentQueue<PendingAllocation> pendingAllocationsRetryQueue;
  utils::ConcurrentQueue<DeferredDealloc> deferredDeallocQueue;
  std::atomic<bool> stopped{false};
  std::atomic<bool> evictionInProgress{false};
  std::thread drainThread;

  // Slots locked from eviction (O(1) lookup).
  mutable std::mutex lockedSlotsMutex;
  std::unordered_set<uint32_t> lockedSlots;
};

}  // namespace tt::services
