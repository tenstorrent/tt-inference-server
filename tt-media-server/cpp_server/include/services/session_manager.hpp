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
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "domain/session.hpp"
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

enum class CloseSessionResult {
  SUCCESS,
  NOT_FOUND,
};

class SessionManager {
 public:
  struct Candidate {
    std::string sessionId;
    size_t
        matchedBlocks;  // total matched blocks (1 for key + matched remaining)
    size_t sessionBlocks;  // total blocks in the cached session
    uint32_t thinkTokens;  // accumulated think tokens at matched block
  };

  // Result of tryAcquireByPrefixHash: the session's UUID and pre-assigned slot.
  struct AcquiredSession {
    bool sessionFound;
    std::string sessionId;
    uint32_t slotId;
    uint32_t numberOfMatchedTokens = 0;
    // Tokens already committed to the slot's KV cache from the prior turn
    // (that turn's full prompt + generated tokens). Used by the response-id
    // path to prefill only tokens[cachedPromptLen:].
    size_t cachedPromptLen = 0;
    uint32_t accumulatedThinkTokens = 0;  // Think tokens at matched block
    std::vector<Candidate> candidatesList;
  };

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

  // Marks the session in-flight and registers the cancel function atomically.
  // The cancel function is invoked if closeSession is called while in-flight.
  // Returns the assigned slot ID (INVALID_SLOT_ID if not yet allocated).
  uint32_t acquireInFlight(const std::string& sessionId,
                           std::function<void()> cancelFn);

  domain::Session* getSession(const std::string& sessionId);
  size_t getActiveSessionCount() const;

  // Lock/unlock a slot to prevent eviction.
  void lockSlot(uint32_t slotId);
  void unlockSlot(uint32_t slotId);

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
   * Response-id continuation lookup. Parallel to tryAcquireByPrefixHash but
   * keyed on the OpenAI Responses API `previous_response_id`.
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
   * Route future lookups of `responseId` to this session, and record the
   * number of tokens now committed to the slot's KV cache so the next turn
   * can prefill only the delta. If the session was previously registered
   * under a different response id, it is moved to the new id's index entry.
   *
   * Called twice per turn:
   *   - at resolve time with the turn's full prompt length (a safe lower
   *     bound, so a lookup that races the completion still under-prefills
   *     rather than skipping uncached tokens), and
   *   - at completion with full prompt + generated tokens (the actual slot
   *     occupancy), which is what the next turn prefills on top of.
   *
   * `cachedLen` is that token count; pass 0 to leave the recorded length
   * unchanged (only (re)pointing the id at this session).
   */
  void registerResponseId(const std::string& sessionId,
                          const std::string& responseId, size_t cachedLen);

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
  void readerLoop();
  void retryFailedAllocations();
  void retryFailedDeallocs();
  void handleMemoryResult(const domain::ManageMemoryResult& result);
  void updateSessionCountMetric();

  // Prefix index helpers: maintain prefixIndex alongside the sessions map.
  void addToPrefixIndex(const std::string& sessionId, uint64_t prefixHash);
  void removeFromPrefixIndex(const std::string& sessionId, uint64_t prefixHash);

  // Drop the responseId -> session mapping when it points at `sessionId`
  // (called on close/evict). No-op if the id is empty or has been re-pointed
  // at a different session.
  void removeFromResponseIdIndex(const std::string& sessionId,
                                 const std::string& responseId);

  mutable utils::ConcurrentMap<std::string, domain::Session> sessions;

  // An entry in the prefix index: a group of sessions sharing the same prefix
  // path, together with the remaining block info that follows (used for deeper
  // prefix matching / numberOfMatchedTokens calculation).
  struct RemainingBlockInfo {
    uint64_t hash;
    uint32_t accumulatedThinkTokens;
  };

  struct PrefixIndexEntry {
    std::list<std::string> sessionIds;              // sessions registered here
    std::list<RemainingBlockInfo> remainingBlocks;  // subsequent block info
    uint32_t keyBlockThinkTokens = 0;  // think tokens at key hash block
  };

  // Secondary index: block hash -> entries (each with different remaining
  // hashes pointing to different sessions/slots).
  // Used by tryAcquireByPrefixHash / registerPrefixHash for prefix caching.
  utils::ConcurrentMap<uint64_t, std::vector<PrefixIndexEntry>> prefixIndex;

  // Value stored in responseIdIndex: the single session a given
  // previous_response_id resolves to, plus the slot's cached prefix length
  // (prior turn's full prompt + generated tokens) for delta prefill.
  struct ResponseIdEntry {
    std::string sessionId;
    size_t cachedLen = 0;
  };

  // Secondary index: previous_response_id -> the session registered under it.
  // Unlike prefixIndex (where many sessions can share a content hash), response
  // ids are unique per turn, so each id maps to exactly one session and the
  // value is a single entry rather than a list. Used by tryAcquireByResponseId
  // / registerResponseId.
  utils::ConcurrentMap<std::string, ResponseIdEntry> responseIdIndex;

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
