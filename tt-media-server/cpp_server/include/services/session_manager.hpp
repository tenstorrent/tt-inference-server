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
    uint32_t slotId;
    size_t
        matchedBlocks;  // total matched blocks (1 for key + matched remaining)
    size_t sessionBlocks;  // total blocks in the cached session
    uint32_t thinkTokens;  // accumulated think tokens at matched block
  };

  // Result of tryAcquireByPrefixHash: the session's slot ID.
  struct AcquiredSession {
    bool sessionFound;
    uint32_t slotId;
    uint32_t numberOfMatchedTokens = 0;
    uint32_t accumulatedThinkTokens = 0;  // Think tokens at matched block
    std::vector<Candidate> candidatesList;
  };

  SessionManager();
  ~SessionManager();

  SessionManager(const SessionManager&) = delete;
  SessionManager& operator=(const SessionManager&) = delete;

  /**
   * Create a session with a pre-assigned slot (fast path, synchronous).
   * The session is immediately available after this call returns.
   */
  domain::Session createSession(
      uint32_t slotId,
      std::vector<utils::BlockHashInfo> initialBlockInfos = {});

  /**
   * Create a session with slot allocation via IPC (async with callbacks).
   * @param onCompletion Callback invoked when session is created
   * @param onError Callback invoked on allocation failure
   * @param eventLoop Event loop to run callbacks on
   * @param initialBlockInfos Block hashes for prefix cache registration
   * @param slotIdToCopyFrom Optional slot to copy KV cache from
   */
  void createSessionAsync(
      std::function<void(const domain::Session&)> onCompletion,
      std::function<void(std::string_view errorMessage)> onError,
      trantor::EventLoop* eventLoop,
      std::vector<utils::BlockHashInfo> initialBlockInfos = {},
      std::optional<uint32_t> slotIdToCopyFrom = std::nullopt);

  /**
   * Create a session with slot allocation via IPC (blocking).
   * Blocks until slot is allocated or timeout expires.
   * @param initialBlockInfos Block hashes for prefix cache registration
   * @param slotIdToCopyFrom Optional slot to copy KV cache from
   * @param timeout Maximum time to wait for allocation
   * @param errorMsg Output parameter for error message on failure
   * @return The created session, or nullopt on timeout/failure
   */
  std::optional<domain::Session> createSessionSync(
      std::vector<utils::BlockHashInfo> initialBlockInfos = {},
      std::optional<uint32_t> slotIdToCopyFrom = std::nullopt,
      std::chrono::milliseconds timeout = std::chrono::seconds(30),
      std::string* errorMsg = nullptr);

  CloseSessionResult closeSession(uint32_t slotId);
  CloseSessionResult closeSession(const std::string& slotIdStr);

  // Marks the session in-flight and registers the cancel function atomically.
  // The cancel function is invoked if closeSession is called while in-flight.
  // Returns true on success, false if session not found.
  bool acquireInFlight(uint32_t slotId, std::function<void()> cancelFn);

  // Atomically transitions the session from IN_FLIGHT back to IDLE.
  // Thread-safe: holds the ConcurrentMap lock during the state transition.
  void releaseInFlight(uint32_t slotId);

  std::shared_ptr<domain::Session> getSession(uint32_t slotId);
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
   * Given a list of candidates, find one whose matched token count exceeds
   * the MIN_TOKENS_TO_COPY threshold. Matched tokens = firstBlockSize for
   * the first block + kvCacheBlockSize for each subsequent matched block.
   * Candidates are assumed sorted by matchedBlocks descending.
   *
   * @return The best qualifying candidate, or std::nullopt if none qualifies.
   */
  std::optional<Candidate> findASlotToCopyFrom(
      const std::vector<Candidate>& candidates) const;

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
  void registerPrefixHash(uint32_t slotId,
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
  void initResponseId(uint32_t slotId, const std::string& responseId);

  /**
   * Re-key an existing response-id index entry. Looks up the session currently
   * registered under `previousResponseId`, removes that entry, and inserts a
   * new entry under `responseId`. No-op when either id is empty.
   */
  void registerResponseId(const std::string& previousResponseId,
                          const std::string& responseId);

  /**
   * Compute how many tokens of `blockInfos` are already cached for the slot
   * in the prefix index. Used after response-id acquisition to derive the
   * delta. Returns {matchedTokens, accumulatedThinkTokens}.
   */
  std::pair<uint32_t, uint32_t> computeMatchedTokens(
      uint32_t slotId, const std::vector<utils::BlockHashInfo>& blockInfos);

  /**
   * Reset accumulatedThinkTokens to 0 on all prefix index entries that contain
   * the given session. Called when prefill-on-decode overrides thinking tokens
   * so that future lookups report zero cached think tokens for this session.
   */
  void clearSessionBlockThinkTokens(uint32_t slotId);

 private:
  struct PendingAllocation {
    std::vector<utils::BlockHashInfo> initialBlockInfos;
    std::optional<uint32_t> slotIdToCopyFrom;
    int attemptsRemaining = 0;
    std::chrono::steady_clock::time_point retryAt{};

    // For async mode (callback-based)
    std::function<void(const domain::Session&)> onCompletion;
    std::function<void(std::string_view errorMessage)> onError;
    trantor::EventLoop* eventLoop = nullptr;

    // Synchronization for blocking wait (sync mode)
    std::mutex mutex;
    std::condition_variable cv;
    bool completed = false;
    std::optional<uint32_t> resultSlotId;
    std::string errorMessage;

    bool isAsync() const { return eventLoop != nullptr; }
  };

  struct DeferredDealloc {
    uint32_t slotId;
  };

  void sendAsyncAllocationRequest(
      uint32_t taskId, std::shared_ptr<PendingAllocation> allocation);
  void evictOldSessions();
  void sendDeallocRequest(uint32_t slotId);
  void finalizeSessionClose(uint32_t slotId, const domain::Session& session);
  void readerLoop();
  void retryFailedAllocations();
  void retryFailedDeallocs();
  void handleMemoryResult(const domain::ManageMemoryResult& result);
  void updateSessionCountMetric();

  // Prefix index helpers: maintain prefixIndex alongside the sessions map.
  void addToPrefixIndex(uint32_t slotId, uint64_t prefixHash);
  void removeFromPrefixIndex(uint32_t slotId, uint64_t prefixHash);

  // Drop the responseId -> session mapping when it points at `slotId`
  // (called on close/evict). No-op if the id is empty or has been re-pointed
  // at a different session.
  void removeFromResponseIdIndex(uint32_t slotId,
                                 const std::string& responseId);

  // Helper to convert slot ID to map key string
  static std::string slotKey(uint32_t slotId) { return std::to_string(slotId); }

  mutable utils::ConcurrentMap<std::string, std::shared_ptr<domain::Session>>
      sessions;

  // An entry in the prefix index: a group of sessions sharing the same prefix
  // path, together with the remaining block info that follows (used for deeper
  // prefix matching / numberOfMatchedTokens calculation).
  struct RemainingBlockInfo {
    uint64_t hash;
    uint32_t accumulatedThinkTokens;
  };

  struct PrefixIndexEntry {
    std::list<uint32_t> slotIds;                    // slots registered here
    std::list<RemainingBlockInfo> remainingBlocks;  // subsequent block info
    uint32_t keyBlockThinkTokens = 0;  // think tokens at key hash block
  };

  // Secondary index: block hash -> entries (each with different remaining
  // hashes pointing to different sessions/slots).
  // Used by tryAcquireByPrefixHash / registerPrefixHash for prefix caching.
  utils::ConcurrentMap<uint64_t, std::vector<PrefixIndexEntry>> prefixIndex;

  // Secondary index: previous_response_id -> the slot registered under it.
  // Unlike prefixIndex (where many sessions can share a content hash), response
  // ids are unique per turn, so each id maps to exactly one slot. The
  // prefix delta is derived from block matching (computeMatchedTokens), not
  // stored here. Used by tryAcquireByResponseId / registerResponseId.
  utils::ConcurrentMap<std::string, uint32_t> responseIdIndex;

  std::unique_ptr<ipc::boost::MemoryRequestQueue> memoryRequestQueue;
  std::unique_ptr<ipc::boost::MemoryResultQueue> memoryResultQueue;

  utils::ConcurrentMap<uint32_t, std::shared_ptr<PendingAllocation>>
      pendingAllocationsMap;
  utils::ConcurrentQueue<
      std::pair<uint32_t, std::shared_ptr<PendingAllocation>>>
      pendingAllocationsRetryQueue;
  utils::ConcurrentQueue<DeferredDealloc> deferredDeallocQueue;
  std::atomic<bool> stopped{false};
  std::atomic<bool> evictionInProgress{false};
  std::thread drainThread;

  // Slots locked from eviction (O(1) lookup).
  mutable std::mutex lockedSlotsMutex;
  std::unordered_set<uint32_t> lockedSlots;
};

}  // namespace tt::services
