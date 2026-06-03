// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/session_manager.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "metrics/metrics.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

namespace tt::services {

namespace {
constexpr std::chrono::milliseconds ALLOCATION_RETRY_BASE_DELAY{2000};
constexpr std::chrono::milliseconds ALLOCATION_RETRY_DELAY_STEP{700};
std::chrono::milliseconds allocationRetryMaxDelay =
    ALLOCATION_RETRY_BASE_DELAY +
    ALLOCATION_RETRY_DELAY_STEP *
        (tt::config::sessionAllocationMaxRetries() - 1);
constexpr std::chrono::milliseconds IPC_QUEUE_FULL_RETRY_DELAY{50};

std::chrono::milliseconds computeAllocationRetryDelay(int failureCount) {
  auto delay =
      ALLOCATION_RETRY_BASE_DELAY + ALLOCATION_RETRY_DELAY_STEP * failureCount;
  return std::min(delay, allocationRetryMaxDelay);
}

int computeFailureCount(int attemptsRemaining) {
  return static_cast<int>(tt::config::sessionAllocationMaxRetries()) -
         attemptsRemaining;
}

domain::ManageMemoryTask makeAllocTask(
    std::optional<uint32_t> slotIdToCopyFrom = std::nullopt) {
  domain::ManageMemoryTask task(tt::utils::TaskIDGenerator::generate(),
                                domain::MemoryManagementAction::ALLOCATE);
  task.slotIdToCopyFrom = slotIdToCopyFrom;
  return task;
}

domain::ManageMemoryTask makeDeallocTask(uint32_t slotId) {
  domain::ManageMemoryTask task(tt::utils::TaskIDGenerator::generate(),
                                domain::MemoryManagementAction::DEALLOCATE);
  task.memoryLayout = domain::KvMemoryLayout::PAGED;
  task.slotId = slotId;
  return task;
}
}  // namespace

SessionManager::SessionManager() {
  try {
    memoryRequestQueue = std::make_unique<ipc::boost::MemoryRequestQueue>(
        tt::config::ttMemoryRequestQueueName(),
        tt::config::memoryQueueCapacity());
    memoryResultQueue = std::make_unique<ipc::boost::MemoryResultQueue>(
        tt::config::ttMemoryResultQueueName(),
        tt::config::memoryQueueCapacity());
    TT_LOG_INFO("[SessionManager] Created memory management IPC queues");
    drainThread = std::thread([this] { readerLoop(); });
  } catch (const std::exception& e) {
    TT_LOG_WARN(
        "[SessionManager] Failed to create memory queues: {}. Slot allocation "
        "will not be available.",
        e.what());
    memoryRequestQueue.reset();
    memoryResultQueue.reset();
  }
}

SessionManager::~SessionManager() {
  stopped.store(true, std::memory_order_relaxed);
  if (drainThread.joinable()) {
    drainThread.join();
  }
}

void SessionManager::readerLoop() {
  while (!stopped.load(std::memory_order_relaxed)) {
    retryFailedAllocations();
    retryFailedDeallocs();
    domain::ManageMemoryResult result;
    bool anyResults = false;
    while (memoryResultQueue->tryPop(result)) {
      anyResults = true;
      TT_LOG_DEBUG(
          "[SessionManager] readerLoop popped result: taskId={}, status={}, "
          "slotId={}",
          result.taskId, static_cast<int>(result.status), result.slotId);
      handleMemoryResult(result);
    }
    if (!anyResults) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void SessionManager::finalizeSessionClose(const std::string& sessionId,
                                          const domain::Session& session) {
  if (session.getSlotId() != tt::domain::INVALID_SLOT_ID) {
    sendDeallocRequest(sessionId, session.getSlotId());
  }
  TT_LOG_INFO("[SessionManager] Closed session: {}", sessionId);
  updateSessionCountMetric();
}

CloseSessionResult SessionManager::closeSession(const std::string& sessionId) {
  TT_LOG_DEBUG("[SessionManager] closeSession called for sessionId={}",
               sessionId);

  auto session = sessions.take(sessionId);
  if (!session.has_value()) {
    TT_LOG_WARN("[SessionManager] Session not found: {}", sessionId);
    return CloseSessionResult::NOT_FOUND;
  }

  // Remove this session from the prefix index so future lookups miss.
  removeFromPrefixIndex(sessionId, session->getHash());

  auto cancelFn = session->takeCancelFn();
  if (cancelFn) {
    cancelFn();
    TT_LOG_INFO("[SessionManager] Cancelled in-flight request for session: {}",
                sessionId);
  }

  finalizeSessionClose(sessionId, *session);
  return CloseSessionResult::SUCCESS;
}

bool SessionManager::assignSlotId(const std::string& sessionId,
                                  uint32_t slotId) {
  bool found = sessions.modify(
      sessionId, [slotId](domain::Session& s) { s.setSlotId(slotId); });

  if (!found) {
    TT_LOG_WARN("[SessionManager] Session not found for slot assignment: {}",
                sessionId);
  } else {
    TT_LOG_INFO("[SessionManager] Assigned slot {} to session {}", slotId,
                sessionId);
  }

  return found;
}

uint32_t SessionManager::getSlotIdBySessionId(
    const std::string& sessionId) const {
  uint32_t result = tt::domain::INVALID_SLOT_ID;
  sessions.modify(sessionId, [&result](domain::Session& s) {
    s.updateActivityTime();
    result = s.getSlotId();
  });
  TT_LOG_DEBUG(
      "[SessionManager] getSlotIdBySessionId sessionId={} -> slotId={}",
      sessionId, result);
  return result;
}

uint32_t SessionManager::acquireInFlight(const std::string& sessionId,
                                         std::function<void()> cancelFn) {
  uint32_t result = tt::domain::INVALID_SLOT_ID;
  bool wasInFlight = false;

  bool found = sessions.modify(
      sessionId, [&result, &wasInFlight,
                  cancelFn = std::move(cancelFn)](domain::Session& s) mutable {
        wasInFlight = s.isInFlight();
        if (wasInFlight) return;
        s.updateActivityTime();
        s.markInFlight();
        s.setCancelFn(std::move(cancelFn));
        result = s.getSlotId();
      });

  if (!found) {
    TT_LOG_WARN("[SessionManager] acquireSessionSlot: sessionId={} not found",
                sessionId);
    return tt::domain::INVALID_SLOT_ID;
  }

  if (wasInFlight) {
    TT_LOG_WARN(
        "[SessionManager] acquireInFlight: sessionId={} already has a "
        "request in flight",
        sessionId);
    throw SessionInFlightException();
  }

  TT_LOG_DEBUG("[SessionManager] acquireInFlight sessionId={} -> slotId={}",
               sessionId, result);
  return result;
}

domain::Session* SessionManager::getSession(const std::string& sessionId) {
  return sessions.getPtr(sessionId);
}

size_t SessionManager::getActiveSessionCount() const { return sessions.size(); }

void SessionManager::lockSlot(uint32_t slotId) {
  std::lock_guard<std::mutex> lock(lockedSlotsMutex);
  lockedSlots.insert(slotId);
  TT_LOG_DEBUG("[SessionManager] lockSlot: slotId={}", slotId);
}

void SessionManager::unlockSlot(uint32_t slotId) {
  std::lock_guard<std::mutex> lock(lockedSlotsMutex);
  lockedSlots.erase(slotId);
  TT_LOG_DEBUG("[SessionManager] unlockSlot: slotId={}", slotId);
}

void SessionManager::evictOldSessions() {
  bool expected = false;
  if (!evictionInProgress.compare_exchange_strong(expected, true)) {
    return;
  }

  struct EvictionGuard {
    std::atomic<bool>& flag;
    ~EvictionGuard() { flag.store(false, std::memory_order_release); }
  } guard{evictionInProgress};

  size_t maxSessions = tt::config::maxSessionsCount();
  unsigned evictionRate = tt::config::sessionEvictionRate();
  size_t evictionCount = tt::config::sessionEvictionCount();

  size_t activeCount = getActiveSessionCount();
  TT_LOG_DEBUG(
      "[SessionManager] evictOldSessions: active={}, max={}, "
      "evictionRate={}%, evictionCount={}",
      activeCount, maxSessions, evictionRate, evictionCount);
  if (activeCount * 100 <= maxSessions * evictionRate) {
    return;
  }

  using Entry = std::pair<std::chrono::system_clock::time_point, std::string>;
  std::vector<Entry> candidates;

  // Snapshot locked slots for the duration of candidate selection.
  std::unordered_set<uint32_t> lockedSnapshot;
  {
    std::lock_guard<std::mutex> lock(lockedSlotsMutex);
    lockedSnapshot = lockedSlots;
  }

  sessions.forEach([&candidates, &lockedSnapshot](const std::string& id,
                                                  const domain::Session& s) {
    if (s.isIdle() &&
        lockedSnapshot.find(s.getSlotId()) == lockedSnapshot.end())
      candidates.emplace_back(s.getLastActivityTime(), id);
  });

  size_t n = std::min(evictionCount, candidates.size());
  std::nth_element(
      candidates.begin(), candidates.begin() + n, candidates.end(),
      [](const Entry& a, const Entry& b) { return a.first < b.first; });
  candidates.resize(n);

  TT_LOG_DEBUG("[SessionManager] evictOldSessions: {} candidates for eviction",
               candidates.size());
  size_t evicted = 0;
  for (const auto& [_, sessionId] : candidates) {
    // A concurrent acquireInFlight or lockSlot call may mark the session
    // busy/locked between the forEach above and here; takeIf checks
    // atomically under the map's entry lock.
    auto ms = sessions.takeIf(sessionId, [&](const domain::Session& s) {
      if (!s.isIdle()) return false;
      std::lock_guard<std::mutex> lk(lockedSlotsMutex);
      return lockedSlots.find(s.getSlotId()) == lockedSlots.end();
    });
    if (!ms.has_value()) {
      TT_LOG_DEBUG(
          "[SessionManager] evictOldSessions: sessionId={} no longer idle, "
          "skipping",
          sessionId);
      continue;
    }
    TT_LOG_DEBUG(
        "[SessionManager] evictOldSessions: evicting sessionId={}, slotId={}",
        sessionId, ms->getSlotId());
    removeFromPrefixIndex(sessionId, ms->getHash());
    finalizeSessionClose(sessionId, *ms);
    ++evicted;
  }

  if (evicted > 0) {
    TT_LOG_INFO(
        "[SessionManager] Evicted {} oldest session(s) (active: {}/{}, "
        "threshold: {}%)",
        evicted, activeCount, maxSessions, evictionRate);
  }
}

void SessionManager::sendDeallocRequest(const std::string& sessionId,
                                        uint32_t slotId) {
  if (!memoryRequestQueue) {
    return;
  }

  auto task = makeDeallocTask(slotId);
  TT_LOG_DEBUG(
      "[SessionManager] sendDeallocRequest: sessionId={}, slotId={}, "
      "taskId={}",
      sessionId, slotId, task.taskId);

  if (!memoryRequestQueue->tryPush(task)) {
    TT_LOG_WARN(
        "[SessionManager] Dealloc queue full, deferring session {} slot {}",
        sessionId, slotId);
    deferredDeallocQueue.push({sessionId, slotId});
  }
}

void SessionManager::createSession(
    std::function<void(const tt::domain::Session&)> onCompletion,
    std::function<void(std::string_view errorMessage)> onError,
    trantor::EventLoop* callerEventLoop,
    std::vector<utils::BlockHashInfo> initialBlockInfos,
    std::optional<uint32_t> slotId, std::optional<uint32_t> slotIdToCopyFrom) {
  TT_LOG_DEBUG(
      "[SessionManager] createSession called, slotId={}, activeSessions={}",
      slotId.has_value() ? std::to_string(slotId.value()) : "none",
      getActiveSessionCount());
  evictOldSessions();

  const uint64_t keyHash =
      initialBlockInfos.empty() ? 0 : initialBlockInfos.front().hash;

  // Fast path: caller supplied a pre-assigned slot. Skip IPC allocation and
  // insert the session synchronously.
  if (slotId.has_value()) {
    domain::Session session(slotId.value(), keyHash);
    sessions.insert(session.getSessionId(), session);
    if (!initialBlockInfos.empty()) {
      registerPrefixHash(session.getSessionId(), initialBlockInfos);
    }
    TT_LOG_INFO("[SessionManager] Created session with pre-assigned slot: {}",
                slotId.value());
    updateSessionCountMetric();
    callerEventLoop->queueInLoop([onCompletion = std::move(onCompletion),
                                  session]() { onCompletion(session); });
    return;
  }

  if (!memoryRequestQueue || !memoryResultQueue) {
    callerEventLoop->queueInLoop([onError = std::move(onError)]() {
      onError("Memory management IPC not available");
    });
    return;
  }

  PendingAllocation pendingAllocation{
      .session = domain::Session(tt::domain::INVALID_SLOT_ID, keyHash),
      .onCompletion = std::move(onCompletion),
      .onError = std::move(onError),
      .eventLoop = callerEventLoop,
      .attemptsRemaining =
          static_cast<int>(tt::config::sessionAllocationMaxRetries()),
      .slotIdToCopyFrom = slotIdToCopyFrom,
  };

  sendAsyncAllocationRequest(pendingAllocation);
}

void SessionManager::sendAsyncAllocationRequest(
    PendingAllocation& pendingAllocation) {
  // Check if max session count is reached
  size_t maxSessions = tt::config::maxSessionsCount();
  size_t activeCount = getActiveSessionCount();

  if (activeCount >= maxSessions) {
    TT_LOG_DEBUG(
        "[SessionManager] sendAsyncAllocationRequest: max sessions reached "
        "({}/{}), deferring sessionId={}",
        activeCount, maxSessions, pendingAllocation.session.getSessionId());

    if (pendingAllocation.attemptsRemaining == 0) {
      TT_LOG_ERROR(
          "[SessionManager] sendAsyncAllocationRequest: no attempts left, "
          "failing sessionId={}",
          pendingAllocation.session.getSessionId());
      pendingAllocation.eventLoop->queueInLoop([onError =
                                                    std::move(pendingAllocation
                                                                  .onError)]() {
        onError(
            "Failed to allocate: max session count reached after all attempts");
      });
    } else {
      pendingAllocation.attemptsRemaining--;
      pendingAllocation.retryAt =
          std::chrono::steady_clock::now() + IPC_QUEUE_FULL_RETRY_DELAY;
      TT_LOG_DEBUG(
          "[SessionManager] sendAsyncAllocationRequest: queuing retry for "
          "sessionId={}, attemptsRemaining={}, delayMs={}",
          pendingAllocation.session.getSessionId(),
          pendingAllocation.attemptsRemaining,
          IPC_QUEUE_FULL_RETRY_DELAY.count());
      pendingAllocationsRetryQueue.push(std::move(pendingAllocation));
    }
    return;
  }

  auto task = makeAllocTask(pendingAllocation.slotIdToCopyFrom);
  TT_LOG_DEBUG(
      "[SessionManager] sendAsyncAllocationRequest: taskId={}, "
      "sessionId={}, attemptsRemaining={}",
      task.taskId, pendingAllocation.session.getSessionId(),
      pendingAllocation.attemptsRemaining);
  pendingAllocationsMap.insert(task.taskId, std::move(pendingAllocation));
  if (!memoryRequestQueue->tryPush(task)) {
    TT_LOG_DEBUG(
        "[SessionManager] sendAsyncAllocationRequest: IPC queue full for "
        "taskId={}",
        task.taskId);
    auto taken = pendingAllocationsMap.take(task.taskId);
    if (!taken.has_value()) return;
    auto& pa = *taken;
    if (pa.attemptsRemaining == 0) {
      TT_LOG_ERROR(
          "[SessionManager] sendAsyncAllocationRequest: no attempts left, "
          "failing sessionId={}",
          pa.session.getSessionId());
      pa.eventLoop->queueInLoop([onError = std::move(pa.onError)]() {
        onError("Failed to allocate: IPC queue full after all attempts");
      });
    } else {
      pa.attemptsRemaining--;
      pa.retryAt =
          std::chrono::steady_clock::now() + IPC_QUEUE_FULL_RETRY_DELAY;
      TT_LOG_DEBUG(
          "[SessionManager] sendAsyncAllocationRequest: queuing retry for "
          "sessionId={}, attemptsRemaining={}, delayMs={}",
          pa.session.getSessionId(), pa.attemptsRemaining,
          IPC_QUEUE_FULL_RETRY_DELAY.count());
      pendingAllocationsRetryQueue.push(std::move(pa));
    }
  }
}

void SessionManager::retryFailedAllocations() {
  auto pendingAllocations = pendingAllocationsRetryQueue.drain();
  if (pendingAllocations.empty()) {
    return;
  }
  TT_LOG_DEBUG("[SessionManager] retryFailedAllocations: {} pending retries",
               pendingAllocations.size());
  evictOldSessions();
  auto now = std::chrono::steady_clock::now();
  for (auto& pendingAllocation : pendingAllocations) {
    if (now >= pendingAllocation.retryAt) {
      TT_LOG_DEBUG(
          "[SessionManager] retryFailedAllocations: retrying sessionId={}, "
          "attemptsRemaining={}",
          pendingAllocation.session.getSessionId(),
          pendingAllocation.attemptsRemaining);
      sendAsyncAllocationRequest(pendingAllocation);
    } else {
      pendingAllocationsRetryQueue.push(std::move(pendingAllocation));
    }
  }
}

void SessionManager::handleMemoryResult(
    const domain::ManageMemoryResult& result) {
  TT_LOG_DEBUG(
      "[SessionManager] handleMemoryResult: taskId={}, status={}, "
      "slotId={}",
      result.taskId, static_cast<int>(result.status), result.slotId);
  auto allocation = pendingAllocationsMap.take(result.taskId);
  if (!allocation.has_value()) {
    TT_LOG_WARN("[SessionManager] No pending allocation found for task ID: {}",
                result.taskId);
    return;
  }
  auto& pendingAllocation = allocation.value();
  bool success = result.status == domain::ManageMemoryStatus::SUCCESS &&
                 result.slotId != tt::domain::INVALID_SLOT_ID;
  if (success) {
    pendingAllocation.session.setSlotId(result.slotId);
    pendingAllocation.session.markPrepared();
    sessions.insert(pendingAllocation.session.getSessionId(),
                    pendingAllocation.session);
    TT_LOG_DEBUG(
        "[SessionManager] handleMemoryResult: SUCCESS sessionId={}, hash={}, "
        "assigned slotId={}",
        pendingAllocation.session.getSessionId(),
        pendingAllocation.session.getHash(), result.slotId);
    updateSessionCountMetric();
    pendingAllocation.eventLoop->queueInLoop(
        [onCompletion = std::move(pendingAllocation.onCompletion),
         session = pendingAllocation.session]() { onCompletion(session); });
  } else if (pendingAllocation.attemptsRemaining > 0) {
    int failureCount = computeFailureCount(pendingAllocation.attemptsRemaining);
    pendingAllocation.attemptsRemaining--;
    auto delay = computeAllocationRetryDelay(failureCount);
    pendingAllocation.retryAt = std::chrono::steady_clock::now() + delay;
    TT_LOG_DEBUG(
        "[SessionManager] handleMemoryResult: FAILURE for sessionId={}, "
        "retrying in {}ms, attemptsRemaining={}",
        pendingAllocation.session.getSessionId(), delay.count(),
        pendingAllocation.attemptsRemaining);
    pendingAllocationsRetryQueue.push(std::move(pendingAllocation));
  } else {
    TT_LOG_ERROR(
        "[SessionManager] Async: failed to allocate slot for "
        "session {} after all attempts",
        pendingAllocation.session.getSessionId());
    pendingAllocation.eventLoop->queueInLoop(
        [onError = std::move(pendingAllocation.onError)]() {
          onError("Failed to allocate slot id: All attempts have failed");
        });
  }
}

void SessionManager::retryFailedDeallocs() {
  for (auto& d : deferredDeallocQueue.drain()) {
    TT_LOG_DEBUG(
        "[SessionManager] retryFailedDeallocs: sessionId={}, slotId={}",
        d.sessionId, d.slotId);
    sendDeallocRequest(d.sessionId, d.slotId);
  }
}

std::optional<SessionManager::AcquiredSession>
SessionManager::tryAcquireByPrefixHash(
    const std::vector<utils::BlockHashInfo>& blockInfos,
    std::function<void()> cancelFn) {
  TT_LOG_DEBUG("[SessionManager] tryAcquireByPrefixHash: blockInfos={}",
               blockInfos.size());

  if (blockInfos.empty()) {
    return std::nullopt;
  }

  const uint64_t keyHash = blockInfos.front().hash;
  const size_t firstBlockTokens = tt::config::kvCacheFirstBlockSize();
  const size_t blockTokens = tt::config::kvCacheBlockSize();

  // Build the caller's remaining block info for comparison (blockInfos[1:]).
  std::list<RemainingBlockInfo> callerRemaining;
  for (size_t i = 1; i < blockInfos.size(); ++i) {
    callerRemaining.push_back(
        {blockInfos[i].hash, blockInfos[i].accumulatedThinkTokens});
  }

  // Snapshot candidates: for each entry under keyHash, count how many
  // consecutive remaining hashes match the caller's remaining hashes.
  // Pick the entry with the longest match.
  std::vector<Candidate> candidates;

  prefixIndex.modify(keyHash, [&](std::vector<PrefixIndexEntry>& entries) {
    for (const auto& entry : entries) {
      // Count consecutive matching remaining hashes.
      size_t matched = 0;
      uint32_t lastMatchedThinkCount = entry.keyBlockThinkTokens;
      auto callerIt = callerRemaining.begin();
      auto entryIt = entry.remainingBlocks.begin();
      while (callerIt != callerRemaining.end() &&
             entryIt != entry.remainingBlocks.end() &&
             callerIt->hash == entryIt->hash) {
        lastMatchedThinkCount = entryIt->accumulatedThinkTokens;
        ++matched;
        ++callerIt;
        ++entryIt;
      }
      // key hash itself counts as 1 block match.
      size_t totalMatched = 1 + matched;
      size_t sessionTotal = 1 + entry.remainingBlocks.size();
      for (const auto& sid : entry.sessionIds) {
        candidates.push_back(
            {sid, totalMatched, sessionTotal, lastMatchedThinkCount});
      }
    }
  });

  if (candidates.empty()) {
    TT_LOG_DEBUG("[SessionManager] tryAcquireByPrefixHash: keyHash={} miss",
                 keyHash);
    return std::nullopt;
  }

  // Sort by matchedBlocks descending so we try longest match first.
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.matchedBlocks > b.matchedBlocks;
            });

  TT_LOG_INFO(
      "[SessionManager] tryAcquireByPrefixHash: {} candidate(s) under "
      "keyHash={}, best match={} blocks",
      candidates.size(), keyHash, candidates.front().matchedBlocks);

  const float threshold = tt::config::prefixCacheHitThreshold();
  bool anyBusy = false;
  for (const auto& candidate : candidates) {
    // Check if match percentage meets threshold (skip if below).
    if (threshold > 0.0f) {
      float matchPercent =
          (candidate.matchedBlocks * 100.0f) / candidate.sessionBlocks;
      if (matchPercent < threshold) {
        TT_LOG_INFO(
            "[SessionManager] Prefix cache candidate rejected: "
            "matchedBlocks={} sessionBlocks={} matchPercent={:.1f}% < "
            "threshold={:.1f}%",
            candidate.matchedBlocks, candidate.sessionBlocks, matchPercent,
            threshold);
        continue;
      }
    }

    std::optional<AcquiredSession> acquired;
    bool busy = false;
    bool stale = false;

    // Compute matched tokens: first block + (matchedBlocks-1) subsequent.
    uint32_t matchedTokens = static_cast<uint32_t>(
        firstBlockTokens + (candidate.matchedBlocks - 1) * blockTokens);

    bool found = sessions.modify(candidate.sessionId, [&](domain::Session& s) {
      if (s.getHash() != keyHash) {
        stale = true;
        return;
      }
      if (s.isInFlight()) {
        busy = true;
        return;
      }
      s.updateActivityTime();
      s.markInFlight();
      s.setCancelFn(cancelFn);
      acquired =
          AcquiredSession{true,          candidate.sessionId,   s.getSlotId(),
                          matchedTokens, candidate.thinkTokens, {}};
    });

    if (!found || stale) {
      removeFromPrefixIndex(candidate.sessionId, keyHash);
      continue;
    }

    if (acquired) {
      TT_LOG_INFO(
          "[SessionManager] tryAcquireByPrefixHash: acquired sessionId={}, "
          "slotId={}, matchedTokens={}, thinkTokens={}, matchedBlocks={}",
          acquired->sessionId, acquired->slotId,
          acquired->numberOfMatchedTokens, acquired->accumulatedThinkTokens,
          candidate.matchedBlocks);
      return acquired;
    }

    anyBusy |= busy;
  }

  if (anyBusy) {
    TT_LOG_INFO(
        "[SessionManager] tryAcquireByPrefixHash: all candidate sessions "
        "are in-flight → falling through to allocate new session");
  }

  TT_LOG_DEBUG(
      "[SessionManager] tryAcquireByPrefixHash: no acquirable session for "
      "keyHash={}",
      keyHash);
  // Return candidates sorted by matched tokens descending even though no
  // session was acquired
  return AcquiredSession{false, {}, 0, 0, 0, std::move(candidates)};
}

void SessionManager::registerPrefixHash(
    const std::string& sessionId,
    const std::vector<utils::BlockHashInfo>& blockInfos) {
  if (blockInfos.empty()) return;

  const uint64_t keyHash = blockInfos.front().hash;
  const uint32_t keyThinkCount = blockInfos.front().accumulatedThinkTokens;
  TT_LOG_DEBUG(
      "[SessionManager] registerPrefixHash: sessionId={}, keyHash={}, "
      "blocks={}, keyThinkCount={}",
      sessionId, keyHash, blockInfos.size(), keyThinkCount);

  // Update session's hash field (stores the key for staleness checks).
  uint64_t oldHash = 0;
  bool sessionFound =
      sessions.modify(sessionId, [&oldHash, keyHash](domain::Session& s) {
        oldHash = s.getHash();
        s.setHash(keyHash);
      });

  if (!sessionFound) {
    TT_LOG_WARN("[SessionManager] registerPrefixHash: sessionId={} not found",
                sessionId);
    return;
  }

  if (oldHash != 0 && oldHash != keyHash) {
    removeFromPrefixIndex(sessionId, oldHash);
  }

  // Build remaining blocks (blockInfos[1:]).
  std::list<RemainingBlockInfo> remaining;
  for (size_t i = 1; i < blockInfos.size(); ++i) {
    remaining.push_back(
        {blockInfos[i].hash, blockInfos[i].accumulatedThinkTokens});
  }

  // Helper to compare remaining block lists by hash only (for dedup).
  auto remainingHashesMatch = [](const std::list<RemainingBlockInfo>& a,
                                 const std::list<RemainingBlockInfo>& b) {
    if (a.size() != b.size()) return false;
    auto itA = a.begin();
    auto itB = b.begin();
    while (itA != a.end()) {
      if (itA->hash != itB->hash) return false;
      ++itA;
      ++itB;
    }
    return true;
  };

  // Insert into prefixIndex: key=keyHash, entry has remaining + sessionId.
  // First, remove sessionId from any existing entry under this key (a session
  // should only appear once — always at its latest registration).
  bool exists = prefixIndex.modify(
      keyHash, [&sessionId, &remaining, &remainingHashesMatch,
                keyThinkCount](std::vector<PrefixIndexEntry>& entries) {
        // Remove sessionId from all entries (it may be in a stale one).
        for (auto it = entries.begin(); it != entries.end();) {
          it->sessionIds.remove(sessionId);
          if (it->sessionIds.empty()) {
            it = entries.erase(it);
          } else {
            ++it;
          }
        }
        // Now add to the matching entry or create a new one.
        for (auto& entry : entries) {
          if (remainingHashesMatch(entry.remainingBlocks, remaining)) {
            entry.sessionIds.push_back(sessionId);
            return;
          }
        }
        entries.push_back(
            PrefixIndexEntry{{sessionId}, remaining, keyThinkCount});
      });
  if (!exists) {
    std::vector<PrefixIndexEntry> entries;
    entries.push_back(PrefixIndexEntry{{sessionId}, remaining, keyThinkCount});
    prefixIndex.insert(keyHash, std::move(entries));
  }

  TT_LOG_INFO(
      "[SessionManager] registerPrefixHash: registered sessionId={} under "
      "keyHash={} with {} remaining blocks",
      sessionId, keyHash, remaining.size());
}

void SessionManager::updateSessionCountMetric() {
  tt::metrics::ServerMetrics::instance().setActiveSessionsCount(
      static_cast<double>(getActiveSessionCount()));
}

void SessionManager::addToPrefixIndex(const std::string& sessionId,
                                      uint64_t prefixHash) {
  if (prefixHash == 0) return;
  bool exists = prefixIndex.modify(
      prefixHash, [&sessionId](std::vector<PrefixIndexEntry>& entries) {
        if (entries.empty()) {
          entries.push_back(PrefixIndexEntry{{sessionId}, {}, 0});
        } else {
          entries.front().sessionIds.push_back(sessionId);
        }
      });
  if (!exists) {
    std::vector<PrefixIndexEntry> entries;
    entries.push_back(PrefixIndexEntry{{sessionId}, {}, 0});
    prefixIndex.insert(prefixHash, std::move(entries));
  }
}

void SessionManager::removeFromPrefixIndex(const std::string& sessionId,
                                           uint64_t prefixHash) {
  if (prefixHash == 0) return;
  bool becameEmpty = false;
  prefixIndex.modify(prefixHash, [&sessionId, &becameEmpty](
                                     std::vector<PrefixIndexEntry>& entries) {
    for (auto& entry : entries) {
      auto& ids = entry.sessionIds;
      ids.erase(std::remove(ids.begin(), ids.end(), sessionId), ids.end());
    }
    // Remove entries with no sessions left
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [](const PrefixIndexEntry& e) {
                                   return e.sessionIds.empty();
                                 }),
                  entries.end());
    becameEmpty = entries.empty();
  });
  if (becameEmpty) {
    prefixIndex.erase(prefixHash);
  }
}

}  // namespace tt::services
