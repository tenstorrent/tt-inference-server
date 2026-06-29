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
  return domain::ManageMemoryTask{
      .taskId = tt::utils::TaskIDGenerator::generate(),
      .action = domain::MemoryManagementAction::ALLOCATE,
      .slotIdToCopyFrom = slotIdToCopyFrom,
  };
}

domain::ManageMemoryTask makeDeallocTask(uint32_t slotId) {
  domain::ManageMemoryTask task{
      .taskId = tt::utils::TaskIDGenerator::generate(),
      .action = domain::MemoryManagementAction::DEALLOCATE,
      .memoryLayout = domain::KvMemoryLayout::PAGED,
      .slotId = slotId,
      .slotIdToCopyFrom = std::nullopt,
  };
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
    tt::metrics::ServerMetrics::instance().removeSlot(session.getSlotId());
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

  // Remove this session from the prefix + response-id indexes so future
  // lookups miss. (*session is the shared_ptr taken out of the map.)
  auto& sessionPtr = *session;

  prefixIndex.remove(sessionId, sessionPtr->getHash());
  responseIdIndex.removeIf(sessionId, sessionPtr->getResponseId());
  auto cancelFn = sessionPtr->takeCancelFn();
  if (cancelFn) {
    cancelFn();
    TT_LOG_INFO("[SessionManager] Cancelled in-flight request for session: {}",
                sessionId);
  }

  finalizeSessionClose(sessionId, *sessionPtr);
  return CloseSessionResult::SUCCESS;
}

bool SessionManager::assignSlotId(const std::string& sessionId,
                                  uint32_t slotId) {
  bool found = sessions.modify(
      sessionId,
      [slotId](std::shared_ptr<domain::Session>& s) { s->setSlotId(slotId); });

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
  sessions.modify(sessionId, [&result](std::shared_ptr<domain::Session>& s) {
    s->updateActivityTime();
    result = s->getSlotId();
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
      sessionId, [&result, &wasInFlight, cancelFn = std::move(cancelFn)](
                     std::shared_ptr<domain::Session>& s) mutable {
        wasInFlight = s->isInFlight();
        if (wasInFlight) return;
        s->updateActivityTime();
        s->markInFlight();
        s->setCancelFn(std::move(cancelFn));
        result = s->getSlotId();
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

std::shared_ptr<domain::Session> SessionManager::getSession(
    const std::string& sessionId) {
  auto s = sessions.get(sessionId);
  return s.has_value() ? *s : nullptr;
}

void SessionManager::releaseInFlight(const std::string& sessionId) {
  sessions.modify(sessionId, [](std::shared_ptr<domain::Session>& s) {
    s->clearInFlight();
  });
}

void SessionManager::insertSession(const domain::Session& session) {
  // Single place that wraps a Session into the map's shared_ptr and injects the
  // release hook. The hook runs clearInFlight() under the ConcurrentMap lock
  // (race-safe vs evictOldSessions); shared ownership keeps the object alive
  // for any request still holding it even after eviction drops the map entry.
  auto sessionPtr = std::make_shared<domain::Session>(session);
  const std::string sid = sessionPtr->getSessionId();
  sessionPtr->setReleaser([this, sid] { releaseInFlight(sid); });
  sessions.insert(sid, std::move(sessionPtr));
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

  sessions.forEach(
      [&candidates, &lockedSnapshot](
          const std::string& id, const std::shared_ptr<domain::Session>& s) {
        if (s->isIdle() &&
            lockedSnapshot.find(s->getSlotId()) == lockedSnapshot.end())
          candidates.emplace_back(s->getLastActivityTime(), id);
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
    auto ms = sessions.takeIf(
        sessionId, [&](const std::shared_ptr<domain::Session>& s) {
          if (!s->isIdle()) return false;
          std::lock_guard<std::mutex> lk(lockedSlotsMutex);
          return lockedSlots.find(s->getSlotId()) == lockedSlots.end();
        });
    if (!ms.has_value()) {
      TT_LOG_DEBUG(
          "[SessionManager] evictOldSessions: sessionId={} no longer idle, "
          "skipping",
          sessionId);
      continue;
    }
    auto& evictedS = *ms;  // shared_ptr taken out of the map
    TT_LOG_DEBUG(
        "[SessionManager] evictOldSessions: evicting sessionId={}, slotId={}",
        sessionId, evictedS->getSlotId());
    prefixIndex.remove(sessionId, evictedS->getHash());
    responseIdIndex.removeIf(sessionId, evictedS->getResponseId());
    finalizeSessionClose(sessionId, *evictedS);
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
    insertSession(session);
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
    insertSession(pendingAllocation.session);
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

  std::vector<Candidate> candidates = prefixIndex.findCandidates(blockInfos);

  if (candidates.empty()) {
    TT_LOG_DEBUG("[SessionManager] tryAcquireByPrefixHash: keyHash={} miss",
                 keyHash);
    return std::nullopt;
  }

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

    bool found = sessions.modify(
        candidate.sessionId, [&](std::shared_ptr<domain::Session>& s) {
          if (s->getHash() != keyHash) {
            stale = true;
            return;
          }
          if (s->isInFlight()) {
            busy = true;
            return;
          }
          s->updateActivityTime();
          s->markInFlight();
          s->setCancelFn(cancelFn);
          acquired = AcquiredSession{
              true,          candidate.sessionId,   s->getSlotId(),
              matchedTokens, candidate.thinkTokens, {}};
        });

    if (!found || stale) {
      prefixIndex.remove(candidate.sessionId, keyHash);
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

std::optional<SessionManager::Candidate> SessionManager::findASlotToCopyFrom(
    const std::vector<Candidate>& candidates) const {
  const size_t firstBlockSize = tt::config::kvCacheFirstBlockSize();
  const size_t blockSize = tt::config::kvCacheBlockSize();
  const size_t minTokens = tt::config::minTokensToCopy();

  for (const auto& candidate : candidates) {
    if (candidate.matchedBlocks == 0) continue;

    const size_t matchedTokens =
        firstBlockSize + (candidate.matchedBlocks > 1
                              ? (candidate.matchedBlocks - 1) * blockSize
                              : 0);

    if (matchedTokens >= minTokens) {
      TT_LOG_DEBUG(
          "[SessionManager] findASlotToCopyFrom: candidate sessionId={} "
          "matchedBlocks={} matchedTokens={} >= minTokensToCopy={}",
          candidate.sessionId, candidate.matchedBlocks, matchedTokens,
          minTokens);
      return candidate;
    }
  }

  TT_LOG_DEBUG(
      "[SessionManager] findASlotToCopyFrom: no candidate meets threshold "
      "(minTokensToCopy={}, candidates={})",
      minTokens, candidates.size());
  return std::nullopt;
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
  uint32_t slotId = tt::domain::INVALID_SLOT_ID;
  bool sessionFound = sessions.modify(
      sessionId,
      [&oldHash, &slotId, keyHash](std::shared_ptr<domain::Session>& s) {
        oldHash = s->getHash();
        slotId = s->getSlotId();
        s->setHash(keyHash);
      });

  if (!sessionFound) {
    TT_LOG_WARN("[SessionManager] registerPrefixHash: sessionId={} not found",
                sessionId);
    return;
  }

  if (oldHash != 0 && oldHash != keyHash) {
    prefixIndex.remove(sessionId, oldHash);
  }

  prefixIndex.registerPrefixHash(sessionId, blockInfos);

  TT_LOG_INFO(
      "[SessionManager] registerPrefixHash: registered sessionId={} under "
      "keyHash={} with {} remaining blocks",
      sessionId, keyHash, blockInfos.size() - 1);

  // Publish the slot's committed block count (1 key block + remaining).
  if (slotId != tt::domain::INVALID_SLOT_ID) {
    tt::metrics::ServerMetrics::instance().setSlotBlocks(
        slotId, static_cast<double>(blockInfos.size()));
  }
}

std::optional<SessionManager::AcquiredSession>
SessionManager::tryAcquireByResponseId(const std::string& previousResponseId,
                                       std::function<void()> cancelFn) {
  if (previousResponseId.empty()) {
    return std::nullopt;
  }
  TT_LOG_DEBUG("[SessionManager] tryAcquireByResponseId: id={}",
               previousResponseId);

  const auto sessionIdOpt = responseIdIndex.lookup(previousResponseId);
  if (!sessionIdOpt.has_value()) {
    TT_LOG_INFO(
        "[SessionManager] tryAcquireByResponseId: id={} MISS "
        "(not found in responseIdIndex)",
        previousResponseId);
    return std::nullopt;
  }

  const std::string sessionId = *sessionIdOpt;

  std::optional<AcquiredSession> acquired;
  bool busy = false;
  bool stale = false;

  bool found =
      sessions.modify(sessionId, [&](std::shared_ptr<domain::Session>& s) {
        if (s->getResponseId() != previousResponseId) {
          stale = true;
          return;
        }
        if (s->isInFlight()) {
          busy = true;
          return;
        }
        s->updateActivityTime();
        s->markInFlight();
        s->setCancelFn(std::move(cancelFn));
        AcquiredSession a;
        a.sessionId = sessionId;
        a.slotId = s->getSlotId();
        acquired = a;
      });

  // The index pointed at a session that's gone or has since been re-keyed to a
  // different id: prune the stale entry and report a miss.
  if (!found || stale) {
    responseIdIndex.removeIf(sessionId, previousResponseId);
    return std::nullopt;
  }

  if (acquired) {
    TT_LOG_INFO(
        "[SessionManager] tryAcquireByResponseId: acquired sessionId={}, "
        "slotId={} for id={}",
        acquired->sessionId, acquired->slotId, previousResponseId);
    return acquired;
  }

  if (busy) {
    TT_LOG_WARN(
        "[SessionManager] tryAcquireByResponseId: session under id={} is "
        "in-flight",
        previousResponseId);
    throw SessionInFlightException();
  }

  return std::nullopt;
}

void SessionManager::initResponseId(const std::string& sessionId,
                                    const std::string& responseId) {
  if (responseId.empty()) {
    return;
  }
  TT_LOG_INFO("[SessionManager] initResponseId: sessionId={}, id={}", sessionId,
              responseId);

  sessions.modify(sessionId,
                  [&responseId](std::shared_ptr<domain::Session>& s) {
                    s->setResponseId(responseId);
                  });
  responseIdIndex.init(responseId, sessionId);
}

void SessionManager::registerResponseId(const std::string& previousResponseId,
                                        const std::string& responseId) {
  if (previousResponseId.empty() || responseId.empty()) {
    return;
  }
  if (previousResponseId == responseId) {
    return;
  }

  const auto sessionIdOpt =
      responseIdIndex.rekey(previousResponseId, responseId);
  if (!sessionIdOpt.has_value()) {
    TT_LOG_WARN(
        "[SessionManager] registerResponseId: previousId={} not in index",
        previousResponseId);
    return;
  }
  const std::string sessionId = *sessionIdOpt;
  TT_LOG_INFO(
      "[SessionManager] registerResponseId: re-keying sessionId={} from "
      "id={} to id={}",
      sessionId, previousResponseId, responseId);

  sessions.modify(sessionId,
                  [&responseId](std::shared_ptr<domain::Session>& s) {
                    s->setResponseId(responseId);
                  });
}

std::pair<uint32_t, uint32_t> SessionManager::computeMatchedTokens(
    const std::string& sessionId,
    const std::vector<utils::BlockHashInfo>& blockInfos) {
  if (blockInfos.empty()) {
    return {0, 0};
  }

  const size_t firstBlockTokens = tt::config::kvCacheFirstBlockSize();
  const size_t blockTokens = tt::config::kvCacheBlockSize();

  const auto [matchedBlocks, thinkTokens] =
      prefixIndex.computeMatchedBlocks(sessionId, blockInfos);

  if (matchedBlocks == 0) {
    return {0, 0};
  }
  uint32_t matchedTokens = static_cast<uint32_t>(
      firstBlockTokens + (matchedBlocks - 1) * blockTokens);
  return {matchedTokens, thinkTokens};
}

void SessionManager::clearSessionBlockThinkTokens(
    const std::string& sessionId) {
  uint64_t keyHash = 0;
  bool found = sessions.modify(sessionId,
                               [&keyHash](std::shared_ptr<domain::Session>& s) {
                                 keyHash = s->getHash();
                               });

  if (!found) {
    TT_LOG_WARN(
        "[SessionManager] clearSessionBlockThinkTokens: sessionId={} not "
        "found",
        sessionId);
    return;
  }

  if (keyHash == 0) {
    return;
  }

  prefixIndex.clearThinkTokens(sessionId, keyHash);
  TT_LOG_INFO(
      "[SessionManager] clearSessionBlockThinkTokens: reset think tokens for "
      "sessionId={}",
      sessionId);
}

void SessionManager::updateSessionCountMetric() {
  tt::metrics::ServerMetrics::instance().setActiveSessionsCount(
      static_cast<double>(getActiveSessionCount()));
}

}  // namespace tt::services
