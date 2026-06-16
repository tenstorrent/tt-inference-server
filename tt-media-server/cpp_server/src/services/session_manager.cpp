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

void SessionManager::finalizeSessionClose(uint32_t slotId,
                                          const domain::Session& session) {
  if (session.getSlotId() != tt::domain::INVALID_SLOT_ID) {
    sendDeallocRequest(session.getSlotId());
    tt::metrics::ServerMetrics::instance().removeSlot(session.getSlotId());
  }
  TT_LOG_INFO("[SessionManager] Closed session with slotId: {}", slotId);
  updateSessionCountMetric();
}

CloseSessionResult SessionManager::closeSession(uint32_t slotId) {
  return closeSession(slotKey(slotId));
}

CloseSessionResult SessionManager::closeSession(const std::string& slotIdStr) {
  TT_LOG_DEBUG("[SessionManager] closeSession called for slotId={}", slotIdStr);

  auto sessionOpt = sessions.take(slotIdStr);
  if (!sessionOpt.has_value()) {
    TT_LOG_WARN("[SessionManager] Session not found for slotId: {}", slotIdStr);
    return CloseSessionResult::NOT_FOUND;
  }

  auto& session = *sessionOpt;
  uint32_t slotId = session->getSlotId();

  // Remove this session from the prefix + response-id indexes so future
  // lookups miss.
  removeFromPrefixIndex(slotId, session->getHash());
  removeFromResponseIdIndex(slotId, session->getResponseId());

  auto cancelFn = session->takeCancelFn();
  if (cancelFn) {
    cancelFn();
    TT_LOG_INFO(
        "[SessionManager] Cancelled in-flight request for session slotId: {}",
        slotId);
  }

  finalizeSessionClose(slotId, *session);
  return CloseSessionResult::SUCCESS;
}

bool SessionManager::acquireInFlight(uint32_t slotId,
                                     std::function<void()> cancelFn) {
  bool wasInFlight = false;

  bool found = sessions.modify(slotKey(slotId),
                               [&wasInFlight, cancelFn = std::move(cancelFn)](
                                   std::shared_ptr<domain::Session>& s) mutable {
                                 wasInFlight = s->isInFlight();
                                 if (wasInFlight) return;
                                 s->updateActivityTime();
                                 s->markInFlight();
                                 s->setCancelFn(std::move(cancelFn));
                               });

  if (!found) {
    TT_LOG_WARN("[SessionManager] acquireInFlight: slotId={} not found",
                slotId);
    return false;
  }

  if (wasInFlight) {
    TT_LOG_WARN(
        "[SessionManager] acquireInFlight: slotId={} already has a "
        "request in flight",
        slotId);
    throw SessionInFlightException();
  }

  TT_LOG_DEBUG("[SessionManager] acquireInFlight slotId={}", slotId);
  return true;
}

std::shared_ptr<domain::Session> SessionManager::getSession(uint32_t slotId) {
  std::shared_ptr<domain::Session> result;
  sessions.modify(slotKey(slotId),
                  [&result](std::shared_ptr<domain::Session>& s) { result = s; });
  return result;
}

void SessionManager::releaseInFlight(uint32_t slotId) {
  sessions.modify(slotKey(slotId),
                  [](std::shared_ptr<domain::Session>& s) { s->clearInFlight(); });
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

  sessions.forEach([&candidates, &lockedSnapshot](const std::string& key,
                                                  const std::shared_ptr<domain::Session>& s) {
    if (s->isIdle() &&
        lockedSnapshot.find(s->getSlotId()) == lockedSnapshot.end())
      candidates.emplace_back(s->getLastActivityTime(), key);
  });

  size_t n = std::min(evictionCount, candidates.size());
  std::nth_element(
      candidates.begin(), candidates.begin() + n, candidates.end(),
      [](const Entry& a, const Entry& b) { return a.first < b.first; });
  candidates.resize(n);

  TT_LOG_DEBUG("[SessionManager] evictOldSessions: {} candidates for eviction",
               candidates.size());
  size_t evicted = 0;
  for (const auto& [_, slotIdStr] : candidates) {
    // A concurrent acquireInFlight or lockSlot call may mark the session
    // busy/locked between the forEach above and here; takeIf checks
    // atomically under the map's entry lock.
    auto ms = sessions.takeIf(slotIdStr, [&](const std::shared_ptr<domain::Session>& s) {
      if (!s->isIdle()) return false;
      std::lock_guard<std::mutex> lk(lockedSlotsMutex);
      return lockedSlots.find(s->getSlotId()) == lockedSlots.end();
    });
    if (!ms.has_value()) {
      TT_LOG_DEBUG(
          "[SessionManager] evictOldSessions: slotId={} no longer idle, "
          "skipping",
          slotIdStr);
      continue;
    }
    uint32_t slotId = (*ms)->getSlotId();
    TT_LOG_DEBUG("[SessionManager] evictOldSessions: evicting slotId={}",
                 slotId);
    removeFromPrefixIndex(slotId, (*ms)->getHash());
    removeFromResponseIdIndex(slotId, (*ms)->getResponseId());
    finalizeSessionClose(slotId, **ms);
    ++evicted;
  }

  if (evicted > 0) {
    TT_LOG_INFO(
        "[SessionManager] Evicted {} oldest session(s) (active: {}/{}, "
        "threshold: {}%)",
        evicted, activeCount, maxSessions, evictionRate);
  }
}

void SessionManager::sendDeallocRequest(uint32_t slotId) {
  if (!memoryRequestQueue) {
    return;
  }

  auto task = makeDeallocTask(slotId);
  TT_LOG_DEBUG("[SessionManager] sendDeallocRequest: slotId={}, taskId={}",
               slotId, task.taskId);

  if (!memoryRequestQueue->tryPush(task)) {
    TT_LOG_WARN("[SessionManager] Dealloc queue full, deferring slotId={}",
                slotId);
    deferredDeallocQueue.push({slotId});
  }
}

domain::Session SessionManager::createSession(
    uint32_t slotId, std::vector<utils::BlockHashInfo> initialBlockInfos) {
  TT_LOG_DEBUG(
      "[SessionManager] createSession called, slotId={}, activeSessions={}",
      slotId, getActiveSessionCount());
  evictOldSessions();

  const uint64_t keyHash =
      initialBlockInfos.empty() ? 0 : initialBlockInfos.front().hash;

  auto session = std::make_shared<domain::Session>(slotId, keyHash);
  sessions.insert(slotKey(slotId), session);
  if (!initialBlockInfos.empty()) {
    registerPrefixHash(slotId, initialBlockInfos);
  }
  TT_LOG_INFO("[SessionManager] Created session with slotId: {}", slotId);
  updateSessionCountMetric();
  return *session;
}

void SessionManager::createSessionAsync(
    std::function<void(const domain::Session&)> onCompletion,
    std::function<void(std::string_view errorMessage)> onError,
    trantor::EventLoop* callerEventLoop,
    std::vector<utils::BlockHashInfo> initialBlockInfos,
    std::optional<uint32_t> slotIdToCopyFrom) {
  TT_LOG_DEBUG("[SessionManager] createSessionAsync called, activeSessions={}",
               getActiveSessionCount());
  evictOldSessions();

  if (!memoryRequestQueue || !memoryResultQueue) {
    callerEventLoop->queueInLoop([onError = std::move(onError)]() {
      onError("Memory management IPC not available");
    });
    return;
  }

  auto allocation = std::make_shared<PendingAllocation>();
  allocation->initialBlockInfos = std::move(initialBlockInfos);
  allocation->slotIdToCopyFrom = slotIdToCopyFrom;
  allocation->attemptsRemaining =
      static_cast<int>(tt::config::sessionAllocationMaxRetries());
  allocation->onCompletion = std::move(onCompletion);
  allocation->onError = std::move(onError);
  allocation->eventLoop = callerEventLoop;

  auto taskId = tt::utils::TaskIDGenerator::generate();
  sendAsyncAllocationRequest(taskId, allocation);
}

std::optional<domain::Session> SessionManager::createSessionSync(
    std::vector<utils::BlockHashInfo> initialBlockInfos,
    std::optional<uint32_t> slotIdToCopyFrom, std::chrono::milliseconds timeout,
    std::string* errorMsg) {
  TT_LOG_DEBUG("[SessionManager] createSessionSync called, activeSessions={}",
               getActiveSessionCount());
  evictOldSessions();

  if (!memoryRequestQueue || !memoryResultQueue) {
    if (errorMsg) *errorMsg = "Memory management IPC not available";
    return std::nullopt;
  }

  auto allocation = std::make_shared<PendingAllocation>();
  allocation->initialBlockInfos = std::move(initialBlockInfos);
  allocation->slotIdToCopyFrom = slotIdToCopyFrom;
  allocation->attemptsRemaining =
      static_cast<int>(tt::config::sessionAllocationMaxRetries());

  auto taskId = tt::utils::TaskIDGenerator::generate();
  sendAsyncAllocationRequest(taskId, allocation);

  // Wait for result with timeout
  std::unique_lock<std::mutex> lock(allocation->mutex);
  bool completed = allocation->cv.wait_for(
      lock, timeout, [&allocation]() { return allocation->completed; });

  if (!completed) {
    // Timeout - remove from pending map if still there
    pendingAllocationsMap.erase(taskId);
    if (errorMsg) *errorMsg = "Allocation timeout";
    TT_LOG_WARN("[SessionManager] createSessionSync: timeout after {}ms",
                timeout.count());
    return std::nullopt;
  }

  if (!allocation->resultSlotId.has_value()) {
    if (errorMsg) *errorMsg = allocation->errorMessage;
    return std::nullopt;
  }

  // Create and insert the session
  uint32_t slotId = *allocation->resultSlotId;
  const uint64_t keyHash = allocation->initialBlockInfos.empty()
                               ? 0
                               : allocation->initialBlockInfos.front().hash;

  auto session = std::make_shared<domain::Session>(slotId, keyHash);
  session->markPrepared();
  sessions.insert(slotKey(slotId), session);

  if (!allocation->initialBlockInfos.empty()) {
    registerPrefixHash(slotId, allocation->initialBlockInfos);
  }

  TT_LOG_INFO("[SessionManager] createSessionSync: created session slotId={}",
              slotId);
  updateSessionCountMetric();
  return *session;
}

void SessionManager::sendAsyncAllocationRequest(
    uint32_t taskId, std::shared_ptr<PendingAllocation> allocation) {
  // Helper to signal failure
  auto signalFailure = [](std::shared_ptr<PendingAllocation>& pa,
                          const std::string& msg) {
    if (pa->isAsync()) {
      pa->eventLoop->queueInLoop(
          [onError = std::move(pa->onError), msg]() { onError(msg); });
    } else {
      std::lock_guard<std::mutex> lock(pa->mutex);
      pa->errorMessage = msg;
      pa->completed = true;
      pa->cv.notify_all();
    }
  };

  // Check if max session count is reached
  size_t maxSessions = tt::config::maxSessionsCount();
  size_t activeCount = getActiveSessionCount();

  if (activeCount >= maxSessions) {
    TT_LOG_DEBUG(
        "[SessionManager] sendAsyncAllocationRequest: max sessions reached "
        "({}/{}), deferring taskId={}",
        activeCount, maxSessions, taskId);

    if (allocation->attemptsRemaining == 0) {
      TT_LOG_ERROR(
          "[SessionManager] sendAsyncAllocationRequest: no attempts left, "
          "failing taskId={}",
          taskId);
      signalFailure(
          allocation,
          "Failed to allocate: max session count reached after all attempts");
    } else {
      allocation->attemptsRemaining--;
      allocation->retryAt =
          std::chrono::steady_clock::now() + IPC_QUEUE_FULL_RETRY_DELAY;
      TT_LOG_DEBUG(
          "[SessionManager] sendAsyncAllocationRequest: queuing retry for "
          "taskId={}, attemptsRemaining={}, delayMs={}",
          taskId, allocation->attemptsRemaining,
          IPC_QUEUE_FULL_RETRY_DELAY.count());
      pendingAllocationsRetryQueue.push({taskId, allocation});
    }
    return;
  }

  auto task = makeAllocTask(allocation->slotIdToCopyFrom);
  task.taskId = taskId;
  TT_LOG_DEBUG(
      "[SessionManager] sendAsyncAllocationRequest: taskId={}, "
      "attemptsRemaining={}",
      task.taskId, allocation->attemptsRemaining);
  pendingAllocationsMap.insert(task.taskId, allocation);
  if (!memoryRequestQueue->tryPush(task)) {
    TT_LOG_DEBUG(
        "[SessionManager] sendAsyncAllocationRequest: IPC queue full for "
        "taskId={}",
        task.taskId);
    auto taken = pendingAllocationsMap.take(task.taskId);
    if (!taken.has_value()) return;
    auto& pa = *taken;
    if (pa->attemptsRemaining == 0) {
      TT_LOG_ERROR(
          "[SessionManager] sendAsyncAllocationRequest: no attempts left, "
          "failing taskId={}",
          taskId);
      signalFailure(pa,
                    "Failed to allocate: IPC queue full after all attempts");
    } else {
      pa->attemptsRemaining--;
      pa->retryAt =
          std::chrono::steady_clock::now() + IPC_QUEUE_FULL_RETRY_DELAY;
      TT_LOG_DEBUG(
          "[SessionManager] sendAsyncAllocationRequest: queuing retry for "
          "taskId={}, attemptsRemaining={}, delayMs={}",
          taskId, pa->attemptsRemaining, IPC_QUEUE_FULL_RETRY_DELAY.count());
      pendingAllocationsRetryQueue.push({taskId, pa});
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
  for (auto& [taskId, allocation] : pendingAllocations) {
    if (now >= allocation->retryAt) {
      TT_LOG_DEBUG(
          "[SessionManager] retryFailedAllocations: retrying taskId={}, "
          "attemptsRemaining={}",
          taskId, allocation->attemptsRemaining);
      sendAsyncAllocationRequest(taskId, allocation);
    } else {
      pendingAllocationsRetryQueue.push({taskId, allocation});
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
  auto& pa = *allocation;
  bool success = result.status == domain::ManageMemoryStatus::SUCCESS &&
                 result.slotId != tt::domain::INVALID_SLOT_ID;
  if (success) {
    TT_LOG_DEBUG(
        "[SessionManager] handleMemoryResult: SUCCESS taskId={}, "
        "assigned slotId={}",
        result.taskId, result.slotId);

    if (pa->isAsync()) {
      // Async mode: create session and invoke callback on event loop
      const uint64_t keyHash = pa->initialBlockInfos.empty()
                                   ? 0
                                   : pa->initialBlockInfos.front().hash;
      auto session = std::make_shared<domain::Session>(result.slotId, keyHash);
      session->markPrepared();
      sessions.insert(slotKey(result.slotId), session);

      if (!pa->initialBlockInfos.empty()) {
        registerPrefixHash(result.slotId, pa->initialBlockInfos);
      }

      TT_LOG_INFO(
          "[SessionManager] handleMemoryResult: created session slotId={}",
          result.slotId);
      updateSessionCountMetric();

      pa->eventLoop->queueInLoop([onCompletion = std::move(pa->onCompletion),
                                  session]() { onCompletion(*session); });
    } else {
      // Sync mode: signal the waiting thread
      std::lock_guard<std::mutex> lock(pa->mutex);
      pa->resultSlotId = result.slotId;
      pa->completed = true;
      pa->cv.notify_all();
    }
  } else if (pa->attemptsRemaining > 0) {
    int failureCount = computeFailureCount(pa->attemptsRemaining);
    pa->attemptsRemaining--;
    auto delay = computeAllocationRetryDelay(failureCount);
    pa->retryAt = std::chrono::steady_clock::now() + delay;
    TT_LOG_DEBUG(
        "[SessionManager] handleMemoryResult: FAILURE for taskId={}, "
        "retrying in {}ms, attemptsRemaining={}",
        result.taskId, delay.count(), pa->attemptsRemaining);
    pendingAllocationsRetryQueue.push({result.taskId, pa});
  } else {
    TT_LOG_ERROR(
        "[SessionManager] Async: failed to allocate slot for "
        "taskId={} after all attempts",
        result.taskId);

    if (pa->isAsync()) {
      pa->eventLoop->queueInLoop([onError = std::move(pa->onError)]() {
        onError("Failed to allocate slot id: All attempts have failed");
      });
    } else {
      std::lock_guard<std::mutex> lock(pa->mutex);
      pa->errorMessage = "Failed to allocate slot id: All attempts have failed";
      pa->completed = true;
      pa->cv.notify_all();
    }
  }
}

void SessionManager::retryFailedDeallocs() {
  for (auto& d : deferredDeallocQueue.drain()) {
    TT_LOG_DEBUG("[SessionManager] retryFailedDeallocs: slotId={}", d.slotId);
    sendDeallocRequest(d.slotId);
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
      for (const auto& slotId : entry.slotIds) {
        candidates.push_back(
            {slotId, totalMatched, sessionTotal, lastMatchedThinkCount});
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

    bool found =
        sessions.modify(slotKey(candidate.slotId), [&](std::shared_ptr<domain::Session>& s) {
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
              true, candidate.slotId, matchedTokens, candidate.thinkTokens, {}};
        });

    if (!found || stale) {
      removeFromPrefixIndex(candidate.slotId, keyHash);
      continue;
    }

    if (acquired) {
      TT_LOG_INFO(
          "[SessionManager] tryAcquireByPrefixHash: acquired slotId={}, "
          "matchedTokens={}, thinkTokens={}, matchedBlocks={}",
          acquired->slotId, acquired->numberOfMatchedTokens,
          acquired->accumulatedThinkTokens, candidate.matchedBlocks);
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
  return AcquiredSession{false, 0, 0, 0, std::move(candidates)};
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
          "[SessionManager] findASlotToCopyFrom: candidate slotId={} "
          "matchedBlocks={} matchedTokens={} >= minTokensToCopy={}",
          candidate.slotId, candidate.matchedBlocks, matchedTokens, minTokens);
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
    uint32_t slotId, const std::vector<utils::BlockHashInfo>& blockInfos) {
  if (blockInfos.empty()) return;

  const uint64_t keyHash = blockInfos.front().hash;
  const uint32_t keyThinkCount = blockInfos.front().accumulatedThinkTokens;
  TT_LOG_DEBUG(
      "[SessionManager] registerPrefixHash: slotId={}, keyHash={}, "
      "blocks={}, keyThinkCount={}",
      slotId, keyHash, blockInfos.size(), keyThinkCount);

  // Update session's hash field (stores the key for staleness checks).
  uint64_t oldHash = 0;
  bool sessionFound =
      sessions.modify(slotKey(slotId), [&oldHash, keyHash](std::shared_ptr<domain::Session>& s) {
        oldHash = s->getHash();
        s->setHash(keyHash);
      });

  if (!sessionFound) {
    TT_LOG_WARN("[SessionManager] registerPrefixHash: slotId={} not found",
                slotId);
    return;
  }

  if (oldHash != 0 && oldHash != keyHash) {
    removeFromPrefixIndex(slotId, oldHash);
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

  // Insert into prefixIndex: key=keyHash, entry has remaining + slotId.
  // First, remove slotId from any existing entry under this key (a session
  // should only appear once — always at its latest registration).
  bool exists = prefixIndex.modify(
      keyHash, [slotId, &remaining, &remainingHashesMatch,
                keyThinkCount](std::vector<PrefixIndexEntry>& entries) {
        // Remove slotId from all entries (it may be in a stale one).
        for (auto it = entries.begin(); it != entries.end();) {
          it->slotIds.remove(slotId);
          if (it->slotIds.empty()) {
            it = entries.erase(it);
          } else {
            ++it;
          }
        }
        // Now add to the matching entry or create a new one.
        for (auto& entry : entries) {
          if (remainingHashesMatch(entry.remainingBlocks, remaining)) {
            entry.slotIds.push_back(slotId);
            return;
          }
        }
        entries.push_back(PrefixIndexEntry{{slotId}, remaining, keyThinkCount});
      });
  if (!exists) {
    std::vector<PrefixIndexEntry> entries;
    entries.push_back(PrefixIndexEntry{{slotId}, remaining, keyThinkCount});
    prefixIndex.insert(keyHash, std::move(entries));
  }

  TT_LOG_INFO(
      "[SessionManager] registerPrefixHash: registered slotId={} under "
      "keyHash={} with {} remaining blocks",
      slotId, keyHash, remaining.size());

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

  // Read the single entry under the responseIdIndex lock, then release it
  // before touching the sessions map (sessions.modify takes that lock, and we
  // avoid holding both simultaneously).
  uint32_t slotId = tt::domain::INVALID_SLOT_ID;
  bool present = responseIdIndex.modify(
      previousResponseId, [&slotId](uint32_t& sid) { slotId = sid; });

  if (!present) {
    TT_LOG_INFO(
        "[SessionManager] tryAcquireByResponseId: id={} MISS "
        "(not found in responseIdIndex)",
        previousResponseId);
    return std::nullopt;
  }

  std::optional<AcquiredSession> acquired;
  bool busy = false;
  bool stale = false;

  bool found = sessions.modify(slotKey(slotId), [&](std::shared_ptr<domain::Session>& s) {
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
    a.slotId = s->getSlotId();
    acquired = a;
  });

  // The index pointed at a session that's gone or has since been re-keyed to a
  // different id: prune the stale entry and report a miss.
  if (!found || stale) {
    removeFromResponseIdIndex(slotId, previousResponseId);
    return std::nullopt;
  }

  if (acquired) {
    TT_LOG_INFO(
        "[SessionManager] tryAcquireByResponseId: acquired slotId={} for id={}",
        acquired->slotId, previousResponseId);
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

void SessionManager::initResponseId(uint32_t slotId,
                                    const std::string& responseId) {
  if (responseId.empty()) {
    return;
  }
  TT_LOG_INFO("[SessionManager] initResponseId: slotId={}, id={}", slotId,
              responseId);

  sessions.modify(slotKey(slotId), [&responseId](std::shared_ptr<domain::Session>& s) {
    s->setResponseId(responseId);
  });

  bool existed = responseIdIndex.modify(
      responseId, [slotId](uint32_t& sid) { sid = slotId; });
  if (!existed) {
    responseIdIndex.insert(responseId, slotId);
  }
}

void SessionManager::registerResponseId(const std::string& previousResponseId,
                                        const std::string& responseId) {
  if (previousResponseId.empty() || responseId.empty()) {
    return;
  }
  if (previousResponseId == responseId) {
    return;
  }

  uint32_t slotId = tt::domain::INVALID_SLOT_ID;
  bool found = responseIdIndex.modify(
      previousResponseId, [&slotId](uint32_t& sid) { slotId = sid; });

  if (!found) {
    TT_LOG_WARN(
        "[SessionManager] registerResponseId: previousId={} not in index",
        previousResponseId);
    return;
  }

  TT_LOG_INFO(
      "[SessionManager] registerResponseId: re-keying slotId={} from "
      "id={} to id={}",
      slotId, previousResponseId, responseId);

  responseIdIndex.erase(previousResponseId);

  bool existed = responseIdIndex.modify(
      responseId, [slotId](uint32_t& sid) { sid = slotId; });
  if (!existed) {
    responseIdIndex.insert(responseId, slotId);
  }

  sessions.modify(slotKey(slotId), [&responseId](std::shared_ptr<domain::Session>& s) {
    s->setResponseId(responseId);
  });
}

std::pair<uint32_t, uint32_t> SessionManager::computeMatchedTokens(
    uint32_t slotId, const std::vector<utils::BlockHashInfo>& blockInfos) {
  if (blockInfos.empty()) {
    return {0, 0};
  }

  const uint64_t keyHash = blockInfos.front().hash;
  const size_t firstBlockTokens = tt::config::kvCacheFirstBlockSize();
  const size_t blockTokens = tt::config::kvCacheBlockSize();

  std::list<RemainingBlockInfo> callerRemaining;
  for (size_t i = 1; i < blockInfos.size(); ++i) {
    callerRemaining.push_back(
        {blockInfos[i].hash, blockInfos[i].accumulatedThinkTokens});
  }

  size_t matchedBlocks = 0;
  uint32_t thinkTokens = 0;

  prefixIndex.modify(keyHash, [&](std::vector<PrefixIndexEntry>& entries) {
    for (const auto& entry : entries) {
      bool hasSession = std::find(entry.slotIds.begin(), entry.slotIds.end(),
                                  slotId) != entry.slotIds.end();
      if (!hasSession) continue;

      size_t matched = 0;
      uint32_t lastThink = entry.keyBlockThinkTokens;
      auto callerIt = callerRemaining.begin();
      auto entryIt = entry.remainingBlocks.begin();
      while (callerIt != callerRemaining.end() &&
             entryIt != entry.remainingBlocks.end() &&
             callerIt->hash == entryIt->hash) {
        lastThink = entryIt->accumulatedThinkTokens;
        ++matched;
        ++callerIt;
        ++entryIt;
      }
      size_t total = 1 + matched;
      if (total > matchedBlocks) {
        matchedBlocks = total;
        thinkTokens = lastThink;
      }
    }
  });

  if (matchedBlocks == 0) {
    return {0, 0};
  }
  uint32_t matchedTokens = static_cast<uint32_t>(
      firstBlockTokens + (matchedBlocks - 1) * blockTokens);
  return {matchedTokens, thinkTokens};
}

void SessionManager::clearSessionBlockThinkTokens(uint32_t slotId) {
  uint64_t keyHash = 0;
  bool found = sessions.modify(slotKey(slotId), [&keyHash](std::shared_ptr<domain::Session>& s) {
    keyHash = s->getHash();
  });

  if (!found) {
    TT_LOG_WARN(
        "[SessionManager] clearSessionBlockThinkTokens: slotId={} not found",
        slotId);
    return;
  }

  if (keyHash == 0) {
    return;
  }

  prefixIndex.modify(keyHash, [slotId](std::vector<PrefixIndexEntry>& entries) {
    for (auto& entry : entries) {
      bool hasSession = std::find(entry.slotIds.begin(), entry.slotIds.end(),
                                  slotId) != entry.slotIds.end();
      if (!hasSession) continue;
      entry.keyBlockThinkTokens = 0;
      for (auto& block : entry.remainingBlocks) {
        block.accumulatedThinkTokens = 0;
      }
    }
  });

  TT_LOG_INFO(
      "[SessionManager] clearSessionBlockThinkTokens: reset think tokens for "
      "slotId={}",
      slotId);
}

void SessionManager::updateSessionCountMetric() {
  tt::metrics::ServerMetrics::instance().setActiveSessionsCount(
      static_cast<double>(getActiveSessionCount()));
}

void SessionManager::addToPrefixIndex(uint32_t slotId, uint64_t prefixHash) {
  if (prefixHash == 0) return;
  bool exists = prefixIndex.modify(
      prefixHash, [slotId](std::vector<PrefixIndexEntry>& entries) {
        if (entries.empty()) {
          entries.push_back(PrefixIndexEntry{{slotId}, {}, 0});
        } else {
          entries.front().slotIds.push_back(slotId);
        }
      });
  if (!exists) {
    std::vector<PrefixIndexEntry> entries;
    entries.push_back(PrefixIndexEntry{{slotId}, {}, 0});
    prefixIndex.insert(prefixHash, std::move(entries));
  }
}

void SessionManager::removeFromPrefixIndex(uint32_t slotId,
                                           uint64_t prefixHash) {
  if (prefixHash == 0) return;
  bool becameEmpty = false;
  prefixIndex.modify(prefixHash, [slotId, &becameEmpty](
                                     std::vector<PrefixIndexEntry>& entries) {
    for (auto& entry : entries) {
      auto& ids = entry.slotIds;
      ids.erase(std::remove(ids.begin(), ids.end(), slotId), ids.end());
    }
    // Remove entries with no sessions left
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [](const PrefixIndexEntry& e) {
                                   return e.slotIds.empty();
                                 }),
                  entries.end());
    becameEmpty = entries.empty();
  });
  if (becameEmpty) {
    prefixIndex.erase(prefixHash);
  }
}

void SessionManager::removeFromResponseIdIndex(uint32_t slotId,
                                               const std::string& responseId) {
  if (responseId.empty()) return;
  bool matches = false;
  bool found = responseIdIndex.modify(
      responseId,
      [slotId, &matches](uint32_t& sid) { matches = (sid == slotId); });
  if (found && matches) {
    responseIdIndex.erase(responseId);
  }
}

}  // namespace tt::services
