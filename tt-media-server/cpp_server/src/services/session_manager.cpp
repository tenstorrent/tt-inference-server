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

domain::ManageMemoryTask makeAllocTask() {
  return domain::ManageMemoryTask(tt::utils::TaskIDGenerator::generate(),
                                  domain::MemoryManagementAction::ALLOCATE);
}

domain::ManageMemoryTask makeDeallocTask(uint32_t slotId) {
  domain::ManageMemoryTask task(tt::utils::TaskIDGenerator::generate(),
                                domain::MemoryManagementAction::DEALLOCATE);
  task.memoryLayout = domain::KvMemoryLayout::Paged;
  task.slotIds = {slotId};
  return task;
}
}  // namespace

SessionManager::SessionManager() {
  try {
    memoryRequestQueue = std::make_unique<ipc::MemoryRequestQueue>(
        tt::config::ttMemoryRequestQueueName(), ipc::MEMORY_QUEUE_CAPACITY);
    memoryResultQueue = std::make_unique<ipc::MemoryResultQueue>(
        tt::config::ttMemoryResultQueueName(), ipc::MEMORY_QUEUE_CAPACITY);
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
          "slotIds.size={}",
          result.taskId, static_cast<int>(result.status),
          result.slotIds.size());
      handleMemoryResult(result);
    }
    if (!anyResults) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void SessionManager::finalizeSessionClose(const std::string& sessionId,
                                          const domain::Session& session) {
  if (session.getSlotId() != domain::INVALID_SLOT_ID) {
    sendDeallocRequest(sessionId, session.getSlotId());
  }
  TT_LOG_INFO("[SessionManager] Closed session: {}", sessionId);
  updateSessionCountMetric();
}

CloseSessionResult SessionManager::closeSession(const std::string& sessionId) {
  TT_LOG_DEBUG("[SessionManager] closeSession called for sessionId={}",
               sessionId);

  auto ms = sessions.take(sessionId);
  if (!ms.has_value()) {
    TT_LOG_WARN("[SessionManager] Session not found: {}", sessionId);
    return CloseSessionResult::NOT_FOUND;
  }

  // Remove this session from the prefix index so future lookups miss.
  removeFromPrefixIndex(sessionId, ms->session.getHash());

  if (ms->cancelFn) {
    ms->cancelFn();
    TT_LOG_INFO("[SessionManager] Cancelled in-flight request for session: {}",
                sessionId);
  }

  finalizeSessionClose(sessionId, ms->session);
  return CloseSessionResult::SUCCESS;
}

bool SessionManager::assignSlotId(const std::string& sessionId,
                                  uint32_t slotId) {
  bool found = sessions.modify(sessionId, [slotId](ManagedSession& ms) {
    ms.session.setSlotId(slotId);
  });

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
  uint32_t result = domain::INVALID_SLOT_ID;
  sessions.modify(sessionId, [&result](ManagedSession& ms) {
    ms.session.updateActivityTime();
    result = ms.session.getSlotId();
  });
  TT_LOG_DEBUG(
      "[SessionManager] getSlotIdBySessionId sessionId={} -> slotId={}",
      sessionId, result);
  return result;
}

uint32_t SessionManager::acquireInFlight(const std::string& sessionId,
                                         std::function<void()> cancelFn) {
  uint32_t result = domain::INVALID_SLOT_ID;
  bool wasInFlight = false;

  bool found = sessions.modify(
      sessionId, [&result, &wasInFlight,
                  cancelFn = std::move(cancelFn)](ManagedSession& ms) mutable {
        wasInFlight = ms.session.isInFlight();
        if (wasInFlight) return;
        ms.session.updateActivityTime();
        ms.session.markInFlight();
        ms.cancelFn = std::move(cancelFn);
        result = ms.session.getSlotId();
      });

  if (!found) {
    TT_LOG_WARN("[SessionManager] acquireSessionSlot: sessionId={} not found",
                sessionId);
    return domain::INVALID_SLOT_ID;
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

std::optional<domain::Session> SessionManager::getSession(
    const std::string& sessionId) const {
  auto ms = sessions.get(sessionId);
  if (!ms.has_value()) return std::nullopt;
  return ms->session;
}

size_t SessionManager::getActiveSessionCount() const { return sessions.size(); }

void SessionManager::releaseInFlight(const std::string& sessionId) {
  bool found = sessions.modify(sessionId, [](ManagedSession& ms) {
    ms.cancelFn = nullptr;
    if (!ms.session.clearInFlight()) {
      TT_LOG_WARN("[Session] clearInFlight: unexpected state {}",
                  static_cast<int>(ms.session.getState()));
    }
  });

  if (!found) {
    TT_LOG_DEBUG(
        "[SessionManager] releaseInFlight: session {} already removed "
        "(closed concurrently), ignoring",
        sessionId);
    return;
  }

  TT_LOG_DEBUG("[SessionManager] Released in-flight for session {}", sessionId);
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

  sessions.forEach(
      [&candidates](const std::string& id, const ManagedSession& ms) {
        if (ms.session.isIdle())
          candidates.emplace_back(ms.session.getLastActivityTime(), id);
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
    // A concurrent acquireInFlight call may mark the session in-flight
    // between the forEach above and here; takeIf skips it atomically.
    auto ms = sessions.takeIf(sessionId, [](const ManagedSession& ms) {
      return ms.session.isIdle();
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
        sessionId, ms->session.getSlotId());
    removeFromPrefixIndex(sessionId, ms->session.getHash());
    finalizeSessionClose(sessionId, ms->session);
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
    trantor::EventLoop* callerEventLoop, size_t initialHash,
    std::optional<uint32_t> slotId) {
  TT_LOG_DEBUG(
      "[SessionManager] createSession called, slotId={}, activeSessions={}",
      slotId.has_value() ? std::to_string(slotId.value()) : "none",
      getActiveSessionCount());
  evictOldSessions();

  // Fast path: caller supplied a pre-assigned slot. Skip IPC allocation and
  // insert the session synchronously.
  if (slotId.has_value()) {
    domain::Session session(slotId.value(), initialHash);
    sessions.insert(session.getSessionId(), ManagedSession{session, {}});
    if (initialHash != 0) {
      addToPrefixIndex(session.getSessionId(), initialHash);
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
      .session = domain::Session(domain::INVALID_SLOT_ID, initialHash),
      .onCompletion = std::move(onCompletion),
      .onError = std::move(onError),
      .eventLoop = callerEventLoop,
      .attemptsRemaining =
          static_cast<int>(tt::config::sessionAllocationMaxRetries()),
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

  auto task = makeAllocTask();
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
      "slotIds.size={}{}",
      result.taskId, static_cast<int>(result.status), result.slotIds.size(),
      result.slotIds.empty()
          ? ""
          : (", slotIds[0]=" + std::to_string(result.slotIds.front())));
  auto allocation = pendingAllocationsMap.take(result.taskId);
  if (!allocation.has_value()) {
    TT_LOG_WARN("[SessionManager] No pending allocation found for task ID: {}",
                result.taskId);
    return;
  }
  auto& pendingAllocation = allocation.value();
  bool success = result.status == domain::ManageMemoryStatus::SUCCESS &&
                 !result.slotIds.empty();
  if (success) {
    pendingAllocation.session.setSlotId(result.slotIds.front());
    pendingAllocation.session.markPrepared();
    sessions.insert(pendingAllocation.session.getSessionId(),
                    ManagedSession{pendingAllocation.session, {}});
    if (pendingAllocation.session.getHash() != 0) {
      addToPrefixIndex(pendingAllocation.session.getSessionId(),
                       pendingAllocation.session.getHash());
    }
    TT_LOG_DEBUG(
        "[SessionManager] handleMemoryResult: SUCCESS sessionId={}, hash={}, "
        "assigned slotId={}",
        pendingAllocation.session.getSessionId(),
        pendingAllocation.session.getHash(), result.slotIds.front());
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
SessionManager::tryAcquireByPrefixHash(uint64_t prefixHash,
                                       std::function<void()> cancelFn) {
  TT_LOG_DEBUG("[SessionManager] tryAcquireByPrefixHash: hash={}", prefixHash);

  // Snapshot candidate sessionIds under the prefixIndex lock, then release it
  // before touching the sessions map (acquireInFlight takes that lock, and we
  // avoid holding both simultaneously).
  std::vector<std::string> candidateIds;
  prefixIndex.modify(prefixHash,
                     [&candidateIds](const std::list<std::string>& ids) {
                       candidateIds.assign(ids.begin(), ids.end());
                     });

  if (candidateIds.empty()) {
    TT_LOG_DEBUG("[SessionManager] tryAcquireByPrefixHash: hash={} miss",
                 prefixHash);
    return std::nullopt;
  }

  TT_LOG_INFO(
      "[SessionManager] tryAcquireByPrefixHash: {} candidate(s) under hash={}",
      candidateIds.size(), prefixHash);

  bool anyBusy = false;
  for (const auto& sessionId : candidateIds) {
    std::optional<AcquiredSession> acquired;
    bool busy = false;
    bool stale = false;

    bool found = sessions.modify(sessionId, [&](ManagedSession& ms) {
      // The session's stored hash is the source of truth; a mismatch means
      // this index entry is stale (e.g. torn update from a concurrent
      // registerPrefixHash). Treat as stale and clean up.
      if (ms.session.getHash() != prefixHash) {
        stale = true;
        return;
      }
      if (ms.session.isInFlight()) {
        busy = true;
        return;
      }
      ms.session.updateActivityTime();
      ms.session.markInFlight();
      ms.cancelFn = cancelFn;  // copied so we can retry other candidates
      acquired = AcquiredSession{sessionId, ms.session.getSlotId()};
    });

    if (!found || stale) {
      removeFromPrefixIndex(sessionId, prefixHash);
      continue;
    }

    if (acquired) {
      TT_LOG_INFO(
          "[SessionManager] tryAcquireByPrefixHash: acquired sessionId={}, "
          "slotId={} for hash={}",
          acquired->sessionId, acquired->slotId, prefixHash);
      return acquired;
    }

    anyBusy |= busy;
  }

  if (anyBusy) {
    TT_LOG_WARN(
        "[SessionManager] tryAcquireByPrefixHash: all sessions under hash={} "
        "are in-flight",
        prefixHash);
    throw SessionInFlightException();
  }

  return std::nullopt;
}

void SessionManager::registerPrefixHash(const std::string& sessionId,
                                        uint64_t prefixHash) {
  TT_LOG_DEBUG("[SessionManager] registerPrefixHash: sessionId={}, hash={}",
               sessionId, prefixHash);

  // Update session's hash field and pick up the old hash (for index update).
  uint64_t oldHash = 0;
  bool sessionFound =
      sessions.modify(sessionId, [&oldHash, prefixHash](ManagedSession& ms) {
        oldHash = ms.session.getHash();
        ms.session.setHash(prefixHash);
      });

  if (!sessionFound) {
    TT_LOG_WARN("[SessionManager] registerPrefixHash: sessionId={} not found",
                sessionId);
    return;
  }

  if (oldHash == prefixHash) {
    return;
  }

  if (oldHash != 0) {
    removeFromPrefixIndex(sessionId, oldHash);
  }
  addToPrefixIndex(sessionId, prefixHash);

  TT_LOG_INFO(
      "[SessionManager] registerPrefixHash: registered sessionId={} under "
      "hash={}",
      sessionId, prefixHash);
}

void SessionManager::updateSessionCountMetric() {
  tt::metrics::ServerMetrics::instance().setActiveSessionsCount(
      static_cast<double>(getActiveSessionCount()));
}

void SessionManager::addToPrefixIndex(const std::string& sessionId,
                                      uint64_t prefixHash) {
  if (prefixHash == 0) return;
  bool exists = prefixIndex.modify(
      prefixHash,
      [&sessionId](std::list<std::string>& ids) { ids.push_back(sessionId); });
  if (!exists) {
    prefixIndex.insert(prefixHash, std::list<std::string>{sessionId});
  }
}

void SessionManager::removeFromPrefixIndex(const std::string& sessionId,
                                           uint64_t prefixHash) {
  if (prefixHash == 0) return;
  bool becameEmpty = false;
  prefixIndex.modify(prefixHash,
                     [&sessionId, &becameEmpty](std::list<std::string>& ids) {
                       ids.remove(sessionId);
                       becameEmpty = ids.empty();
                     });
  if (becameEmpty) {
    prefixIndex.erase(prefixHash);
  }
}

}  // namespace tt::services
