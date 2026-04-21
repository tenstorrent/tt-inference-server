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
  abortCallbacks_.take(sessionId);
  if (session.getSlotId() != domain::INVALID_SLOT_ID) {
    sendDeallocRequest(sessionId, session.getSlotId());
  }
  TT_LOG_INFO("[SessionManager] Closed session: {}", sessionId);
  updateSessionCountMetric();
}

CloseSessionResult SessionManager::closeSession(const std::string& sessionId) {
  TT_LOG_DEBUG("[SessionManager] closeSession called for sessionId={}",
               sessionId);

  // Search for session by UUID across all hash buckets. If it is IDLE, remove
  // it immediately; if it is IN_FLIGHT, mark it for deferred close and we'll
  // fire the abort callback below so the request unwinds. The actual
  // deallocation then happens in releaseInFlight() when the request finishes.
  bool found = false;
  bool closedImmediately = false;
  std::optional<domain::Session> takenSession;
  size_t foundHash = 0;

  sessions.forEach([&](size_t hash, std::list<domain::Session>& sessionList) {
    for (auto it = sessionList.begin(); it != sessionList.end(); ++it) {
      if (it->getSessionId() != sessionId) continue;
      found = true;
      foundHash = hash;

      if (it->isIdle()) {
        takenSession = *it;
        sessionList.erase(it);
        closedImmediately = true;
      } else if (!it->markCloseRequested()) {
        TT_LOG_WARN("[Session] markCloseRequested: unexpected state {}",
                    static_cast<int>(it->getState()));
      }
      return;  // exit forEach early
    }
  });

  if (!found) {
    TT_LOG_WARN("[SessionManager] Session not found: {}", sessionId);
    return CloseSessionResult::NOT_FOUND;
  }

  if (closedImmediately) {
    // Clean up empty bucket to avoid leaking entries under stale hashes.
    sessions.modify(foundHash, [](std::list<domain::Session>&) {});
    auto bucket = sessions.get(foundHash);
    if (bucket.has_value() && bucket->empty()) {
      sessions.erase(foundHash);
    }
    finalizeSessionClose(sessionId, *takenSession);
    return CloseSessionResult::SUCCESS;
  }

  auto abortCallback = abortCallbacks_.take(sessionId);
  if (abortCallback.has_value()) {
    (*abortCallback)();
    TT_LOG_INFO("[SessionManager] Aborted in-flight request for session: {}",
                sessionId);
  } else {
    TT_LOG_WARN(
        "[SessionManager] closeSession: sessionId={} is in-flight but no abort "
        "callback registered; will close when request completes",
        sessionId);
  }

  // The in-flight request may have completed between the forEach above and
  // here; resolve that race by attempting to finalize if the session is now
  // CLOSING.
  std::optional<domain::Session> deferred;
  sessions.modify(foundHash, [&](std::list<domain::Session>& list) {
    for (auto it = list.begin(); it != list.end(); ++it) {
      if (it->getSessionId() == sessionId && it->isClosing()) {
        deferred = *it;
        list.erase(it);
        return;
      }
    }
  });
  if (deferred.has_value()) {
    auto bucket = sessions.get(foundHash);
    if (bucket.has_value() && bucket->empty()) {
      sessions.erase(foundHash);
    }
    finalizeSessionClose(sessionId, *deferred);
  }

  return CloseSessionResult::SUCCESS;
}

bool SessionManager::assignSlotId(const std::string& sessionId,
                                  uint32_t slotId) {
  bool found = false;

  sessions.forEach([&](size_t, std::list<domain::Session>& sessionList) {
    for (auto& session : sessionList) {
      if (session.getSessionId() == sessionId) {
        session.setSlotId(slotId);
        found = true;
        TT_LOG_INFO("[SessionManager] Assigned slot {} to session {}", slotId,
                    sessionId);
        return;
      }
    }
  });

  if (!found) {
    TT_LOG_WARN("[SessionManager] Session not found for slot assignment: {}",
                sessionId);
  }

  return found;
}

uint32_t SessionManager::getSlotIdBySessionId(
    const std::string& sessionId) const {
  uint32_t result = domain::INVALID_SLOT_ID;

  sessions.forEach([&](size_t, std::list<domain::Session>& sessionList) {
    for (auto& session : sessionList) {
      if (session.getSessionId() == sessionId) {
        session.updateActivityTime();
        result = session.getSlotId();
        return;
      }
    }
  });

  TT_LOG_DEBUG(
      "[SessionManager] getSlotIdBySessionId sessionId={} -> slotId={}",
      sessionId, result);
  return result;
}

uint32_t SessionManager::acquireSessionSlot(const std::string& sessionId) {
  uint32_t result = domain::INVALID_SLOT_ID;
  bool wasInFlight = false;
  bool found = false;

  sessions.forEach([&](size_t, std::list<domain::Session>& sessionList) {
    for (auto& session : sessionList) {
      if (session.getSessionId() != sessionId) continue;
      found = true;
      if (!session.isIdle()) {
        wasInFlight = true;
        return;
      }
      session.updateActivityTime();
      if (!session.markInFlight()) {
        TT_LOG_WARN("[Session] markInFlight: unexpected state {}",
                    static_cast<int>(session.getState()));
        return;
      }
      result = session.getSlotId();
      return;
    }
  });

  if (!found) {
    TT_LOG_WARN("[SessionManager] acquireSessionSlot: sessionId={} not found",
                sessionId);
    return domain::INVALID_SLOT_ID;
  }

  if (wasInFlight) {
    TT_LOG_WARN(
        "[SessionManager] acquireSessionSlot: sessionId={} already has a "
        "request in flight",
        sessionId);
    throw SessionInFlightException();
  }

  TT_LOG_DEBUG("[SessionManager] acquireSessionSlot sessionId={} -> slotId={}",
               sessionId, result);
  return result;
}

std::optional<domain::Session> SessionManager::getSession(
    const std::string& sessionId) const {
  std::optional<domain::Session> result;

  sessions.forEach([&](size_t, const std::list<domain::Session>& sessionList) {
    for (const auto& session : sessionList) {
      if (session.getSessionId() == sessionId) {
        result = session;
        return;
      }
    }
  });

  return result;
}

size_t SessionManager::getActiveSessionCount() const {
  // `sessions` is keyed by prefix hash and each bucket may hold multiple
  // sessions, so sum across buckets instead of returning the bucket count.
  size_t count = 0;
  sessions.forEach([&count](size_t, const std::list<domain::Session>& list) {
    count += list.size();
  });
  return count;
}

void SessionManager::releaseInFlight(const std::string& sessionId) {
  bool found = false;
  bool becameClosing = false;
  size_t foundHash = 0;

  sessions.forEach([&](size_t hash, std::list<domain::Session>& sessionList) {
    for (auto& session : sessionList) {
      if (session.getSessionId() != sessionId) continue;
      found = true;
      foundHash = hash;
      if (!session.clearInFlight()) {
        TT_LOG_WARN("[Session] clearInFlight: unexpected state {}",
                    static_cast<int>(session.getState()));
        return;
      }
      becameClosing = session.isClosing();
      return;
    }
  });

  if (!found) {
    TT_LOG_WARN("[SessionManager] Session not found for in-flight update: {}",
                sessionId);
    return;
  }

  TT_LOG_DEBUG("[SessionManager] Released in-flight for session {}", sessionId);

  if (!becameClosing) {
    abortCallbacks_.take(sessionId);
    return;
  }

  // A close was requested while in-flight: remove from bucket and finalize.
  std::optional<domain::Session> takenSession;
  sessions.modify(foundHash, [&](std::list<domain::Session>& list) {
    for (auto it = list.begin(); it != list.end(); ++it) {
      if (it->getSessionId() == sessionId && it->isClosing()) {
        takenSession = *it;
        list.erase(it);
        return;
      }
    }
  });
  if (takenSession.has_value()) {
    auto bucket = sessions.get(foundHash);
    if (bucket.has_value() && bucket->empty()) {
      sessions.erase(foundHash);
    }
    finalizeSessionClose(sessionId, *takenSession);
  }
}

void SessionManager::setSessionAbortCallback(const std::string& sessionId,
                                             std::function<void()> onAbort) {
  abortCallbacks_.insert(sessionId, std::move(onAbort));
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

  // Collect oldest idle sessions across all hash buckets. Each heap entry
  // references a hash bucket; within the bucket we'll pick the oldest idle
  // session when we evict below.
  using Entry = std::pair<std::chrono::system_clock::time_point, size_t>;
  auto newer = [](const Entry& a, const Entry& b) { return a.first < b.first; };
  std::vector<Entry> heap;
  heap.reserve(evictionCount + 1);

  sessions.forEach(
      [&heap, &newer, evictionCount](
          size_t hash, const std::list<domain::Session>& sessionList) {
        for (const auto& session : sessionList) {
          if (!session.isIdle()) continue;

          auto t = session.getLastActivityTime();
          if (heap.size() < evictionCount) {
            heap.emplace_back(t, hash);
            std::push_heap(heap.begin(), heap.end(), newer);
          } else if (t < heap.front().first) {
            std::pop_heap(heap.begin(), heap.end(), newer);
            heap.back() = {t, hash};
            std::push_heap(heap.begin(), heap.end(), newer);
          }
        }
      });

  TT_LOG_DEBUG("[SessionManager] evictOldSessions: {} candidates for eviction",
               heap.size());
  size_t evicted = 0;
  for (const auto& [_, hash] : heap) {
    // A concurrent request may have marked the candidate in-flight between
    // the forEach above and here; removeIf below skips it atomically.
    std::optional<domain::Session> takenSession;
    sessions.modify(hash, [&](std::list<domain::Session>& list) {
      for (auto it = list.begin(); it != list.end(); ++it) {
        if (!it->isIdle()) continue;
        TT_LOG_DEBUG(
            "[SessionManager] evictOldSessions: evicting sessionId={}, "
            "slotId={}",
            it->getSessionId(), it->getSlotId());
        takenSession = *it;
        list.erase(it);
        return;
      }
    });

    if (!takenSession.has_value()) {
      TT_LOG_DEBUG(
          "[SessionManager] evictOldSessions: no idle session left in hash={}, "
          "skipping",
          hash);
      continue;
    }

    auto bucket = sessions.get(hash);
    if (bucket.has_value() && bucket->empty()) {
      sessions.erase(hash);
    }

    finalizeSessionClose(takenSession->getSessionId(), *takenSession);
    ++evicted;
  }

  if (evicted > 0) {
    TT_LOG_INFO(
        "[SessionManager] Evicted {} oldest session(s) (active: {}/{}, "
        "threshold: {}%)",
        evicted, activeCount, maxSessions, evictionRate);
    updateSessionCountMetric();
  }
}

void SessionManager::sendDeallocRequest(const std::string& sessionId,
                                        uint32_t slotId) {
  if (!memoryRequestQueue) {
    return;
  }

  domain::ManageMemoryTask task;
  task.taskId = utils::TaskIDGenerator::generate();
  task.action = domain::MemoryManagementAction::DEALLOCATE;
  task.memoryLayout = domain::KvMemoryLayout::Paged;
  task.slotIds = {slotId};
  TT_LOG_DEBUG(
      "[SessionManager] sendDeallocRequest: sessionId={}, slotId={}, "
      "taskId={}",
      sessionId, slotId, task.taskId);

  if (memoryRequestQueue->tryPush(task)) {
    TT_LOG_DEBUG("[SessionManager] Sent dealloc request for session {} slot {}",
                 sessionId, slotId);
  } else {
    TT_LOG_WARN(
        "[SessionManager] Dealloc queue full, deferring session {} slot {}",
        sessionId, slotId);
    deferredDeallocQueue.push({sessionId, slotId});
  }
}

void SessionManager::createSession(
    std::function<void(const tt::domain::Session&)> onCompletion,
    std::function<void(std::string_view errorMessage)> onError,
    trantor::EventLoop* callerEventLoop, const std::string& requestPrompt,
    size_t initialHash, std::optional<uint32_t> slotId) {
  TT_LOG_DEBUG(
      "[SessionManager] createSession called, slotId={}, activeSessions={}",
      slotId.has_value() ? std::to_string(slotId.value()) : "none",
      getActiveSessionCount());
  evictOldSessions();

  // Fast path: caller supplied a pre-assigned slot. Skip IPC allocation and
  // insert the session synchronously.
  if (slotId.has_value()) {
    domain::Session session(slotId.value(), initialHash);
    size_t hash = session.getHash();
    bool exists =
        sessions.modify(hash, [&](std::list<domain::Session>& sessionList) {
          sessionList.push_back(session);
        });
    if (!exists) {
      sessions.insert(hash, {session});
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

  domain::Session session =
      domain::Session(domain::INVALID_SLOT_ID, initialHash);
  auto pendingAllocation = PendingAllocation(
      std::move(session), std::move(onCompletion), std::move(onError),
      callerEventLoop, tt::config::sessionAllocationMaxRetries());

  sendAsyncAllocationRequest(pendingAllocation);
}

void SessionManager::sendAsyncAllocationRequest(
    PendingAllocation& pendingAllocation) {
  auto task =
      domain::ManageMemoryTask(tt::utils::TaskIDGenerator::generate(),
                               domain::MemoryManagementAction::ALLOCATE);
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
  } else {
    TT_LOG_DEBUG(
        "[SessionManager] sendAsyncAllocationRequest: pushed taskId={} to "
        "IPC queue",
        task.taskId);
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
    if (!pendingAllocation.session.markInFlight()) {
      TT_LOG_WARN("[Session] markInFlight: unexpected state {} for session {}",
                  static_cast<int>(pendingAllocation.session.getState()),
                  pendingAllocation.session.getSessionId());
    }

    // Append to existing hash bucket or create a new one.
    size_t hash = pendingAllocation.session.getHash();
    bool exists =
        sessions.modify(hash, [&](std::list<domain::Session>& sessionList) {
          sessionList.push_back(pendingAllocation.session);
        });

    if (!exists) {
      sessions.insert(hash, {pendingAllocation.session});
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
          onError("Failed to allocate slot id: All attemps have failed");
        });
  }
}

void SessionManager::retryFailedDeallocs() {
  auto deferredDeallocs = deferredDeallocQueue.drain();
  if (!deferredDeallocs.empty()) {
    TT_LOG_DEBUG("[SessionManager] retryFailedDeallocs: {} deferred deallocs",
                 deferredDeallocs.size());
  }
  for (auto& deferredDealloc : deferredDeallocs) {
    TT_LOG_DEBUG(
        "[SessionManager] retryFailedDeallocs: retrying sessionId={}, "
        "slotId={}",
        deferredDealloc.sessionId, deferredDealloc.slotId);
    sendDeallocRequest(deferredDealloc.sessionId, deferredDealloc.slotId);
  }
}

std::optional<SessionManager::AcquiredSession>
SessionManager::tryAcquireByPrefixHash(uint64_t prefixHash) {
  TT_LOG_DEBUG("[SessionManager] tryAcquireByPrefixHash: hash={}", prefixHash);

  std::optional<AcquiredSession> result;
  bool allBusy = false;

  bool hashExists =
      sessions.modify(prefixHash, [&](std::list<domain::Session>& sessionList) {
        TT_LOG_INFO(
            "[SessionManager] tryAcquireByPrefixHash: found {} session(s) "
            "under hash={}",
            sessionList.size(), prefixHash);
        for (const auto& s : sessionList) {
          TT_LOG_INFO("[SessionManager]   - sessionId={}, slotId={}, state={}",
                      s.getSessionId(), s.getSlotId(),
                      static_cast<int>(s.getState()));
        }

        for (auto& session : sessionList) {
          if (!session.isIdle()) continue;
          if (!session.markInFlight()) {
            TT_LOG_WARN("[Session] markInFlight: unexpected state {}",
                        static_cast<int>(session.getState()));
            continue;
          }
          session.updateActivityTime();
          result = AcquiredSession{session.getSessionId(), session.getSlotId()};
          TT_LOG_INFO(
              "[SessionManager] tryAcquireByPrefixHash: acquired "
              "sessionId={}, slotId={} for hash={}",
              result->sessionId, result->slotId, prefixHash);
          return;
        }
        allBusy = !sessionList.empty();
      });

  if (!hashExists) {
    TT_LOG_DEBUG(
        "[SessionManager] tryAcquireByPrefixHash: hash={} not found (miss)",
        prefixHash);
    return std::nullopt;
  }

  if (!result.has_value() && allBusy) {
    TT_LOG_WARN(
        "[SessionManager] tryAcquireByPrefixHash: all sessions under hash={} "
        "are in-flight",
        prefixHash);
    throw SessionInFlightException();
  }

  return result;
}

void SessionManager::registerPrefixHash(const std::string& sessionId,
                                        uint64_t prefixHash) {
  TT_LOG_DEBUG("[SessionManager] registerPrefixHash: sessionId={}, hash={}",
               sessionId, prefixHash);

  // Remove the session from its current bucket.
  domain::Session targetSession;
  size_t oldHash = 0;
  bool sessionFound = false;

  sessions.forEach([&](size_t hash, std::list<domain::Session>& sessionList) {
    for (auto it = sessionList.begin(); it != sessionList.end(); ++it) {
      if (it->getSessionId() == sessionId) {
        targetSession = *it;
        oldHash = hash;
        sessionFound = true;
        sessionList.erase(it);
        TT_LOG_DEBUG(
            "[SessionManager] registerPrefixHash: removed sessionId={} from "
            "old hash={}",
            sessionId, oldHash);
        return;
      }
    }
  });

  if (!sessionFound) {
    TT_LOG_WARN(
        "[SessionManager] registerPrefixHash: sessionId={} not found in any "
        "hash bucket",
        sessionId);
    return;
  }

  // Drop the old bucket if it's now empty.
  auto oldList = sessions.get(oldHash);
  if (oldList.has_value() && oldList->empty()) {
    sessions.erase(oldHash);
    TT_LOG_DEBUG(
        "[SessionManager] registerPrefixHash: erased empty hash bucket {}",
        oldHash);
  }

  targetSession.setHash(prefixHash);

  bool exists =
      sessions.modify(prefixHash, [&](std::list<domain::Session>& sessionList) {
        sessionList.push_back(targetSession);
      });

  if (!exists) {
    std::list<domain::Session> newList;
    newList.push_back(targetSession);
    sessions.insert(prefixHash, std::move(newList));
  }

  TT_LOG_INFO(
      "[SessionManager] registerPrefixHash: registered sessionId={} under "
      "hash={}",
      sessionId, prefixHash);
}

void SessionManager::updateSessionCountMetric() {
  tt::metrics::ServerMetrics::instance().setActiveSessionsCount(
      static_cast<double>(getActiveSessionCount()));
}

}  // namespace tt::services
