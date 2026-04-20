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

CloseSessionResult SessionManager::closeSession(const std::string& sessionId) {
  TT_LOG_DEBUG("[SessionManager] closeSession called for sessionId={}",
               sessionId);

  // Search for session by UUID across all hash buckets
  bool found = false;
  bool inFlight = false;
  size_t foundHash = 0;

  sessions.forEach([&](size_t hash, std::list<domain::Session>& sessionList) {
    for (auto it = sessionList.begin(); it != sessionList.end(); ++it) {
      if (it->getSessionId() == sessionId) {
        found = true;
        foundHash = hash;

        if (it->isInFlight()) {
          // Mark for deferred close
          it->setPendingClose(true);
          inFlight = true;
          TT_LOG_WARN(
              "[SessionManager] closeSession: session={} is in-flight, "
              "will close when request completes",
              sessionId);
        } else {
          // Close immediately
          uint32_t slotId = it->getSlotId();
          if (slotId != domain::INVALID_SLOT_ID) {
            sendDeallocRequest(sessionId, slotId);
          }
          sessionList.erase(it);
          TT_LOG_INFO(
              "[SessionManager] Closed session: sessionId={}, slotId={}",
              sessionId, slotId);
        }
        return;  // Exit forEach early
      }
    }
  });

  if (!found) {
    TT_LOG_WARN("[SessionManager] Session not found: {}", sessionId);
    return CloseSessionResult::NOT_FOUND;
  }

  if (inFlight) {
    return CloseSessionResult::IN_FLIGHT;
  }

  updateSessionCountMetric();
  return CloseSessionResult::SUCCESS;
}

bool SessionManager::assignSlotId(const std::string& sessionId,
                                  uint32_t slotId) {
  // Search for session by UUID
  bool found = false;

  sessions.forEach([&](size_t hash, std::list<domain::Session>& sessionList) {
    for (auto& session : sessionList) {
      if (session.getSessionId() == sessionId) {
        session.setSlotId(slotId);
        found = true;
        TT_LOG_INFO("[SessionManager] Assigned slot {} to session {}", slotId,
                    sessionId);
        return;  // Exit forEach early
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

  sessions.forEach([&](size_t hash, std::list<domain::Session>& sessionList) {
    for (auto& session : sessionList) {
      if (session.getSessionId() == sessionId) {
        session.updateActivityTime();
        result = session.getSlotId();
        return;  // Exit forEach early
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
      if (session.getSessionId() == sessionId) {
        found = true;
        wasInFlight = session.isInFlight();
        if (!wasInFlight) {
          session.updateActivityTime();
          session.setInFlight(true);
          result = session.getSlotId();
        }
        return;  // Exit forEach early
      }
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
  // Search for session by UUID
  std::optional<domain::Session> result;

  sessions.forEach([&](size_t, const std::list<domain::Session>& sessionList) {
    for (const auto& session : sessionList) {
      if (session.getSessionId() == sessionId) {
        result = session;
        return;  // Exit forEach early
      }
    }
  });

  return result;
}

size_t SessionManager::getActiveSessionCount() const { return sessions.size(); }

void SessionManager::setSessionInFlight(const std::string& sessionId,
                                        bool inFlight) {
  // Search for session by UUID across all hash buckets
  bool found = false;
  size_t foundHash = 0;

  sessions.forEach([&](size_t hash, std::list<domain::Session>& sessionList) {
    for (auto& session : sessionList) {
      if (session.getSessionId() == sessionId) {
        session.setInFlight(inFlight);
        found = true;
        foundHash = hash;

        TT_LOG_INFO("[SessionManager] Set session {} in-flight: {} (hash={})",
                    sessionId, inFlight, hash);

        // Check for pending close if marking not in-flight
        if (!inFlight && session.isPendingClose()) {
          uint32_t slotId = session.getSlotId();
          if (slotId != domain::INVALID_SLOT_ID) {
            sendDeallocRequest(sessionId, slotId);
          }
        }
        return;  // Exit forEach early
      }
    }
  });

  if (!found) {
    TT_LOG_WARN("[SessionManager] Session not found for in-flight update: {}",
                sessionId);
    return;
  }

  // Clean up if pending close was handled
  if (!inFlight) {
    bool cleaned = false;
    sessions.modify(
        foundHash, [&sessionId, &cleaned](std::list<domain::Session>& list) {
          auto removed = list.remove_if([&sessionId](const domain::Session& s) {
            return s.getSessionId() == sessionId && s.isPendingClose() &&
                   !s.isInFlight();
          });
          cleaned = (removed > 0);
        });

    if (cleaned) {
      TT_LOG_INFO("[SessionManager] Deferred close executed for session: {}",
                  sessionId);
      updateSessionCountMetric();
    }
  }
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

  size_t activeCount = sessions.size();
  TT_LOG_DEBUG(
      "[SessionManager] evictOldSessions: active={}, max={}, "
      "evictionRate={}%, evictionCount={}",
      activeCount, maxSessions, evictionRate, evictionCount);
  if (activeCount * 100 <= maxSessions * evictionRate) {
    return;
  }

  // Note: With hash-based sessions, eviction is more complex
  // For now, collect oldest sessions across all hash buckets
  using Entry = std::pair<std::chrono::system_clock::time_point, size_t>;
  auto newer = [](const Entry& a, const Entry& b) { return a.first < b.first; };
  std::vector<Entry> heap;
  heap.reserve(evictionCount + 1);

  sessions.forEach(
      [&heap, &newer, evictionCount](
          size_t hash, const std::list<domain::Session>& sessionList) {
        // Check each session in the list
        for (const auto& session : sessionList) {
          if (session.isInFlight()) continue;

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
    // Try to evict oldest non-in-flight session from this hash bucket
    auto sessionList = sessions.get(hash);
    if (!sessionList.has_value() || sessionList->empty()) {
      continue;
    }

    // Find first non-in-flight session
    bool found = false;
    sessions.modify(hash, [&](std::list<domain::Session>& list) {
      for (auto it = list.begin(); it != list.end(); ++it) {
        if (!it->isInFlight()) {
          uint32_t slotId = it->getSlotId();
          std::string sessionId = it->getSessionId();
          TT_LOG_DEBUG(
              "[SessionManager] evictOldSessions: evicting sessionId={}, "
              "slotId={}",
              sessionId, slotId);
          if (slotId != domain::INVALID_SLOT_ID) {
            sendDeallocRequest(sessionId, slotId);
          }
          list.erase(it);
          found = true;
          break;
        }
      }
    });

    if (found) {
      ++evicted;
    }
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
    size_t initialHash) {
  TT_LOG_DEBUG("[SessionManager] createSession called activeSessions={}",
               sessions.size());
  evictOldSessions();

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
    pendingAllocation.session.setInFlight(true);

    // Append to existing list or create new one
    size_t hash = pendingAllocation.session.getHash();
    bool exists =
        sessions.modify(hash, [&](std::list<domain::Session>& sessionList) {
          sessionList.push_back(pendingAllocation.session);
        });

    if (!exists) {
      // Hash doesn't exist yet, create new list
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
  bool allInFlight = false;
  bool found = false;

  bool hashExists =
      sessions.modify(prefixHash, [&](std::list<domain::Session>& sessionList) {
        found = true;

        // Log all sessions in this bucket for debugging
        TT_LOG_INFO(
            "[SessionManager] tryAcquireByPrefixHash: found {} session(s) "
            "under hash={}",
            sessionList.size(), prefixHash);
        for (const auto& s : sessionList) {
          TT_LOG_INFO(
              "[SessionManager]   - sessionId={}, slotId={}, inFlight={}",
              s.getSessionId(), s.getSlotId(), s.isInFlight());
        }

        // Find first non-in-flight session
        for (auto& session : sessionList) {
          if (!session.isInFlight()) {
            // Found an available session, mark it in-flight
            session.setInFlight(true);
            session.updateActivityTime();
            result =
                AcquiredSession{session.getSlotId(), session.getSessionId()};
            TT_LOG_INFO(
                "[SessionManager] tryAcquireByPrefixHash: acquired "
                "sessionId={}, slotId={} "
                "for hash={}",
                result->sessionId, result->slotId, prefixHash);
            return;
          }
        }
        // All sessions in the list are in-flight
        allInFlight = true;
      });

  if (!hashExists) {
    // Hash not found in map
    TT_LOG_DEBUG(
        "[SessionManager] tryAcquireByPrefixHash: hash={} not found (miss)",
        prefixHash);
    return std::nullopt;
  }

  if (allInFlight) {
    // All sessions under this hash are busy
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

  // Find the session by UUID across all hash buckets
  domain::Session targetSession;
  size_t oldHash = 0;
  bool sessionFound = false;

  sessions.forEach([&](size_t hash, std::list<domain::Session>& sessionList) {
    for (auto it = sessionList.begin(); it != sessionList.end(); ++it) {
      if (it->getSessionId() == sessionId) {
        targetSession = *it;
        oldHash = hash;
        sessionFound = true;
        // Remove from old location
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
        "hash "
        "bucket",
        sessionId);
    return;
  }

  // Clean up empty list at old hash if needed
  if (oldHash != 0) {
    sessions.modify(oldHash, [](std::list<domain::Session>& sessionList) {
      // Check if empty and erase outside if needed
    });
    // Actually, we need to erase if empty - let me check if list is empty
    auto oldList = sessions.get(oldHash);
    if (oldList.has_value() && oldList->empty()) {
      sessions.erase(oldHash);
      TT_LOG_DEBUG(
          "[SessionManager] registerPrefixHash: erased empty hash bucket {}",
          oldHash);
    }
  }

  // Update the session's content hash (session ID remains stable)
  targetSession.setHash(prefixHash);

  // Insert into new hash bucket (append to list)
  bool exists =
      sessions.modify(prefixHash, [&](std::list<domain::Session>& sessionList) {
        sessionList.push_back(targetSession);
      });

  if (!exists) {
    // Hash doesn't exist yet, create new list
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
  size_t count = sessions.size();
  tt::metrics::ServerMetrics::instance().setActiveSessionsCount(
      static_cast<double>(count));
}

}  // namespace tt::services
