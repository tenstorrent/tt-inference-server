// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/session_manager.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

namespace tt::services {

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

bool SessionManager::closeSession(const std::string& sessionId) {
  TT_LOG_DEBUG("[SessionManager] closeSession called for sessionId={}",
               sessionId);
  auto session = sessions.take(sessionId);
  if (!session.has_value()) {
    TT_LOG_WARN("[SessionManager] Session not found: {}", sessionId);
    return false;
  }

  uint32_t slotId = session->getSlotId();
  TT_LOG_DEBUG("[SessionManager] closeSession sessionId={} has slotId={}",
               sessionId, slotId);
  if (slotId != domain::INVALID_SLOT_ID) {
    sendDeallocRequest(sessionId, slotId);
  }

  TT_LOG_INFO("[SessionManager] Closed session: {}", sessionId);
  return true;
}

bool SessionManager::assignSlotId(const std::string& sessionId,
                                  uint32_t slotId) {
  bool found = sessions.modify(
      sessionId, [slotId](domain::Session& s) { s.setSlotId(slotId); });

  if (!found) {
    TT_LOG_WARN("[SessionManager] Session not found for slot assignment: {}",
                sessionId);
    return false;
  }

  TT_LOG_INFO("[SessionManager] Assigned slot {} to session {}", slotId,
              sessionId);
  return true;
}

uint32_t SessionManager::getSlotIdBySessionId(
    const std::string& sessionId) const {
  uint32_t result = domain::INVALID_SLOT_ID;
  sessions.modify(sessionId, [&result](domain::Session& s) {
    s.updateActivityTime();
    result = s.getSlotId();
  });
  TT_LOG_DEBUG(
      "[SessionManager] getSlotIdBySessionId sessionId={} -> slotId={}",
      sessionId, result);
  return result;
}

uint32_t SessionManager::acquireSessionSlot(const std::string& sessionId) {
  uint32_t result = domain::INVALID_SLOT_ID;
  bool wasInFlight = false;

  sessions.modify(sessionId, [&result, &wasInFlight](domain::Session& s) {
    wasInFlight = s.isInFlight();
    if (wasInFlight) {
      // Session is already in flight, don't modify
      return;
    }
    s.updateActivityTime();
    s.setInFlight(true);
    result = s.getSlotId();
  });

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
  return sessions.get(sessionId);
}

size_t SessionManager::getActiveSessionCount() const { return sessions.size(); }

void SessionManager::setSessionInFlight(const std::string& sessionId,
                                        bool inFlight) {
  bool found = sessions.modify(
      sessionId, [inFlight](domain::Session& s) { s.setInFlight(inFlight); });

  if (!found) {
    TT_LOG_WARN("[SessionManager] Session not found for in-flight update: {}",
                sessionId);
  } else {
    TT_LOG_DEBUG("[SessionManager] Set session {} in-flight: {}", sessionId,
                 inFlight);
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

  using Entry = std::pair<std::chrono::system_clock::time_point, std::string>;
  auto newer = [](const Entry& a, const Entry& b) { return a.first < b.first; };
  std::vector<Entry> heap;
  heap.reserve(evictionCount + 1);

  sessions.forEach([&heap, &newer, evictionCount](const std::string& id,
                                                  domain::Session& session) {
    if (session.isInFlight()) {
      return;
    }

    auto t = session.getLastActivityTime();
    if (heap.size() < evictionCount) {
      heap.emplace_back(t, id);
      std::push_heap(heap.begin(), heap.end(), newer);
    } else if (t < heap.front().first) {
      std::pop_heap(heap.begin(), heap.end(), newer);
      heap.back() = {t, id};
      std::push_heap(heap.begin(), heap.end(), newer);
    }
  });

  TT_LOG_DEBUG("[SessionManager] evictOldSessions: {} candidates for eviction",
               heap.size());
  size_t evicted = 0;
  for (const auto& [_, sessionId] : heap) {
    auto session = sessions.take(sessionId);
    if (!session.has_value()) {
      TT_LOG_DEBUG(
          "[SessionManager] evictOldSessions: session {} already removed",
          sessionId);
      continue;
    }
    uint32_t slotId = session->getSlotId();
    TT_LOG_DEBUG(
        "[SessionManager] evictOldSessions: evicting sessionId={}, slotId={}",
        sessionId, slotId);
    if (slotId != domain::INVALID_SLOT_ID) {
      sendDeallocRequest(sessionId, slotId);
    }
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
    trantor::EventLoop* callerEventLoop, std::optional<uint32_t> slotId) {
  TT_LOG_DEBUG(
      "[SessionManager] createSession called, slotId={}, activeSessions={}",
      slotId.has_value() ? std::to_string(slotId.value()) : "none",
      sessions.size());
  evictOldSessions();

  if (slotId.has_value()) {
    domain::Session session(slotId.value());
    sessions.insert(session.getSessionId(), session);
    TT_LOG_INFO("[SessionManager] Created session with pre-assigned slot: {}",
                slotId.value());
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

  domain::Session session = domain::Session(domain::INVALID_SLOT_ID);
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
          std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
      TT_LOG_DEBUG(
          "[SessionManager] sendAsyncAllocationRequest: queuing retry for "
          "sessionId={}, attemptsRemaining={}",
          pa.session.getSessionId(), pa.attemptsRemaining);
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
    sessions.insert(pendingAllocation.session.getSessionId(),
                    pendingAllocation.session);
    TT_LOG_DEBUG(
        "[SessionManager] handleMemoryResult: SUCCESS sessionId={}, "
        "assigned slotId={}",
        pendingAllocation.session.getSessionId(), result.slotIds.front());
    pendingAllocation.eventLoop->queueInLoop(
        [onCompletion = std::move(pendingAllocation.onCompletion),
         session = pendingAllocation.session]() { onCompletion(session); });
  } else if (pendingAllocation.attemptsRemaining > 0) {
    pendingAllocation.attemptsRemaining--;
    pendingAllocation.retryAt =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    TT_LOG_DEBUG(
        "[SessionManager] handleMemoryResult: FAILURE for sessionId={}, "
        "retrying in 500ms, attemptsRemaining={}",
        pendingAllocation.session.getSessionId(),
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

}  // namespace tt::services
