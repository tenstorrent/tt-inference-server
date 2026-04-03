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
        ipc::k_memory_request_queue_name, ipc::MEMORY_QUEUE_CAPACITY);
    memoryResultQueue = std::make_unique<ipc::MemoryResultQueue>(
        ipc::k_memory_result_queue_name, ipc::MEMORY_QUEUE_CAPACITY);
    TT_LOG_INFO("[SessionManager] Created memory management IPC queues");
    drainThread = std::thread([this] { drainResultQueue(); });
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

void SessionManager::drainResultQueue() {
  while (!stopped.load(std::memory_order_relaxed)) {
    domain::ManageMemoryResult result;
    if (memoryResultQueue->tryPop(result)) {
      auto promise = pendingAllocations.take(result.taskId);
      if (promise.has_value()) {
        if (result.status == domain::ManageMemoryStatus::SUCCESS &&
            !result.slotIds.empty()) {
          (*promise)->set_value(result.slotIds[0]);
        } else {
          (*promise)->set_value(INVALID_SLOT_ID);
        }
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

domain::Session SessionManager::createSession(std::optional<uint32_t> slotId,
                                              bool inFlight) {
  evictOldSessions();

  uint32_t slot = slotId.value_or(INVALID_SLOT_ID);
  domain::Session session(slot);
  std::string sessionId = session.getSessionId();

  if (!slotId.has_value() && memoryRequestQueue && memoryResultQueue) {
    constexpr int maxRetries = 10;
    constexpr int retryDelayMs = 500;

    for (int attempt = 0; attempt < maxRetries; ++attempt) {
      if (attempt > 0) {
        TT_LOG_DEBUG(
            "[SessionManager] Retry attempt {}/{} for slot allocation for "
            "session {}",
            attempt + 1, maxRetries, sessionId);
        std::this_thread::sleep_for(std::chrono::milliseconds(retryDelayMs));
      }

      auto future = requestSlotIdFromMemoryManager(sessionId);
      auto status = future.wait_for(std::chrono::seconds(1));

      if (status == std::future_status::ready) {
        uint32_t allocatedSlot = future.get();
        if (allocatedSlot != INVALID_SLOT_ID) {
          slot = allocatedSlot;
          session.setSlotId(slot);
          TT_LOG_INFO(
              "[SessionManager] Received slot {} for session {} from memory "
              "manager (attempt {}/{})",
              slot, sessionId, attempt + 1, maxRetries);
          break;
        } else {
          TT_LOG_WARN(
              "[SessionManager] Memory manager returned INVALID_SLOT_ID for "
              "session {} (attempt {}/{})",
              sessionId, attempt + 1, maxRetries);
        }
      } else {
        TT_LOG_WARN(
            "[SessionManager] Timeout waiting for slot allocation for session "
            "{} (attempt {}/{})",
            sessionId, attempt + 1, maxRetries);
      }
    }

    if (slot == INVALID_SLOT_ID) {
      TT_LOG_ERROR(
          "[SessionManager] Failed to allocate slot for session {} after {} "
          "attempts",
          sessionId, maxRetries);
      throw std::runtime_error(
          "Failed to allocate memory slot after " + std::to_string(maxRetries) +
          " attempts. System may be out of memory resources.");
    }
  }

  session.setInFlight(inFlight);
  sessions.insert(sessionId, session);

  TT_LOG_INFO("[SessionManager] Created session: {} with slot: {} inFlight: {}",
              sessionId,
              slot == INVALID_SLOT_ID ? "none" : std::to_string(slot),
              inFlight);
  return session;
}

bool SessionManager::closeSession(const std::string& sessionId) {
  auto session = sessions.take(sessionId);
  if (!session.has_value()) {
    TT_LOG_WARN("[SessionManager] Session not found: {}", sessionId);
    return false;
  }

  uint32_t slotId = session->getSlotId();
  if (slotId != INVALID_SLOT_ID) {
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
  uint32_t result = INVALID_SLOT_ID;
  sessions.modify(sessionId, [&result](domain::Session& s) {
    s.updateActivityTime();
    result = s.getSlotId();
  });
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

  size_t evicted = 0;
  for (const auto& [_, sessionId] : heap) {
    auto session = sessions.take(sessionId);
    if (!session.has_value()) {
      continue;
    }
    uint32_t slotId = session->getSlotId();
    if (slotId != INVALID_SLOT_ID) {
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

std::future<uint32_t> SessionManager::requestSlotIdFromMemoryManager(
    const std::string& sessionId) {
  auto promise = std::make_shared<std::promise<uint32_t>>();
  auto future = promise->get_future();

  domain::ManageMemoryTask task;
  uint32_t requestTaskId = tt::utils::TaskIDGenerator::generate();
  task.taskId = requestTaskId;

  pendingAllocations.insert(requestTaskId, promise);
  task.action = domain::MemoryManagementAction::ALLOCATE;
  task.inputSeqLen = 0;
  task.memoryLayout = domain::KvMemoryLayout::Paged;
  task.slotIds = {};

  try {
    memoryRequestQueue->push(task);
    TT_LOG_DEBUG("[SessionManager] Sent slot allocation request for session {}",
                 sessionId);
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[SessionManager] Error requesting slot for session {}: {}",
                 sessionId, e.what());
    pendingAllocations.erase(requestTaskId);
    promise->set_value(INVALID_SLOT_ID);
  }

  return future;
}

void SessionManager::sendDeallocRequest(const std::string& sessionId,
                                        uint32_t slotId) {
  if (!memoryRequestQueue) {
    return;
  }

  domain::ManageMemoryTask task;
  task.taskId = utils::TaskIDGenerator::generate();
  task.action = domain::MemoryManagementAction::DEALLOCATE;
  task.inputSeqLen = 0;
  task.memoryLayout = domain::KvMemoryLayout::Paged;
  task.slotIds = {slotId};

  try {
    memoryRequestQueue->push(task);
    TT_LOG_INFO("[SessionManager] Sent dealloc request for session {} slot {}",
                sessionId, slotId);
  } catch (const std::exception& e) {
    TT_LOG_ERROR(
        "[SessionManager] Failed to send dealloc for session {} slot {}: {}",
        sessionId, slotId, e.what());
  }
}

}  // namespace tt::services
