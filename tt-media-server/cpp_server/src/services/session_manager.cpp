// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/session_manager.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>

#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "utils/logger.hpp"
namespace {
constexpr uint32_t INVALID_SLOT_ID = std::numeric_limits<uint32_t>::max();
}

namespace tt::services {

SessionManager::SessionManager() {
  try {
    memoryRequestQueue_ = std::make_unique<ipc::MemoryRequestQueue>(
        ipc::k_memory_request_queue_name, ipc::MEMORY_QUEUE_CAPACITY);
    memoryResultQueue_ = std::make_unique<ipc::MemoryResultQueue>(
        ipc::k_memory_result_queue_name, ipc::MEMORY_QUEUE_CAPACITY);
    TT_LOG_INFO("[SessionManager] Created memory management IPC queues");
  } catch (const std::exception& e) {
    TT_LOG_WARN(
        "[SessionManager] Failed to create memory queues: {}. Slot allocation "
        "will not be available.",
        e.what());
    memoryRequestQueue_.reset();
    memoryResultQueue_.reset();
  }
}

domain::Session SessionManager::createSession(std::optional<uint32_t> slotId) {
  evictOldSessions();

  uint32_t slot = slotId.value_or(INVALID_SLOT_ID);
  domain::Session session(slot);
  std::string sessionId = session.getSessionId();

  if (!slotId.has_value() && memoryRequestQueue_ && memoryResultQueue_) {
    uint32_t allocatedSlot = requestSlotIdFromMemoryManager(sessionId);
    if (allocatedSlot != INVALID_SLOT_ID) {
      slot = allocatedSlot;
      session.setSlotId(slot);
    }
  }

  sessions_.insert(sessionId, session);

  TT_LOG_INFO("[SessionManager] Created session: {} with slot: {}", sessionId,
              slot);
  return session;
}

bool SessionManager::closeSession(const std::string& sessionId) {
  auto session = sessions_.take(sessionId);
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
  bool found = sessions_.modify(
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
  sessions_.modify(sessionId, [&result](domain::Session& s) {
    s.updateActivityTime();
    result = s.getSlotId();
  });
  return result;
}

std::optional<domain::Session> SessionManager::getSession(
    const std::string& sessionId) const {
  return sessions_.get(sessionId);
}

size_t SessionManager::getActiveSessionCount() const {
  return sessions_.size();
}

void SessionManager::evictOldSessions() {
  size_t maxSessions = tt::config::maxSessionsCount();
  unsigned evictionRate = tt::config::sessionEvictionRate();
  size_t evictionCount = tt::config::sessionEvictionCount();

  size_t activeCount = sessions_.size();
  if (activeCount * 100 <= maxSessions * evictionRate) {
    return;
  }

  using Entry = std::pair<std::chrono::system_clock::time_point, std::string>;
  // Max-heap by time: the top is the youngest among the k oldest candidates.
  // For each session, if the heap is under capacity or the session is older
  // than the heap top, it belongs in our eviction set.
  auto newer = [](const Entry& a, const Entry& b) { return a.first < b.first; };
  std::vector<Entry> heap;
  heap.reserve(evictionCount + 1);

  sessions_.forEach([&heap, &newer, evictionCount](const std::string& id,
                                                   domain::Session& session) {
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
    auto session = sessions_.take(sessionId);
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

uint32_t SessionManager::requestSlotIdFromMemoryManager(
    const std::string& sessionId) {
  std::lock_guard<std::mutex> lock(allocationMutex_);

  domain::ManageMemoryTask task;
  task.taskId = domain::TaskID(sessionId);
  task.action = domain::MemoryManagementAction::ALLOCATE;
  task.inputSeqLen = 0;
  task.memoryLayout = domain::KvMemoryLayout::Paged;
  task.slotIds = {};

  try {
    memoryRequestQueue_->push(task);
    TT_LOG_DEBUG("[SessionManager] Sent slot allocation request for session {}",
                 sessionId);

    const int maxAttempts = 100;  // ~1 second with 10ms sleep
    int attempts = 0;

    while (attempts < maxAttempts) {
      domain::ManageMemoryResult result;
      if (memoryResultQueue_->tryPop(result)) {
        if (result.taskId.id == sessionId) {
          if (result.status == domain::ManageMemoryStatus::SUCCESS &&
              !result.slotIds.empty()) {
            TT_LOG_INFO(
                "[SessionManager] Received slot {} for session {} from memory "
                "manager",
                result.slotIds[0], sessionId);
            return result.slotIds[0];
          }
          TT_LOG_WARN(
              "[SessionManager] Memory manager failed to allocate slot for "
              "session {}",
              sessionId);
          return INVALID_SLOT_ID;
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      attempts++;
    }

    TT_LOG_WARN(
        "[SessionManager] Timeout waiting for slot allocation for session {}",
        sessionId);
    return INVALID_SLOT_ID;

  } catch (const std::exception& e) {
    TT_LOG_ERROR("[SessionManager] Error requesting slot for session {}: {}",
                 sessionId, e.what());
    return INVALID_SLOT_ID;
  }
}

void SessionManager::sendDeallocRequest(const std::string& sessionId,
                                        uint32_t slotId) {
  if (!memoryRequestQueue_) {
    return;
  }

  domain::ManageMemoryTask task;
  task.taskId = domain::TaskID(sessionId);
  task.action = domain::MemoryManagementAction::DEALLOCATE;
  task.inputSeqLen = 0;
  task.memoryLayout = domain::KvMemoryLayout::Paged;
  task.slotIds = {slotId};

  try {
    memoryRequestQueue_->push(task);
    TT_LOG_INFO("[SessionManager] Sent dealloc request for session {} slot {}",
                sessionId, slotId);
  } catch (const std::exception& e) {
    TT_LOG_ERROR(
        "[SessionManager] Failed to send dealloc for session {} slot {}: {}",
        sessionId, slotId, e.what());
  }
}

}  // namespace tt::services
