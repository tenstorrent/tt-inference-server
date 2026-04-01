// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/session_manager.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>

#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

namespace tt::services {

SessionManager::SessionManager() {
  // Create IPC queues for communication with MemoryManager (in worker
  // processes) If workers aren't running or in non-decode mode, these queues
  // won't be used
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
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if we need to evict old sessions
  evictOldSessions();

  // Create session first to get the session ID
  uint32_t slot = slotId.value_or(std::numeric_limits<uint32_t>::max());
  domain::Session session(slot);
  std::string sessionId = session.getSessionId();

  // If no slot ID was provided, request one from memory manager
  if (!slotId.has_value() && memoryRequestQueue_ && memoryResultQueue_) {
    uint32_t allocatedSlot = requestSlotIdFromMemoryManager(sessionId);
    if (allocatedSlot != std::numeric_limits<uint32_t>::max()) {
      slot = allocatedSlot;
      session.setSlotId(slot);
    }
  }

  sessions_[sessionId] = session;

  TT_LOG_INFO("[SessionManager] Created session: {} with slot: {}", sessionId,
              slot);

  return session;
}

bool SessionManager::closeSession(const std::string& sessionId) {
  std::lock_guard<std::mutex> lock(mutex_);
  return closeSessionLocked(sessionId);
}

bool SessionManager::closeSessionLocked(const std::string& sessionId) {
  // Note: mutex_ must already be locked by caller
  auto it = sessions_.find(sessionId);
  if (it == sessions_.end()) {
    TT_LOG_WARN("[SessionManager] Session not found: {}", sessionId);
    return false;
  }

  sessions_.erase(it);
  TT_LOG_INFO("[SessionManager] Closed session: {}", sessionId);
  return true;
}

bool SessionManager::assignSlotId(const std::string& sessionId,
                                  uint32_t slotId) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = sessions_.find(sessionId);
  if (it == sessions_.end()) {
    TT_LOG_WARN("[SessionManager] Session not found for slot assignment: {}",
                sessionId);
    return false;
  }

  it->second.setSlotId(slotId);
  TT_LOG_INFO("[SessionManager] Assigned slot {} to session {}", slotId,
              sessionId);
  return true;
}

uint32_t SessionManager::getSlotIdBySessionId(
    const std::string& sessionId) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = sessions_.find(sessionId);
  if (it == sessions_.end()) {
    return std::numeric_limits<uint32_t>::max();
  }

  // Update activity time (mutable access through const_cast for this use case)
  const_cast<domain::Session&>(it->second).updateActivityTime();

  return it->second.getSlotId();
}

std::optional<domain::Session> SessionManager::getSession(
    const std::string& sessionId) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = sessions_.find(sessionId);
  if (it == sessions_.end()) {
    return std::nullopt;
  }

  return it->second;
}

size_t SessionManager::getActiveSessionCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return sessions_.size();
}

void SessionManager::evictOldSessions() {
  // Note: mutex_ is already locked by the caller (createSession)

  size_t maxSessions = tt::config::maxSessionsCount();
  unsigned evictionRate = tt::config::sessionEvictionRate();
  size_t evictionCount = tt::config::sessionEvictionCount();

  size_t activeCount = sessions_.size();

  // Calculate if we've exceeded the eviction threshold
  // (activeCount / maxSessions) * 100 > evictionRate
  if (activeCount * 100 > maxSessions * evictionRate) {
    // Find and evict the oldest sessions
    auto oldestSessions = findOldestSessions(evictionCount);

    if (!oldestSessions.empty()) {
      for (const auto& sessionId : oldestSessions) {
        closeSessionLocked(sessionId);
      }

      TT_LOG_INFO(
          "[SessionManager] Evicted {} oldest session(s) (active: {}/{}, "
          "threshold: {}%)",
          oldestSessions.size(), activeCount, maxSessions, evictionRate);
    }
  }
}

std::vector<std::string> SessionManager::findOldestSessions(
    size_t count) const {
  // Note: mutex_ is already locked by the caller

  if (sessions_.empty()) {
    return {};
  }

  // Create a vector of (sessionId, activityTime) pairs
  std::vector<std::pair<std::string, std::chrono::system_clock::time_point>>
      sessionTimes;
  sessionTimes.reserve(sessions_.size());

  for (const auto& [sessionId, session] : sessions_) {
    sessionTimes.emplace_back(sessionId, session.getLastActivityTime());
  }

  // Sort by activity time (oldest first)
  std::partial_sort(sessionTimes.begin(),
                    sessionTimes.begin() + std::min(count, sessionTimes.size()),
                    sessionTimes.end(), [](const auto& a, const auto& b) {
                      return a.second < b.second;
                    });

  // Extract session IDs
  std::vector<std::string> result;
  size_t numToEvict = std::min(count, sessionTimes.size());
  result.reserve(numToEvict);

  for (size_t i = 0; i < numToEvict; ++i) {
    result.push_back(sessionTimes[i].first);
  }

  return result;
}

uint32_t SessionManager::requestSlotIdFromMemoryManager(
    const std::string& sessionId) {
  // Note: mutex_ is already locked by the caller

  // Create memory allocation request
  domain::ManageMemoryTask task;
  uint32_t requestTaskId = tt::utils::TaskIDGenerator::generate();
  task.taskId = requestTaskId;
  task.action = domain::MemoryManagementAction::ALLOCATE;
  task.inputSeqLen = 0;  // Not used for slot allocation
  task.memoryLayout = domain::KvMemoryLayout::Paged;
  task.slotIds = {};  // Empty for allocation request

  try {
    // Send allocation request
    memoryRequestQueue_->push(task);
    TT_LOG_DEBUG("[SessionManager] Sent slot allocation request for session {}",
                 sessionId);

    // Wait for response with timeout
    const int maxAttempts = 100;  // ~1 second with 10ms sleep
    int attempts = 0;

    while (attempts < maxAttempts) {
      domain::ManageMemoryResult result;
      if (memoryResultQueue_->tryPop(result)) {
        // Check if this is our result
        if (result.taskId == requestTaskId) {
          if (result.status == domain::ManageMemoryStatus::SUCCESS &&
              !result.slotIds.empty()) {
            TT_LOG_INFO(
                "[SessionManager] Received slot {} for session {} from memory "
                "manager",
                result.slotIds[0], sessionId);
            return result.slotIds[0];
          } else {
            TT_LOG_WARN(
                "[SessionManager] Memory manager failed to allocate slot for "
                "session {}",
                sessionId);
            return std::numeric_limits<uint32_t>::max();
          }
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      attempts++;
    }

    TT_LOG_WARN(
        "[SessionManager] Timeout waiting for slot allocation for session {}",
        sessionId);
    return std::numeric_limits<uint32_t>::max();

  } catch (const std::exception& e) {
    TT_LOG_ERROR("[SessionManager] Error requesting slot for session {}: {}",
                 sessionId, e.what());
    return std::numeric_limits<uint32_t>::max();
  }
}

}  // namespace tt::services
