// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/session_manager.hpp"

#include <algorithm>

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace tt::services {

domain::Session SessionManager::createSession(std::optional<int> slotId) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if we need to evict old sessions
  evictOldSessions();

  int slot = slotId.value_or(-1);
  domain::Session session(slot);

  std::string sessionId = session.getSessionId();
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

bool SessionManager::assignSlotId(const std::string& sessionId, int slotId) {
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

int SessionManager::getSlotIdBySessionId(const std::string& sessionId) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = sessions_.find(sessionId);
  if (it == sessions_.end()) {
    return -1;
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

}  // namespace tt::services
