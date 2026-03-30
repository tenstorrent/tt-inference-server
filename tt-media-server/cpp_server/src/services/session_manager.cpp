// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/session_manager.hpp"

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

  size_t activeCount = sessions_.size();

  // Calculate if we've exceeded the eviction threshold
  // (activeCount / maxSessions) * 100 > evictionRate
  if (activeCount * 100 > maxSessions * evictionRate) {
    auto oldestSessionId = findOldestSession();
    if (oldestSessionId.has_value()) {
      closeSession(oldestSessionId.value());
      TT_LOG_INFO(
          "[SessionManager] Evicted oldest session: {} (active: {}/{}, "
          "threshold: {}%)",
          oldestSessionId.value(), activeCount, maxSessions, evictionRate);
    }
  }
}

std::optional<std::string> SessionManager::findOldestSession() const {
  // Note: mutex_ is already locked by the caller

  if (sessions_.empty()) {
    return std::nullopt;
  }

  auto oldestIt = sessions_.begin();
  auto oldestTime = oldestIt->second.getLastActivityTime();

  for (auto it = sessions_.begin(); it != sessions_.end(); ++it) {
    auto activityTime = it->second.getLastActivityTime();
    if (activityTime < oldestTime) {
      oldestTime = activityTime;
      oldestIt = it;
    }
  }

  return oldestIt->first;
}

}  // namespace tt::services
