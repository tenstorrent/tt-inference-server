// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/session_manager.hpp"

#include "utils/logger.hpp"

namespace tt::services {

domain::Session SessionManager::createSession(std::optional<int> slotId) {
  std::lock_guard<std::mutex> lock(mutex_);

  int slot = slotId.value_or(-1);
  domain::Session session(slot);

  std::string sessionId = session.getSessionId();
  sessions_[sessionId] = session;

  TT_LOG_INFO("[SessionManager] Created session: {} with slot: {}",
              sessionId, slot);

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

}  // namespace tt::services
