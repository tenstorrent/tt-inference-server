// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "domain/session.hpp"
#include "ipc/boost_ipc_memory_queue.hpp"

namespace tt::services {

/**
 * SessionManager manages user sessions and slot assignments.
 */
class SessionManager {
 public:
  SessionManager();
  ~SessionManager() = default;

  // Non-copyable
  SessionManager(const SessionManager&) = delete;
  SessionManager& operator=(const SessionManager&) = delete;

  /**
   * Create a new session.
   * @param slotId Optional slot ID to assign (max uint32_t means unassigned)
   * @return The created session
   */
  domain::Session createSession(std::optional<uint32_t> slotId = std::nullopt);

  /**
   * Close a session and remove it from the manager.
   * @param sessionId The session ID to close
   * @return true if session was found and closed, false otherwise
   */
  bool closeSession(const std::string& sessionId);

  /**
   * Assign a slot ID to an existing session.
   * @param sessionId The session ID
   * @param slotId The slot ID to assign
   * @return true if session was found and slot assigned, false otherwise
   */
  bool assignSlotId(const std::string& sessionId, uint32_t slotId);

  /**
   * Get the slot ID for a session.
   * @param sessionId The session ID
   * @return The slot ID, or max uint32_t if session not found or no slot
   * assigned
   */
  uint32_t getSlotIdBySessionId(const std::string& sessionId) const;

  /**
   * Get a session by ID.
   * @param sessionId The session ID
   * @return Optional session object
   */
  std::optional<domain::Session> getSession(const std::string& sessionId) const;

  /**
   * Get the count of active sessions.
   */
  size_t getActiveSessionCount() const;

 private:
  /**
   * Evict old sessions if the eviction threshold is exceeded.
   * Called automatically when creating new sessions.
   */
  void evictOldSessions();

  /**
   * Find the N oldest sessions (by last activity time).
   * @param count Number of oldest sessions to find
   * @return Vector of session IDs, sorted from oldest to newest
   */
  std::vector<std::string> findOldestSessions(size_t count) const;

  /**
   * Close a session without locking (assumes mutex is already locked).
   * @param sessionId The session ID to close
   * @return true if session was found and closed, false otherwise
   */
  bool closeSessionLocked(const std::string& sessionId);

  /**
   * Request a slot ID from the memory manager.
   * @param sessionId The session ID requesting the slot
   * @return The allocated slot ID, or max uint32_t on failure
   */
  uint32_t requestSlotIdFromMemoryManager(const std::string& sessionId);

  mutable std::mutex mutex_;
  std::unordered_map<std::string, domain::Session> sessions_;

  // IPC queues for memory management
  std::unique_ptr<ipc::MemoryRequestQueue> memoryRequestQueue_;
  std::unique_ptr<ipc::MemoryResultQueue> memoryResultQueue_;
};

}  // namespace tt::services
