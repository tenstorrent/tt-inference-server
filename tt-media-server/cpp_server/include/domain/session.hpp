// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <chrono>
#include <cstdint>
#include <string>

#include "domain/manage_memory.hpp"

namespace tt::domain {

// Lifecycle state of a Session.  IDLE <--(clearInFlight)--> IN_FLIGHT.
enum class SessionState {
  IDLE,       // no active request
  IN_FLIGHT,  // request actively being processed
};

class Session {
 public:
  /**
   * Create a new session with a generated UUID.
   * @param slotId Optional slot ID (max uint32_t means unassigned)
   * @param initialHash Optional initial content hash (0 if not provided)
   */
  explicit Session(uint32_t slotId = INVALID_SLOT_ID, size_t initialHash = 0);

  /**
   * Get the stable session ID (UUID).
   */
  const std::string& getSessionId() const { return session_id_; }

  /**
   * Get the current content hash.
   */
  size_t getHash() const { return hash_; }

  /**
   * Update the content hash (called when conversation state changes).
   */
  void setHash(size_t hash) { hash_ = hash; }

  /**
   * Get the assigned slot ID.
   * @return Slot ID, or max uint32_t if unassigned
   */
  uint32_t getSlotId() const { return slot_id_; }
  void setSlotId(uint32_t slotId) { slot_id_ = slotId; }
  bool hasSlot() const { return slot_id_ != INVALID_SLOT_ID; }

  bool isIdle() const { return state_ == SessionState::IDLE; }
  bool isInFlight() const { return state_ == SessionState::IN_FLIGHT; }

  SessionState getState() const { return state_; }

  // Transition methods return false (without changing state) if the
  // precondition is not met.
  bool markInFlight();   // IDLE      -> IN_FLIGHT
  bool clearInFlight();  // IN_FLIGHT -> IDLE

  std::chrono::system_clock::time_point getLastActivityTime() const {
    return last_activity_time_;
  }

  void updateActivityTime() {
    last_activity_time_ = std::chrono::system_clock::now();
  }

  Json::Value toJson() const {
    Json::Value json;
    json["session_id"] = session_id_;
    json["slot_id"] = slot_id_;
    return json;
  }

 private:
  std::string session_id_;  // Stable UUID, never changes
  size_t hash_;             // Current content hash, changes with conversation
  uint32_t slot_id_;
  SessionState state_{SessionState::IDLE};
  std::chrono::system_clock::time_point last_activity_time_;

  static std::string generateUuid();
};

}  // namespace tt::domain
