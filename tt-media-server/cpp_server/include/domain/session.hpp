// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <random>
#include <sstream>
#include <string>

#include "domain/manage_memory.hpp"

namespace tt::domain {

/**
 * Session represents a user session with an optional slot assignment.
 */
class Session {
 public:
  /**
   * Create a new session with a generated UUID.
   * @param slotId Optional slot ID (max uint32_t means unassigned)
   */
  explicit Session(uint32_t slotId = INVALID_SLOT_ID);

  /**
   * Get the session ID (UUID).
   */
  std::string getSessionId() const { return session_id_; }

  /**
   * Get the assigned slot ID.
   * @return Slot ID, or max uint32_t if unassigned
   */
  uint32_t getSlotId() const { return slot_id_; }

  /**
   * Assign a slot ID to this session.
   */
  void setSlotId(uint32_t slotId) { slot_id_ = slotId; }

  /**
   * Check if a slot is assigned.
   */
  bool hasSlot() const { return slot_id_ != INVALID_SLOT_ID; }

  /**
   * Check if the session is in-flight (has an active request).
   */
  bool isInFlight() const { return in_flight_; }

  /**
   * Set the in-flight status of the session.
   */
  void setInFlight(bool inFlight) { in_flight_ = inFlight; }

  /**
   * Check if a close was requested while the session was in-flight.
   */
  bool isPendingClose() const { return pending_close_; }

  /**
   * Mark the session for deferred close (set when closeSession is called
   * while a request is in-flight).
   */
  void setPendingClose(bool pendingClose) { pending_close_ = pendingClose; }

  /**
   * Get the last activity time.
   */
  std::chrono::system_clock::time_point getLastActivityTime() const {
    return last_activity_time_;
  }

  /**
   * Update the last activity time to now.
   */
  void updateActivityTime() {
    last_activity_time_ = std::chrono::system_clock::now();
  }

  /**
   * Convert to JSON representation.
   */
  Json::Value toJson() const {
    Json::Value json;
    json["session_id"] = session_id_;
    json["slot_id"] = slot_id_;
    return json;
  }

 private:
  std::string session_id_;
  uint32_t slot_id_;
  bool in_flight_{false};
  bool pending_close_{false};
  std::chrono::system_clock::time_point last_activity_time_;

  /**
   * Generate a UUID v4 string.
   */
  static std::string generateUuid();
};

}  // namespace tt::domain
