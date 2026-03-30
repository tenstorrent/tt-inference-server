// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <chrono>
#include <optional>
#include <random>
#include <sstream>
#include <string>

namespace tt::domain {

/**
 * Session represents a user session with an optional slot assignment.
 */
class Session {
 public:
  /**
   * Create a new session with a generated UUID.
   * @param slotId Optional slot ID (-1 means unassigned)
   */
  explicit Session(int slotId = -1);

  /**
   * Get the session ID (UUID).
   */
  std::string getSessionId() const { return session_id_; }

  /**
   * Get the assigned slot ID.
   * @return Slot ID, or -1 if unassigned
   */
  int getSlotId() const { return slot_id_; }

  /**
   * Assign a slot ID to this session.
   */
  void setSlotId(int slotId) { slot_id_ = slotId; }

  /**
   * Check if a slot is assigned.
   */
  bool hasSlot() const { return slot_id_ != -1; }

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
  int slot_id_;
  std::chrono::system_clock::time_point last_activity_time_;

  /**
   * Generate a UUID v4 string.
   */
  static std::string generateUuid();
};

}  // namespace tt::domain
