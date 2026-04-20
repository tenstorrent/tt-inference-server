// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <chrono>
#include <cstdint>
#include <string>

#include "domain/manage_memory.hpp"

namespace tt::domain {

/**
 * Lifecycle state of a Session.
 *
 * Transitions:
 *   IDLE          --(markInFlight)-----> IN_FLIGHT
 *   IN_FLIGHT     --(clearInFlight)----> IDLE
 *   IN_FLIGHT     --(markPendingClose)-> PENDING_CLOSE
 *   PENDING_CLOSE --(clearInFlight)----> CLOSING
 */
enum class SessionState {
  IDLE,           // no active request
  IN_FLIGHT,      // request actively being processed
  PENDING_CLOSE,  // close requested; slot freed once in-flight request finishes
  CLOSING,        // in-flight request finished; slot deallocation pending
};

class Session {
 public:
  explicit Session(uint32_t slotId = INVALID_SLOT_ID);

  std::string getSessionId() const { return session_id_; }
  uint32_t getSlotId() const { return slot_id_; }
  void setSlotId(uint32_t slotId) { slot_id_ = slotId; }
  bool hasSlot() const { return slot_id_ != INVALID_SLOT_ID; }

  bool isIdle() const { return state_ == SessionState::IDLE; }
  bool isInFlight() const { return state_ == SessionState::IN_FLIGHT; }
  bool isPendingClose() const { return state_ == SessionState::PENDING_CLOSE; }
  bool isClosing() const { return state_ == SessionState::CLOSING; }

  SessionState getState() const { return state_; }

  // Transition methods return false (without changing state) if the
  // precondition is not met.
  bool markInFlight();      // IDLE      -> IN_FLIGHT
  bool clearInFlight();     // IN_FLIGHT -> IDLE  |  PENDING_CLOSE -> CLOSING
  bool markPendingClose();  // IN_FLIGHT -> PENDING_CLOSE

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
  std::string session_id_;
  uint32_t slot_id_;
  SessionState state_{SessionState::IDLE};
  std::chrono::system_clock::time_point last_activity_time_;

  static std::string generateUuid();
};

}  // namespace tt::domain
