// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/session.hpp"

#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>

namespace tt::domain {

Session::Session(uint32_t slotId)
    : slot_id_(slotId), last_activity_time_(std::chrono::system_clock::now()) {
  session_id_ = generateUuid();
}

bool Session::markInFlight() {
  if (state_ != SessionState::IDLE) return false;
  state_ = SessionState::IN_FLIGHT;
  return true;
}

bool Session::clearInFlight() {
  if (state_ != SessionState::IN_FLIGHT &&
      state_ != SessionState::PENDING_CLOSE) {
    return false;
  }
  state_ = (state_ == SessionState::IN_FLIGHT) ? SessionState::IDLE
                                               : SessionState::CLOSING;
  return true;
}

bool Session::markPendingClose() {
  if (state_ != SessionState::IN_FLIGHT) return false;
  state_ = SessionState::PENDING_CLOSE;
  return true;
}

std::string Session::generateUuid() {
  static std::mutex genMutex;
  static std::random_device rd;
  static std::mt19937_64 gen(rd());

  std::lock_guard<std::mutex> lock(genMutex);

  // Generate two 64-bit random numbers
  uint64_t part1 = gen();
  uint64_t part2 = gen();

  // Format as UUID v4: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
  std::ostringstream ss;
  ss << std::hex << std::setfill('0');
  ss << std::setw(8) << (part1 & 0xFFFFFFFF) << '-';
  ss << std::setw(4) << ((part1 >> 32) & 0xFFFF) << '-';
  ss << "4" << std::setw(3) << ((part1 >> 48) & 0x0FFF) << '-';
  ss << std::setw(1) << (8 | ((part2 & 0x3))) << std::setw(3)
     << ((part2 >> 2) & 0xFFF) << '-';
  ss << std::setw(12) << ((part2 >> 14) & 0xFFFFFFFFFFFF);

  return ss.str();
}

}  // namespace tt::domain
