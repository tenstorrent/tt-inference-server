// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "domain/manage_memory.hpp"
#include "domain/sentinel_values.hpp"
#include "utils/conversation_hasher.hpp"

namespace tt::domain {

// Lifecycle state of a Session.  IDLE --(markPrepared)--> PREPARED
// --(markInFlight)--> IN_FLIGHT --(clearInFlight)--> IDLE.
// IDLE can also transition directly to IN_FLIGHT via markInFlight (fast path).
enum class SessionState {
  IDLE,       // no active request
  PREPARED,   // session has been allocated to a slot
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

  bool isPrepared() const { return state_ == SessionState::PREPARED; }
  bool markPrepared();

  SessionState getState() const { return state_; }

  // Transition methods return false (without changing state) if the
  // precondition is not met.
  bool markInFlight();   // IDLE      -> IN_FLIGHT
  bool clearInFlight();  // IN_FLIGHT -> IDLE, also clears cancelFn

  void setCancelFn(std::function<void()> fn) { cancelFn_ = std::move(fn); }
  std::function<void()> takeCancelFn() {
    return std::exchange(cancelFn_, nullptr);
  }

  std::chrono::system_clock::time_point getLastActivityTime() const {
    return last_activity_time_;
  }

  void updateActivityTime() {
    last_activity_time_ = std::chrono::system_clock::now();
  }

  /**
   * Initialize token accumulator for streaming hash computation.
   * Called once per request when session routing is resolved.
   *
   * @param deltaTokens Delta prompt tokens (after matched prefix trimmed)
   * @param initialBlocks Block info computed from the prompt (for prepending)
   * @param onComplete Callback invoked at stream end with final block info
   * @param parentThinkCount Cumulative think tokens already present in the
   *        matched KV prefix. Seeded from the matched session's accumulated
   *        count on a prefix-cache HIT so think tokens accumulate across turns;
   *        0 for a fresh session.
   */
  void initTokenAccumulator(
      std::vector<int> deltaTokens,
      std::vector<utils::BlockHashInfo> initialBlocks,
      std::function<void(const std::string&,
                         const std::vector<utils::BlockHashInfo>&)>
          onComplete,
      uint32_t parentThinkCount = 0);

  /**
   * Add a generated token to the accumulator.
   */
  void addGeneratedToken(int tokenId);

  /**
   * Compute final hashes and register any new blocks.
   * Called at stream end before clearInFlight().
   */
  void finalizeAndRegisterHashes();

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
  std::function<void()> cancelFn_;

  // Streaming token accumulator (initialized per-request)
  std::vector<int> deltaTokens_;
  std::vector<int> generatedTokens_;
  std::vector<utils::BlockHashInfo> initialBlocks_;
  uint64_t parentHash_ = 0;
  uint32_t parentThinkCount_ = 0;
  std::function<void(const std::string&,
                     const std::vector<utils::BlockHashInfo>&)>
      onComplete_;

  // Thinking token tracking
  bool inThinkingBlock_ = false;
  uint32_t accumulatedThinkTokens_ = 0;
  int64_t thinkStartTokenId_ = 0;
  int64_t thinkEndTokenId_ = 0;

  static std::string generateUuid();
};

}  // namespace tt::domain
