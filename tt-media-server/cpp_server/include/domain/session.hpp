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

namespace tt::services {
class SessionManager;  // friend: owns the locked state transitions below
}

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
   * Create a new session with an assigned slot.
   * @param slotId The assigned slot ID (required, must be valid)
   * @param initialHash Optional initial content hash (0 if not provided)
   */
  explicit Session(uint32_t slotId, size_t initialHash = 0);

  /**
   * Get the slot ID as a string (used as the session identifier).
   */
  std::string getSlotIdString() const { return std::to_string(slot_id_); }

  /**
   * Get the current content hash.
   */
  size_t getHash() const { return hash_; }

  /**
   * Update the content hash (called when conversation state changes).
   */
  void setHash(size_t hash) { hash_ = hash; }

  /**
   * Get the response id this session is currently registered under.
   * Empty when the session has never been registered under a response id.
   */
  const std::string& getResponseId() const { return response_id_; }

  /**
   * Update the response id this session is registered under (called when a
   * turn completes and the next turn should be reachable via
   * previous_response_id).
   */
  void setResponseId(const std::string& responseId) {
    response_id_ = responseId;
  }

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
  SessionState getState() const { return state_; }

  void setCancelFn(std::function<void()> fn) { cancelFn_ = std::move(fn); }
  std::function<void()> takeCancelFn() {
    return std::exchange(cancelFn_, nullptr);
  }

  // Release this session's in-flight hold (IN_FLIGHT -> IDLE) via an injected
  // callback. SessionManager sets this to run clearInFlight() under the
  // ConcurrentMap lock, so the transition can't race evictOldSessions(); kept
  // as a std::function so the domain layer doesn't depend on SessionManager.
  // No-op if unset (e.g. a session not owned by a SessionManager).
  void setReleaser(std::function<void()> r) { releaser_ = std::move(r); }
  void release() {
    if (releaser_) releaser_();
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
      std::function<void(uint32_t, const std::vector<utils::BlockHashInfo>&)>
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
    json["slot_id"] = slot_id_;
    return json;
  }

 protected:
  // State transitions are owned by SessionManager, which performs them under
  // the ConcurrentMap lock (serializing them against evictOldSessions()).
  // Protected + friend so only SessionManager — or a test subclass — can call
  // them; a direct unlocked call would re-introduce the clearInFlight()-vs-
  // eviction data race. Each returns false (state unchanged) if its
  // precondition is not met.
  friend class tt::services::SessionManager;
  bool markPrepared();   // IDLE           -> PREPARED
  bool markInFlight();   // IDLE/PREPARED  -> IN_FLIGHT
  bool clearInFlight();  // IN_FLIGHT      -> IDLE, also clears cancelFn

 private:
  size_t hash_;              // Current content hash, changes with conversation
  std::string response_id_;  // Current response id (Responses API key), empty
                             // until registered. Kept on the session so
                             // close/evict can remove the matching index entry.
  uint32_t slot_id_;
  SessionState state_{SessionState::IDLE};
  std::chrono::system_clock::time_point last_activity_time_;
  std::function<void()> cancelFn_;
  std::function<void()>
      releaser_;  // injected by SessionManager (see release())

  // Streaming token accumulator (initialized per-request)
  std::vector<int> deltaTokens_;
  std::vector<int> generatedTokens_;
  std::vector<utils::BlockHashInfo> initialBlocks_;
  uint64_t parentHash_ = 0;
  uint32_t parentThinkCount_ = 0;
  std::function<void(uint32_t, const std::vector<utils::BlockHashInfo>&)>
      onComplete_;

  // Thinking token tracking
  bool inThinkingBlock_ = false;
  uint32_t accumulatedThinkTokens_ = 0;
  int64_t thinkStartTokenId_ = 0;
  int64_t thinkEndTokenId_ = 0;
};

}  // namespace tt::domain
