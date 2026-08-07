// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "domain/prefix_cache/helpers.hpp"
#include "domain/prefix_cache/prefix_index.hpp"
#include "domain/prefix_cache/response_id_index.hpp"
#include "domain/session.hpp"
#include "domain/session_manager_structs.hpp"
#include "utils/conversation_hasher.hpp"

namespace trantor {
class EventLoop;
}

namespace tt::services {

struct PrefixCacheAcquireResult {
  bool sessionFound = false;
  std::string sessionId;
  uint32_t slotId = 0;
  uint32_t numberOfMatchedTokens = 0;
  uint32_t accumulatedThinkTokens = 0;
  std::vector<domain::prefix_cache::Candidate> candidatesList;
};

/**
 * Options for getSlot() - all the optional routing hints.
 */
struct GetSlotOptions {
  std::optional<std::string> previousResponseId;  // Response-id continuation
  std::optional<std::string> responseId;          // New response id to register
  std::function<void()> cancelFn;                 // Cancellation callback
};

/**
 * Result of getSlot() - a unified acquisition result that hides the
 * distinction between cache hit, partial hit with slot copy, and new session.
 */
struct SlotAcquireResult {
  std::string sessionId;
  uint32_t slotId = 0;
  uint32_t matchedTokens = 0;
  uint32_t accumulatedThinkTokens = 0;
  bool isNewSession = false;
  std::vector<utils::BlockHashInfo> blocks;  // For token accumulator init
};

struct PrefixCacheRouterCallbacks {
  std::function<domain::MarkInFlightResult(
      const std::string& sessionId, std::function<void()>& cancelFn,
      std::optional<uint64_t> expectedKeyHash,
      const std::string* expectedResponseId)>
      tryMarkInFlight;

  std::function<std::shared_ptr<domain::Session>(const std::string& sessionId)>
      getSession;

  std::function<std::optional<uint64_t>(const std::string& sessionId)>
      getSessionHash;

  std::function<bool(const std::string& sessionId, uint64_t keyHash)>
      setSessionHash;

  std::function<bool(const std::string& sessionId,
                     const std::string& responseId)>
      setSessionResponseId;

  std::function<void()> onSessionInFlight;

  // Callbacks for getSlot() - session creation and slot management
  std::function<void(std::function<void(const domain::Session&)> onCompletion,
                     std::function<void(std::string_view)> onError,
                     trantor::EventLoop* eventLoop,
                     std::vector<utils::BlockHashInfo> initialBlockInfos,
                     std::optional<uint32_t> slotIdToCopyFrom)>
      createSession;

  std::function<uint32_t(const std::string& sessionId,
                         std::function<void()> cancelFn)>
      acquireInFlight;

  std::function<void(uint32_t slotId)> lockSlot;
  std::function<void(uint32_t slotId)> unlockSlot;

  std::function<void(const std::string& sessionId, uint32_t matchedTokens)>
      shrinkResidentPrefixToMatchedTokens;
};

class PrefixCacheRouter {
 public:
  using Candidate = domain::prefix_cache::Candidate;
  using AcquireResult = PrefixCacheAcquireResult;

  explicit PrefixCacheRouter(PrefixCacheRouterCallbacks callbacks);

  PrefixCacheRouter(const PrefixCacheRouter&) = delete;
  PrefixCacheRouter& operator=(const PrefixCacheRouter&) = delete;

  /**
   * Compute block hashes from prompt tokens.
   * Internalizes block splitting and hashing logic.
   */
  std::vector<utils::BlockHashInfo> computeBlockInfos(
      std::span<const uint32_t> promptTokenIds) const;

  std::optional<AcquireResult> tryAcquireByPrefixHash(
      const std::vector<utils::BlockHashInfo>& blockInfos,
      std::function<void()> cancelFn);

  std::optional<AcquireResult> tryAcquireByResponseId(
      const std::string& previousResponseId, std::function<void()> cancelFn);

  void registerPrefixHash(const std::string& sessionId,
                          const std::vector<utils::BlockHashInfo>& blockInfos);

  void registerResponseId(const std::string& sessionId,
                          const std::string& responseId);

  void updateResponseId(const std::string& previousResponseId,
                        const std::string& responseId);

  std::pair<uint32_t, uint32_t> computeMatchedTokens(
      const std::string& sessionId,
      const std::vector<utils::BlockHashInfo>& blockInfos);

  void clearSessionBlockThinkTokens(const std::string& sessionId);

  void onSessionClosed(const std::string& sessionId, uint64_t keyHash,
                       const std::string& responseId);

  /**
   * Unified slot acquisition - the main entry point for prefix cache routing.
   *
   * Internally handles all routing layers:
   *   1. Compute block hashes from tokens
   *   2. Response-id lookup (if previousResponseId provided)
   *   3. Prefix-hash lookup
   *   4. New session allocation (if no cache hit)
   *
   * @param promptTokenIds  Token IDs from the request prompt.
   * @param opts            Routing options (previousResponseId, responseId,
   * cancelFn).
   * @param onResolved      Callback with the result (session found or created).
   * @param onError         Callback for errors (e.g., rate limit).
   */
  void getSlot(std::span<const uint32_t> promptTokenIds, GetSlotOptions opts,
               trantor::EventLoop* eventLoop,
               std::function<void(SlotAcquireResult)> onResolved,
               std::function<void(const std::string&)> onError);

 private:
  PrefixCacheRouterCallbacks callbacks;
  domain::prefix_cache::PrefixIndex prefixIndex;
  domain::prefix_cache::ResponseIdIndex responseIdIndex;
};

}  // namespace tt::services
