// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "domain/prefix_index.hpp"
#include "domain/response_id_index.hpp"
#include "services/session_lease.hpp"
#include "utils/conversation_hasher.hpp"

namespace tt::services {

struct PrefixCacheAcquireResult {
  bool sessionFound = false;
  std::string sessionId;
  uint32_t slotId = 0;
  uint32_t numberOfMatchedTokens = 0;
  uint32_t accumulatedThinkTokens = 0;
  std::vector<domain::Candidate> candidatesList;
};

enum class PrefixCacheResolveOutcome {
  Hit,
  Miss,
};

struct PrefixCacheResolveResult {
  PrefixCacheResolveOutcome outcome = PrefixCacheResolveOutcome::Miss;
  PrefixCacheAcquireResult acquired;
};

struct ContinuationCommit {
  std::string sessionId;
  std::vector<utils::BlockHashInfo> blocks;
  uint32_t matchedTokens = 0;
  std::optional<std::string> previousResponseId;
  std::optional<std::string> responseId;
};

class PrefixCacheRouter {
 public:
  using Candidate = domain::Candidate;
  using AcquireResult = PrefixCacheAcquireResult;

  explicit PrefixCacheRouter(SessionLease& lease);

  PrefixCacheRouter(const PrefixCacheRouter&) = delete;
  PrefixCacheRouter& operator=(const PrefixCacheRouter&) = delete;

  std::optional<AcquireResult> tryAcquireByPrefixHash(
      const std::vector<utils::BlockHashInfo>& blockInfos,
      std::function<void()> cancelFn);

  std::optional<AcquireResult> tryAcquireByResponseId(
      const std::string& previousResponseId, std::function<void()> cancelFn);

  /**
   * Unified routing: response-id lookup first (when set), else prefix-hash
   * lookup. Throws SessionInFlightException when a response-id session is busy.
   */
  PrefixCacheResolveResult tryResolve(
      const std::optional<std::string>& previousResponseId,
      const std::vector<utils::BlockHashInfo>& blockInfos,
      std::function<void()> cancelFn);

  /**
   * Post-hit index and session updates after the caller applies the delta
   * prompt. Registers the prefix, shrinks resident KV to the matched prefix,
   * and re-keys the response id when both ids are provided.
   */
  void commitContinuation(const ContinuationCommit& commit);

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

 private:
  SessionLease& lease;
  domain::PrefixIndex prefixIndex;
  domain::ResponseIdIndex responseIdIndex;
};

}  // namespace tt::services
