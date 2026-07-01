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

  void registerPrefixHash(const std::string& sessionId,
                          const std::vector<utils::BlockHashInfo>& blockInfos);

  void initResponseId(const std::string& sessionId,
                      const std::string& responseId);

  void registerResponseId(const std::string& previousResponseId,
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
