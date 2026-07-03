// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "domain/prefix_cache/helpers.hpp"
#include "domain/prefix_cache/prefix_index.hpp"
#include "domain/prefix_cache/response_id_index.hpp"
#include "domain/session.hpp"
#include "domain/session_manager_structs.hpp"
#include "utils/conversation_hasher.hpp"

namespace tt::services {

struct PrefixCacheAcquireResult {
  bool sessionFound = false;
  std::string sessionId;
  uint32_t slotId = 0;
  uint32_t numberOfMatchedTokens = 0;
  uint32_t accumulatedThinkTokens = 0;
  std::vector<domain::prefix_cache::Candidate> candidatesList;
};

struct PrefixCacheRouterCallbacks {
  std::function<domain::MarkInFlightResult(const std::string& sessionId,
                                   std::function<void()>& cancelFn,
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
};

class PrefixCacheRouter {
 public:
  using Candidate = domain::prefix_cache::Candidate;
  using AcquireResult = PrefixCacheAcquireResult;

  explicit PrefixCacheRouter(PrefixCacheRouterCallbacks callbacks);

  PrefixCacheRouter(const PrefixCacheRouter&) = delete;
  PrefixCacheRouter& operator=(const PrefixCacheRouter&) = delete;

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

 private:
  PrefixCacheRouterCallbacks callbacks;
  domain::prefix_cache::PrefixIndex prefixIndex;
  domain::prefix_cache::ResponseIdIndex responseIdIndex;
};

}  // namespace tt::services
