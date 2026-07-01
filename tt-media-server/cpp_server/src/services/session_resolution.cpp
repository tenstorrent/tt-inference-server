// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/session_resolution.hpp"

#include "domain/block_matcher.hpp"
#include "utils/logger.hpp"

namespace tt::services::session_resolution {

std::optional<SlotCopyPlan> prepareSlotCopy(
    SessionManager& sessionManager,
    const std::vector<SessionManager::Candidate>& candidates, uint32_t taskId,
    std::string_view logPrefix) {
  if (candidates.empty()) {
    return std::nullopt;
  }

  auto copyCandidate = domain::BlockMatcher::findSlotToCopyFrom(
      candidates, [&sessionManager](const std::string& sessionId) {
        uint32_t committedBlocks = 0;
        if (auto session = sessionManager.getSession(sessionId)) {
          committedBlocks = session->committedBlocks();
        }
        return committedBlocks;
      });
  if (!copyCandidate.has_value()) {
    return std::nullopt;
  }

  uint32_t sourceSlot =
      sessionManager.getSlotIdBySessionId(copyCandidate->sessionId);
  if (sourceSlot == tt::domain::INVALID_SLOT_ID) {
    return std::nullopt;
  }

  sessionManager.lockSlot(sourceSlot);

  const uint32_t matchedTokens =
      domain::BlockMatcher::blocksToTokens(copyCandidate->matchedBlocks);

  TT_LOG_INFO(
      "{} Found slot to copy from: slotId={} matchedTokens={} for taskId={}",
      logPrefix, sourceSlot, matchedTokens, taskId);

  return SlotCopyPlan{.slotToCopyFrom = sourceSlot,
                      .matchedTokens = matchedTokens};
}

}  // namespace tt::services::session_resolution
