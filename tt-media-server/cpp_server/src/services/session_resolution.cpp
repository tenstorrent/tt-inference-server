// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/session_resolution.hpp"

#include "config/settings.hpp"
#include "domain/prefix_cache/block_matcher.hpp"
#include "utils/logger.hpp"

namespace tt::services::session_resolution {

std::optional<SlotCopyPlan> prepareSlotCopy(
    SessionManager& sessionManager,
    const std::vector<SessionManager::Candidate>& candidates, uint32_t taskId,
    std::string_view logPrefix) {
  if (candidates.empty()) {
    return std::nullopt;
  }

  const auto mode = tt::config::llmMode();
  auto copyCandidate = domain::prefix_cache::BlockMatcher::findSlotToCopyFrom(
      candidates,
      [&sessionManager](const std::string& sessionId) {
        return sessionManager.getCommittedBlocks(sessionId);
      },
      [&sessionManager, mode](const std::string& sessionId) -> bool {
        // copy is eligible if the source session is not in-flight (prefill-only
        // mode) or not being generated (decode mode)
        // or session is not in flight anymore
        auto session = sessionManager.getSession(sessionId);
        if (!session) return false;
        if (mode == tt::config::LLMMode::PREFILL_ONLY) {
          return !session->isInFlight();
        }
        return !session->isBeingGenerated() || !session->isInFlight();
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
      domain::prefix_cache::BlockMatcher::blocksToTokens(
          copyCandidate->matchedBlocks);

  TT_LOG_INFO(
      "{} Found slot to copy from: slotId={} matchedTokens={} for taskId={}",
      logPrefix, sourceSlot, matchedTokens, taskId);

  return SlotCopyPlan{.slotToCopyFrom = sourceSlot,
                      .matchedTokens = matchedTokens};
}

}  // namespace tt::services::session_resolution
