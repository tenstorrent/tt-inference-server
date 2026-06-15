// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/session_resolution.hpp"

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace tt::services::session_resolution {

std::optional<SlotCopyPlan> prepareSlotCopy(
    SessionManager& sessionManager,
    const std::vector<SessionManager::Candidate>& candidates, uint32_t taskId,
    std::string_view logPrefix) {
  if (candidates.empty()) {
    return std::nullopt;
  }

  auto copyCandidate = sessionManager.findASlotToCopyFrom(candidates);
  if (!copyCandidate.has_value()) {
    return std::nullopt;
  }

  uint32_t sourceSlot =
      sessionManager.getSlotIdBySessionId(copyCandidate->sessionId);
  if (sourceSlot == tt::domain::INVALID_SLOT_ID) {
    return std::nullopt;
  }

  sessionManager.lockSlot(sourceSlot);

  const size_t firstBlockSize = tt::config::kvCacheFirstBlockSize();
  const size_t blockSize = tt::config::kvCacheBlockSize();
  const uint32_t matchedTokens = static_cast<uint32_t>(
      firstBlockSize + (copyCandidate->matchedBlocks > 1
                            ? (copyCandidate->matchedBlocks - 1) * blockSize
                            : 0));

  TT_LOG_INFO(
      "{} Found slot to copy from: slotId={} matchedTokens={} for taskId={}",
      logPrefix, sourceSlot, matchedTokens, taskId);

  return SlotCopyPlan{.slotToCopyFrom = sourceSlot,
                      .matchedTokens = matchedTokens};
}

}  // namespace tt::services::session_resolution
