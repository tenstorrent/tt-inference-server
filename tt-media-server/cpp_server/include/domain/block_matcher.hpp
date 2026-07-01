// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "domain/prefix_index.hpp"
#include "utils/conversation_hasher.hpp"

namespace tt::domain {

struct ConsecutiveMatch {
  std::size_t matchedRemainingBlocks = 0;
  uint32_t lastMatchedThinkTokens = 0;
};

class BlockMatcher {
 public:
  static std::list<RemainingBlockInfo> buildCallerRemaining(
      const std::vector<tt::utils::BlockHashInfo>& blockInfos);

  static ConsecutiveMatch countConsecutiveRemainingMatch(
      const std::list<RemainingBlockInfo>& callerRemaining,
      const std::list<RemainingBlockInfo>& entryRemaining,
      std::uint32_t keyBlockThinkTokens);

  static std::vector<Candidate> buildCandidates(
      const std::vector<utils::BlockHashInfo>& blockInfos,
      const std::vector<PrefixIndexEntry>& entries);

  static void sortCandidates(std::vector<Candidate>& candidates);
  static bool passesHitThreshold(const Candidate& candidate, float threshold);
  static std::uint32_t blocksToTokens(std::size_t matchedBlocks);
  static std::uint32_t tokensToBlocks(std::uint32_t tokens);

  static std::pair<std::size_t, std::uint32_t> computeMatchedBlocksForSession(
      const std::string& sessionId,
      const std::vector<utils::BlockHashInfo>& blockInfos,
      const std::vector<PrefixIndexEntry>& entries);

  static std::optional<Candidate> findSlotToCopyFrom(
      const std::vector<Candidate>& candidates,
      std::function<uint32_t(const std::string& sessionId)> getCommittedBlocks);
};

}  // namespace tt::domain
