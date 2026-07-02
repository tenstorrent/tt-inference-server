// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/prefix_cache/block_matcher.hpp"

#include <algorithm>
#include <list>

#include "config/settings.hpp"
#include "domain/prefix_cache/helpers.hpp"
#include "utils/logger.hpp"

namespace tt::domain::prefix_cache {

std::list<RemainingBlockInfo> BlockMatcher::buildCallerRemaining(
    const std::vector<tt::utils::BlockHashInfo>& blockInfos) {
  std::list<RemainingBlockInfo> remaining;
  for (std::size_t i = 1; i < blockInfos.size(); ++i) {
    remaining.push_back(
        {blockInfos[i].hash, blockInfos[i].accumulatedThinkTokens});
  }
  return remaining;
}

MatchedTokens BlockMatcher::countMatchedTokens(
    const std::list<RemainingBlockInfo>& callerRemaining,
    const std::list<RemainingBlockInfo>& entryRemaining,
    std::uint32_t keyBlockThinkTokens) {
  MatchedTokens result;
  result.matchedThinkTokens = keyBlockThinkTokens;

  auto callerIt = callerRemaining.begin();
  auto entryIt = entryRemaining.begin();
  while (callerIt != callerRemaining.end() && entryIt != entryRemaining.end() &&
         callerIt->hash == entryIt->hash) {
    result.matchedThinkTokens = entryIt->accumulatedThinkTokens;
    ++result.matchedBlocks;
    ++callerIt;
    ++entryIt;
  }
  return result;
}

std::vector<Candidate> BlockMatcher::buildCandidates(
    const std::vector<utils::BlockHashInfo>& blockInfos,
    const std::vector<PrefixIndexEntry>& entries) {
  std::vector<Candidate> candidates;
  if (blockInfos.empty()) {
    return candidates;
  }

  const std::list<RemainingBlockInfo> callerRemaining =
      buildCallerRemaining(blockInfos);

  for (const auto& entry : entries) {
    const MatchedTokens match = countMatchedTokens(
        callerRemaining, entry.remainingBlocks, entry.keyBlockThinkTokens);
    const std::size_t totalMatched = 1 + match.matchedBlocks;
    const std::size_t sessionTotal = 1 + entry.remainingBlocks.size();

    for (const auto& sessionId : entry.sessionIds) {
      candidates.push_back(
          {sessionId, totalMatched, sessionTotal, match.matchedThinkTokens});
    }
  }
  return candidates;
}

void BlockMatcher::sortCandidates(std::vector<Candidate>& candidates) {
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.matchedBlocks > b.matchedBlocks;
            });
}

bool BlockMatcher::passesHitThreshold(const Candidate& candidate) {
  const float threshold = tt::config::prefixCacheHitThreshold();

  const float matchPercent = (candidate.matchedBlocks * 100.0f) /
                             static_cast<float>(candidate.sessionBlocks);
  bool passesThreshold = matchPercent >= threshold;
  if (!passesThreshold && threshold > 0.0f) {
    TT_LOG_INFO(
        "[BlockMatcher] Prefix cache candidate rejected: "
        "matchedBlocks={} sessionBlocks={} matchPercent={:.1f}% < "
        "threshold={:.1f}%",
        candidate.matchedBlocks, candidate.sessionBlocks, matchPercent,
        threshold);
  }
  return passesThreshold;
}

std::uint32_t BlockMatcher::blocksToTokens(std::size_t matchedBlocks) {
  if (matchedBlocks == 0) {
    return 0;
  }

  const std::size_t firstBlockSize = tt::config::kvCacheFirstBlockSize();
  const std::size_t blockSize = tt::config::kvCacheBlockSize();
  return static_cast<std::uint32_t>(firstBlockSize +
                                    (matchedBlocks - 1) * blockSize);
}

std::uint32_t BlockMatcher::tokensToBlocks(std::uint32_t tokens) {
  if (tokens == 0) {
    return 0;
  }

  const std::size_t firstBlockSize = tt::config::kvCacheFirstBlockSize();
  const std::size_t blockSize = tt::config::kvCacheBlockSize();
  if (tokens <= firstBlockSize || blockSize == 0) {
    return 1;
  }
  return static_cast<std::uint32_t>(
      1 + (tokens - firstBlockSize + blockSize - 1) / blockSize);
}

std::pair<std::size_t, std::uint32_t>
BlockMatcher::computeMatchedBlocksForSession(
    const std::string& sessionId,
    const std::vector<utils::BlockHashInfo>& blockInfos,
    const std::vector<PrefixIndexEntry>& entries) {
  if (blockInfos.empty()) {
    return {0, 0};
  }

  const std::list<RemainingBlockInfo> callerRemaining =
      buildCallerRemaining(blockInfos);

  std::size_t bestMatchedBlocks = 0;
  std::uint32_t bestThinkTokens = 0;
  for (const auto& entry : entries) {
    const bool hasSession =
        std::find(entry.sessionIds.begin(), entry.sessionIds.end(),
                  sessionId) != entry.sessionIds.end();
    if (!hasSession) {
      continue;
    }

    const MatchedTokens match = countMatchedTokens(
        callerRemaining, entry.remainingBlocks, entry.keyBlockThinkTokens);
    const std::size_t totalMatched = 1 + match.matchedBlocks;
    if (totalMatched > bestMatchedBlocks) {
      bestMatchedBlocks = totalMatched;
      bestThinkTokens = match.matchedThinkTokens;
    }
  }
  return {bestMatchedBlocks, bestThinkTokens};
}

std::optional<Candidate> BlockMatcher::findSlotToCopyFrom(
    const std::vector<Candidate>& candidates,
    std::function<uint32_t(const std::string& sessionId)> getCommittedBlocks) {
  const std::size_t minTokens = tt::config::minTokensToCopy();

  for (const auto& candidate : candidates) {
    if (candidate.matchedBlocks == 0) {
      continue;
    }

    const uint32_t residentBlocks = getCommittedBlocks(candidate.sessionId);
    const std::size_t usableBlocks =
        std::min<std::size_t>(candidate.matchedBlocks, residentBlocks);
    if (usableBlocks == 0) {
      TT_LOG_DEBUG(
          "[BlockMatcher] findSlotToCopyFrom: candidate sessionId={} has no "
          "resident KV (matchedBlocks={}, committedBlocks={}), skipping",
          candidate.sessionId, candidate.matchedBlocks, residentBlocks);
      continue;
    }

    const std::uint32_t usableTokens = blocksToTokens(usableBlocks);
    if (usableTokens >= minTokens) {
      TT_LOG_DEBUG(
          "[BlockMatcher] findSlotToCopyFrom: candidate sessionId={} "
          "matchedBlocks={} committedBlocks={} usableBlocks={} usableTokens={} "
          ">= minTokensToCopy={}",
          candidate.sessionId, candidate.matchedBlocks, residentBlocks,
          usableBlocks, usableTokens, minTokens);
      Candidate capped = candidate;
      capped.matchedBlocks = usableBlocks;
      return capped;
    }
  }

  TT_LOG_DEBUG(
      "[BlockMatcher] findSlotToCopyFrom: no candidate meets threshold "
      "(minTokensToCopy={}, candidates={})",
      minTokens, candidates.size());
  return std::nullopt;
}

}  // namespace tt::domain::prefix_cache
