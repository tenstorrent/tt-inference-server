// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/prefix_cache_router.hpp"

#include "config/settings.hpp"
#include "domain/block_matcher.hpp"
#include "metrics/metrics.hpp"
#include "utils/logger.hpp"

namespace tt::services {

PrefixCacheRouter::PrefixCacheRouter(SessionLease& lease) : lease(lease) {}

std::optional<PrefixCacheRouter::AcquireResult>
PrefixCacheRouter::tryAcquireByPrefixHash(
    const std::vector<utils::BlockHashInfo>& blockInfos,
    std::function<void()> cancelFn) {
  TT_LOG_DEBUG("[PrefixCacheRouter] tryAcquireByPrefixHash: blockInfos={}",
               blockInfos.size());

  if (blockInfos.empty()) {
    return std::nullopt;
  }

  const uint64_t keyHash = blockInfos.front().hash;

  const std::vector<domain::PrefixIndexEntry> entries =
      prefixIndex.getEntriesForKey(keyHash);
  std::vector<Candidate> candidates =
      domain::BlockMatcher::buildCandidates(blockInfos, entries);
  domain::BlockMatcher::sortCandidates(candidates);

  if (candidates.empty()) {
    TT_LOG_DEBUG("[PrefixCacheRouter] tryAcquireByPrefixHash: keyHash={} miss",
                 keyHash);
    return std::nullopt;
  }

  TT_LOG_INFO(
      "[PrefixCacheRouter] tryAcquireByPrefixHash: {} candidate(s) under "
      "keyHash={}, best match={} blocks",
      candidates.size(), keyHash, candidates.front().matchedBlocks);

  const float threshold = tt::config::prefixCacheHitThreshold();
  bool anyBusy = false;
  for (const auto& candidate : candidates) {
    if (!domain::BlockMatcher::passesHitThreshold(candidate, threshold)) {
      if (threshold > 0.0f) {
        const float matchPercent = (candidate.matchedBlocks * 100.0f) /
                                   static_cast<float>(candidate.sessionBlocks);
        TT_LOG_INFO(
            "[PrefixCacheRouter] Prefix cache candidate rejected: "
            "matchedBlocks={} sessionBlocks={} matchPercent={:.1f}% < "
            "threshold={:.1f}%",
            candidate.matchedBlocks, candidate.sessionBlocks, matchPercent,
            threshold);
      }
      continue;
    }

    std::optional<AcquireResult> acquired;
    bool busy = false;

    const uint32_t matchedTokens =
        domain::BlockMatcher::blocksToTokens(candidate.matchedBlocks);

    auto markResult =
        lease.tryMarkInFlight(candidate.sessionId, cancelFn, keyHash, nullptr);

    if (markResult.outcome == MarkInFlightOutcome::Stale ||
        markResult.outcome == MarkInFlightOutcome::NotFound) {
      prefixIndex.remove(candidate.sessionId, keyHash);
      continue;
    }

    if (markResult.outcome == MarkInFlightOutcome::Marked) {
      acquired =
          AcquireResult{true,          candidate.sessionId,   markResult.slotId,
                        matchedTokens, candidate.thinkTokens, {}};
    } else {
      busy = true;
    }

    if (acquired) {
      TT_LOG_INFO(
          "[PrefixCacheRouter] tryAcquireByPrefixHash: acquired sessionId={}, "
          "slotId={}, matchedTokens={}, thinkTokens={}, matchedBlocks={}",
          acquired->sessionId, acquired->slotId,
          acquired->numberOfMatchedTokens, acquired->accumulatedThinkTokens,
          candidate.matchedBlocks);
      return acquired;
    }

    anyBusy |= busy;
  }

  if (anyBusy) {
    TT_LOG_INFO(
        "[PrefixCacheRouter] tryAcquireByPrefixHash: all candidate sessions "
        "are in-flight → falling through to allocate new session");
  }

  TT_LOG_DEBUG(
      "[PrefixCacheRouter] tryAcquireByPrefixHash: no acquirable session for "
      "keyHash={}",
      keyHash);
  return AcquireResult{false, {}, 0, 0, 0, std::move(candidates)};
}

std::optional<PrefixCacheRouter::AcquireResult>
PrefixCacheRouter::tryAcquireByResponseId(const std::string& previousResponseId,
                                          std::function<void()> cancelFn) {
  if (previousResponseId.empty()) {
    return std::nullopt;
  }
  TT_LOG_DEBUG("[PrefixCacheRouter] tryAcquireByResponseId: id={}",
               previousResponseId);

  const auto sessionIdOpt = responseIdIndex.lookup(previousResponseId);
  if (!sessionIdOpt.has_value()) {
    TT_LOG_INFO(
        "[PrefixCacheRouter] tryAcquireByResponseId: id={} MISS "
        "(not found in responseIdIndex)",
        previousResponseId);
    return std::nullopt;
  }

  const std::string sessionId = *sessionIdOpt;

  std::optional<AcquireResult> acquired;
  auto markResult = lease.tryMarkInFlight(sessionId, cancelFn, std::nullopt,
                                          &previousResponseId);

  if (markResult.outcome == MarkInFlightOutcome::Stale ||
      markResult.outcome == MarkInFlightOutcome::NotFound) {
    responseIdIndex.removeIf(sessionId, previousResponseId);
    return std::nullopt;
  }

  if (markResult.outcome == MarkInFlightOutcome::Marked) {
    acquired = AcquireResult{};
    acquired->sessionFound = true;
    acquired->sessionId = sessionId;
    acquired->slotId = markResult.slotId;
  }

  if (acquired) {
    TT_LOG_INFO(
        "[PrefixCacheRouter] tryAcquireByResponseId: acquired sessionId={}, "
        "slotId={} for id={}",
        acquired->sessionId, acquired->slotId, previousResponseId);
    return acquired;
  }

  if (markResult.outcome == MarkInFlightOutcome::Busy) {
    TT_LOG_WARN(
        "[PrefixCacheRouter] tryAcquireByResponseId: session under id={} is "
        "in-flight",
        previousResponseId);
    throw SessionInFlightException();
  }

  return std::nullopt;
}

PrefixCacheResolveResult PrefixCacheRouter::tryResolve(
    const std::optional<std::string>& previousResponseId,
    const std::vector<utils::BlockHashInfo>& blockInfos,
    std::function<void()> cancelFn) {
  if (previousResponseId.has_value() && !previousResponseId->empty()) {
    auto acquired = tryAcquireByResponseId(*previousResponseId, cancelFn);
    if (acquired.has_value() && acquired->sessionFound) {
      const auto [matchedTokens, thinkTokens] =
          computeMatchedTokens(acquired->sessionId, blockInfos);
      acquired->numberOfMatchedTokens = matchedTokens;
      acquired->accumulatedThinkTokens = thinkTokens;
      return {PrefixCacheResolveOutcome::Hit, std::move(*acquired)};
    }
    return {PrefixCacheResolveOutcome::Miss, {}};
  }

  if (blockInfos.empty()) {
    return {PrefixCacheResolveOutcome::Miss, {}};
  }

  auto acquired = tryAcquireByPrefixHash(blockInfos, cancelFn);
  if (acquired.has_value() && acquired->sessionFound) {
    return {PrefixCacheResolveOutcome::Hit, std::move(*acquired)};
  }
  if (acquired.has_value()) {
    return {PrefixCacheResolveOutcome::Miss, std::move(*acquired)};
  }
  return {PrefixCacheResolveOutcome::Miss, {}};
}

void PrefixCacheRouter::commitContinuation(const ContinuationCommit& commit) {
  registerPrefixHash(commit.sessionId, commit.blocks);
  lease.shrinkResidentPrefixToMatchedTokens(commit.sessionId,
                                            commit.matchedTokens);
  if (commit.previousResponseId.has_value() && commit.responseId.has_value()) {
    updateResponseId(*commit.previousResponseId, *commit.responseId);
  }
}

void PrefixCacheRouter::registerPrefixHash(
    const std::string& sessionId,
    const std::vector<utils::BlockHashInfo>& blockInfos) {
  if (blockInfos.empty()) return;

  const uint64_t keyHash = blockInfos.front().hash;
  const uint32_t keyThinkCount = blockInfos.front().accumulatedThinkTokens;
  TT_LOG_DEBUG(
      "[PrefixCacheRouter] registerPrefixHash: sessionId={}, keyHash={}, "
      "blocks={}, keyThinkCount={}",
      sessionId, keyHash, blockInfos.size(), keyThinkCount);

  const auto oldHash = lease.getSessionHash(sessionId);
  if (!lease.setSessionHash(sessionId, keyHash)) {
    TT_LOG_WARN(
        "[PrefixCacheRouter] registerPrefixHash: sessionId={} not found",
        sessionId);
    return;
  }

  uint32_t slotId = tt::domain::INVALID_SLOT_ID;
  if (auto session = lease.getSession(sessionId)) {
    slotId = session->getSlotId();
  }

  if (oldHash.has_value() && *oldHash != 0 && *oldHash != keyHash) {
    prefixIndex.remove(sessionId, *oldHash);
  }

  prefixIndex.registerPrefixHash(sessionId, blockInfos);

  TT_LOG_INFO(
      "[PrefixCacheRouter] registerPrefixHash: registered sessionId={} under "
      "keyHash={} with {} remaining blocks",
      sessionId, keyHash, blockInfos.size() - 1);

  if (slotId != tt::domain::INVALID_SLOT_ID) {
    tt::metrics::ServerMetrics::instance().setSlotBlocks(
        slotId, static_cast<double>(blockInfos.size()));
  }
}

void PrefixCacheRouter::registerResponseId(const std::string& sessionId,
                                           const std::string& responseId) {
  if (responseId.empty()) {
    return;
  }
  TT_LOG_INFO("[PrefixCacheRouter] registerResponseId: sessionId={}, id={}",
              sessionId, responseId);

  if (!lease.setSessionResponseId(sessionId, responseId)) {
    TT_LOG_WARN("[PrefixCacheRouter] registerResponseId: sessionId={} not found",
                sessionId);
    return;
  }
  responseIdIndex.registerId(responseId, sessionId);
}

void PrefixCacheRouter::updateResponseId(
    const std::string& previousResponseId, const std::string& responseId) {
  if (previousResponseId.empty() || responseId.empty()) {
    return;
  }
  if (previousResponseId == responseId) {
    return;
  }

  const auto sessionIdOpt =
      responseIdIndex.updateId(previousResponseId, responseId);
  if (!sessionIdOpt.has_value()) {
    TT_LOG_WARN(
        "[PrefixCacheRouter] updateResponseId: previousId={} not in index",
        previousResponseId);
    return;
  }
  const std::string sessionId = *sessionIdOpt;
  TT_LOG_INFO(
      "[PrefixCacheRouter] updateResponseId: sessionId={} from id={} to id={}",
      sessionId, previousResponseId, responseId);

  if (!lease.setSessionResponseId(sessionId, responseId)) {
    TT_LOG_WARN("[PrefixCacheRouter] updateResponseId: sessionId={} not found",
                sessionId);
    return;
  }
}

std::pair<uint32_t, uint32_t> PrefixCacheRouter::computeMatchedTokens(
    const std::string& sessionId,
    const std::vector<utils::BlockHashInfo>& blockInfos) {
  if (blockInfos.empty()) {
    return {0, 0};
  }

  const std::vector<domain::PrefixIndexEntry> entries =
      prefixIndex.getEntriesForKey(blockInfos.front().hash);
  const auto [matchedBlocks, thinkTokens] =
      domain::BlockMatcher::computeMatchedBlocksForSession(sessionId,
                                                           blockInfos, entries);

  if (matchedBlocks == 0) {
    return {0, 0};
  }
  return {domain::BlockMatcher::blocksToTokens(matchedBlocks), thinkTokens};
}

void PrefixCacheRouter::clearSessionBlockThinkTokens(
    const std::string& sessionId) {
  const auto keyHash = lease.getSessionHash(sessionId);

  if (!keyHash.has_value()) {
    TT_LOG_WARN(
        "[PrefixCacheRouter] clearSessionBlockThinkTokens: sessionId={} not "
        "found",
        sessionId);
    return;
  }

  if (*keyHash == 0) {
    return;
  }

  prefixIndex.clearThinkTokens(sessionId, *keyHash);
  TT_LOG_INFO(
      "[PrefixCacheRouter] clearSessionBlockThinkTokens: reset think tokens "
      "for sessionId={}",
      sessionId);
}

void PrefixCacheRouter::onSessionClosed(const std::string& sessionId,
                                        uint64_t keyHash,
                                        const std::string& responseId) {
  prefixIndex.remove(sessionId, keyHash);
  responseIdIndex.removeIf(sessionId, responseId);
}

}  // namespace tt::services
