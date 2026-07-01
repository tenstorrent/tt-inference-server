// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/prefix_cache_router.hpp"

#include "config/settings.hpp"
#include "metrics/metrics.hpp"
#include "utils/logger.hpp"

namespace tt::services {

PrefixCacheRouter::PrefixCacheRouter(SessionLease& lease) : lease_(lease) {}

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
  const size_t firstBlockTokens = tt::config::kvCacheFirstBlockSize();
  const size_t blockTokens = tt::config::kvCacheBlockSize();

  std::vector<Candidate> candidates = prefixIndex.findCandidates(blockInfos);

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
    if (threshold > 0.0f) {
      float matchPercent =
          (candidate.matchedBlocks * 100.0f) / candidate.sessionBlocks;
      if (matchPercent < threshold) {
        TT_LOG_INFO(
            "[PrefixCacheRouter] Prefix cache candidate rejected: "
            "matchedBlocks={} sessionBlocks={} matchPercent={:.1f}% < "
            "threshold={:.1f}%",
            candidate.matchedBlocks, candidate.sessionBlocks, matchPercent,
            threshold);
        continue;
      }
    }

    std::optional<AcquireResult> acquired;
    bool busy = false;

    uint32_t matchedTokens = static_cast<uint32_t>(
        firstBlockTokens + (candidate.matchedBlocks - 1) * blockTokens);

    auto markResult =
        lease_.tryMarkInFlight(candidate.sessionId, cancelFn, keyHash, nullptr);

    if (markResult.outcome == MarkInFlightOutcome::Stale ||
        markResult.outcome == MarkInFlightOutcome::NotFound) {
      prefixIndex.remove(candidate.sessionId, keyHash);
      continue;
    }

    if (markResult.outcome == MarkInFlightOutcome::Marked) {
      acquired = AcquireResult{
          true,          candidate.sessionId,   markResult.slotId,
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
  auto markResult = lease_.tryMarkInFlight(sessionId, cancelFn, std::nullopt,
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

  const auto oldHash = lease_.getSessionHash(sessionId);
  if (!lease_.setSessionHash(sessionId, keyHash)) {
    TT_LOG_WARN("[PrefixCacheRouter] registerPrefixHash: sessionId={} not found",
                sessionId);
    return;
  }

  uint32_t slotId = tt::domain::INVALID_SLOT_ID;
  if (auto session = lease_.getSession(sessionId)) {
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

void PrefixCacheRouter::initResponseId(const std::string& sessionId,
                                       const std::string& responseId) {
  if (responseId.empty()) {
    return;
  }
  TT_LOG_INFO("[PrefixCacheRouter] initResponseId: sessionId={}, id={}",
              sessionId, responseId);

  if (!lease_.setSessionResponseId(sessionId, responseId)) {
    TT_LOG_WARN("[PrefixCacheRouter] initResponseId: sessionId={} not found",
                sessionId);
    return;
  }
  responseIdIndex.init(responseId, sessionId);
}

void PrefixCacheRouter::registerResponseId(
    const std::string& previousResponseId, const std::string& responseId) {
  if (previousResponseId.empty() || responseId.empty()) {
    return;
  }
  if (previousResponseId == responseId) {
    return;
  }

  const auto sessionIdOpt =
      responseIdIndex.rekey(previousResponseId, responseId);
  if (!sessionIdOpt.has_value()) {
    TT_LOG_WARN(
        "[PrefixCacheRouter] registerResponseId: previousId={} not in index",
        previousResponseId);
    return;
  }
  const std::string sessionId = *sessionIdOpt;
  TT_LOG_INFO(
      "[PrefixCacheRouter] registerResponseId: re-keying sessionId={} from "
      "id={} to id={}",
      sessionId, previousResponseId, responseId);

  if (!lease_.setSessionResponseId(sessionId, responseId)) {
    TT_LOG_WARN(
        "[PrefixCacheRouter] registerResponseId: sessionId={} not found",
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

  const size_t firstBlockTokens = tt::config::kvCacheFirstBlockSize();
  const size_t blockTokens = tt::config::kvCacheBlockSize();

  const auto [matchedBlocks, thinkTokens] =
      prefixIndex.computeMatchedBlocks(sessionId, blockInfos);

  if (matchedBlocks == 0) {
    return {0, 0};
  }
  uint32_t matchedTokens = static_cast<uint32_t>(
      firstBlockTokens + (matchedBlocks - 1) * blockTokens);
  return {matchedTokens, thinkTokens};
}

std::optional<PrefixCacheRouter::Candidate>
PrefixCacheRouter::findASlotToCopyFrom(
    const std::vector<Candidate>& candidates) {
  const size_t firstBlockSize = tt::config::kvCacheFirstBlockSize();
  const size_t blockSize = tt::config::kvCacheBlockSize();
  const size_t minTokens = tt::config::minTokensToCopy();

  for (const auto& candidate : candidates) {
    if (candidate.matchedBlocks == 0) continue;

    uint32_t residentBlocks = 0;
    if (auto session = lease_.getSession(candidate.sessionId)) {
      residentBlocks = session->committedBlocks();
    }

    const size_t usableBlocks =
        std::min<size_t>(candidate.matchedBlocks, residentBlocks);
    if (usableBlocks == 0) {
      TT_LOG_DEBUG(
          "[PrefixCacheRouter] findASlotToCopyFrom: candidate sessionId={} "
          "has no resident KV (matchedBlocks={}, committedBlocks={}), skipping",
          candidate.sessionId, candidate.matchedBlocks, residentBlocks);
      continue;
    }

    const size_t usableTokens =
        firstBlockSize +
        (usableBlocks > 1 ? (usableBlocks - 1) * blockSize : 0);

    if (usableTokens >= minTokens) {
      TT_LOG_DEBUG(
          "[PrefixCacheRouter] findASlotToCopyFrom: candidate sessionId={} "
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
      "[PrefixCacheRouter] findASlotToCopyFrom: no candidate meets threshold "
      "(minTokensToCopy={}, candidates={})",
      minTokens, candidates.size());
  return std::nullopt;
}

void PrefixCacheRouter::clearSessionBlockThinkTokens(
    const std::string& sessionId) {
  const auto keyHash = lease_.getSessionHash(sessionId);

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
