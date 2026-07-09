// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/prefix_cache_router.hpp"

#include "config/settings.hpp"
#include "domain/prefix_cache/block_matcher.hpp"
#include "metrics/metrics.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::services {

PrefixCacheRouter::PrefixCacheRouter(PrefixCacheRouterCallbacks callbacks)
    : callbacks(std::move(callbacks)) {}

std::vector<utils::BlockHashInfo> PrefixCacheRouter::computeBlockInfos(
    std::span<const uint32_t> promptTokenIds) const {
  if (promptTokenIds.empty()) {
    return {};
  }
  auto [thinkStart, thinkEnd] = utils::tokenizers::thinkTokenIds();
  return utils::getPrefixCacheHashesByBlocksWithThinking(promptTokenIds,
                                                         thinkStart, thinkEnd);
}

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

  const std::vector<domain::prefix_cache::PrefixIndexEntry> entries =
      prefixIndex.getEntriesForKey(keyHash);
  std::vector<Candidate> candidates =
      domain::prefix_cache::BlockMatcher::buildCandidates(blockInfos, entries);
  domain::prefix_cache::BlockMatcher::sortCandidates(candidates);

  if (candidates.empty()) {
    TT_LOG_DEBUG("[PrefixCacheRouter] tryAcquireByPrefixHash: keyHash={} miss",
                 keyHash);
    return std::nullopt;
  }

  TT_LOG_INFO(
      "[PrefixCacheRouter] tryAcquireByPrefixHash: {} candidate(s) under "
      "keyHash={}, best match={} blocks",
      candidates.size(), keyHash, candidates.front().matchedBlocks);

  bool anyBusy = false;
  for (const auto& candidate : candidates) {
    if (!domain::prefix_cache::BlockMatcher::passesHitThreshold(candidate)) {
      continue;
    }

    std::optional<AcquireResult> acquired;
    bool busy = false;

    const uint32_t matchedTokens =
        domain::prefix_cache::BlockMatcher::blocksToTokens(
            candidate.matchedBlocks);

    auto markResult = callbacks.tryMarkInFlight(candidate.sessionId, cancelFn,
                                                keyHash, nullptr);

    if (markResult.outcome == domain::MarkInFlightOutcome::Stale ||
        markResult.outcome == domain::MarkInFlightOutcome::NotFound) {
      prefixIndex.remove(candidate.sessionId, keyHash);
      continue;
    }

    if (markResult.outcome == domain::MarkInFlightOutcome::Marked) {
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
  auto markResult = callbacks.tryMarkInFlight(sessionId, cancelFn, std::nullopt,
                                              &previousResponseId);

  if (markResult.outcome == domain::MarkInFlightOutcome::Stale ||
      markResult.outcome == domain::MarkInFlightOutcome::NotFound) {
    responseIdIndex.removeIf(sessionId, previousResponseId);
    return std::nullopt;
  }

  if (markResult.outcome == domain::MarkInFlightOutcome::Marked) {
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

  if (markResult.outcome == domain::MarkInFlightOutcome::Busy) {
    TT_LOG_WARN(
        "[PrefixCacheRouter] tryAcquireByResponseId: session under id={} is "
        "in-flight",
        previousResponseId);
    callbacks.onSessionInFlight();
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

  const auto oldHash = callbacks.getSessionHash(sessionId);
  if (!callbacks.setSessionHash(sessionId, keyHash)) {
    TT_LOG_WARN(
        "[PrefixCacheRouter] registerPrefixHash: sessionId={} not found",
        sessionId);
    return;
  }

  uint32_t slotId = tt::domain::INVALID_SLOT_ID;
  if (auto session = callbacks.getSession(sessionId)) {
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

  if (!callbacks.setSessionResponseId(sessionId, responseId)) {
    TT_LOG_WARN(
        "[PrefixCacheRouter] registerResponseId: sessionId={} not found",
        sessionId);
    return;
  }
  responseIdIndex.registerId(responseId, sessionId);
}

void PrefixCacheRouter::updateResponseId(const std::string& previousResponseId,
                                         const std::string& responseId) {
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

  if (!callbacks.setSessionResponseId(sessionId, responseId)) {
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

  const std::vector<domain::prefix_cache::PrefixIndexEntry> entries =
      prefixIndex.getEntriesForKey(blockInfos.front().hash);
  const auto [matchedBlocks, thinkTokens] =
      domain::prefix_cache::BlockMatcher::computeMatchedBlocksForSession(
          sessionId, blockInfos, entries);

  if (matchedBlocks == 0) {
    return {0, 0};
  }
  return {domain::prefix_cache::BlockMatcher::blocksToTokens(matchedBlocks),
          thinkTokens};
}

void PrefixCacheRouter::clearSessionBlockThinkTokens(
    const std::string& sessionId) {
  const auto keyHash = callbacks.getSessionHash(sessionId);

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

void PrefixCacheRouter::getSlot(
    std::span<const uint32_t> promptTokenIds, GetSlotOptions opts,
    trantor::EventLoop* eventLoop,
    std::function<void(SlotAcquireResult)> onResolved,
    std::function<void(const std::string&)> onError) {
  auto attachPrefixRegistrar = [this](SlotAcquireResult& result) {
    const std::string sessionId = result.sessionId;
    result.registerPrefixBlocks =
        [this, sessionId](const std::vector<utils::BlockHashInfo>& blocks) {
          registerPrefixHash(sessionId, blocks);
        };
  };

  // Step 1: Compute block hashes from tokens (or use precomputed blocks).
  std::vector<utils::BlockHashInfo> blocks;
  if (opts.precomputedBlocks.has_value()) {
    blocks = std::move(*opts.precomputedBlocks);
    opts.precomputedBlocks.reset();
  } else {
    blocks = computeBlockInfos(promptTokenIds);
  }
  TT_LOG_INFO("[PrefixCacheRouter::getSlot] tokens={} blocks={}",
              promptTokenIds.size(), blocks.size());

  // Step 2: Try response-id path first (if provided)
  const bool useResponseId =
      opts.previousResponseId.has_value() && !opts.previousResponseId->empty();

  if (useResponseId) {
    auto acquired =
        tryAcquireByResponseId(*opts.previousResponseId, opts.cancelFn);

    if (acquired.has_value()) {
      TT_LOG_INFO(
          "[PrefixCacheRouter::getSlot] Response-id HIT prevId={} sessionId={} "
          "slotId={}",
          *opts.previousResponseId, acquired->sessionId, acquired->slotId);

      auto [matchedTokens, thinkTokens] =
          computeMatchedTokens(acquired->sessionId, blocks);

      registerPrefixHash(acquired->sessionId, blocks);
      callbacks.shrinkResidentPrefixToMatchedTokens(acquired->sessionId,
                                                    matchedTokens);

      if (opts.responseId.has_value()) {
        updateResponseId(*opts.previousResponseId, *opts.responseId);
      }

      SlotAcquireResult result;
      result.sessionId = acquired->sessionId;
      result.slotId = acquired->slotId;
      result.matchedTokens = matchedTokens;
      result.accumulatedThinkTokens = thinkTokens;
      result.isNewSession = false;
      result.blocks = std::move(blocks);
      attachPrefixRegistrar(result);

      onResolved(std::move(result));
      return;
    }

    TT_LOG_INFO(
        "[PrefixCacheRouter::getSlot] Response-id MISS prevId={} → trying "
        "prefix-hash",
        *opts.previousResponseId);
  }

  // Step 3: Try prefix-hash lookup
  std::optional<AcquireResult> acquired;
  if (!blocks.empty()) {
    acquired = tryAcquireByPrefixHash(blocks, opts.cancelFn);

    if (acquired.has_value() && acquired->sessionFound) {
      TT_LOG_INFO(
          "[PrefixCacheRouter::getSlot] Prefix-cache HIT sessionId={} "
          "slotId={} "
          "matchedTokens={} thinkTokens={}",
          acquired->sessionId, acquired->slotId,
          acquired->numberOfMatchedTokens, acquired->accumulatedThinkTokens);

      registerPrefixHash(acquired->sessionId, blocks);
      callbacks.shrinkResidentPrefixToMatchedTokens(
          acquired->sessionId, acquired->numberOfMatchedTokens);

      SlotAcquireResult result;
      result.sessionId = acquired->sessionId;
      result.slotId = acquired->slotId;
      result.matchedTokens = acquired->numberOfMatchedTokens;
      result.accumulatedThinkTokens = acquired->accumulatedThinkTokens;
      result.isNewSession = false;
      result.blocks = std::move(blocks);
      attachPrefixRegistrar(result);

      onResolved(std::move(result));
      return;
    }

    TT_LOG_INFO(
        "[PrefixCacheRouter::getSlot] Prefix-cache MISS blocks={} → "
        "allocating new session",
        blocks.size());
  }

  // Step 4: Allocate new session
  // Check if we have a candidate slot to copy from
  std::optional<uint32_t> slotToCopyFrom;
  uint32_t copyMatchedTokens = 0;
  if (acquired.has_value() && !acquired->candidatesList.empty()) {
    const auto& best = acquired->candidatesList.front();
    if (domain::prefix_cache::BlockMatcher::passesHitThreshold(best)) {
      auto session = callbacks.getSession(best.sessionId);
      if (session) {
        slotToCopyFrom = session->getSlotId();
        copyMatchedTokens = domain::prefix_cache::BlockMatcher::blocksToTokens(
            best.matchedBlocks);
        callbacks.lockSlot(*slotToCopyFrom);
        TT_LOG_INFO(
            "[PrefixCacheRouter::getSlot] Will copy from slotId={} "
            "matchedTokens={}",
            *slotToCopyFrom, copyMatchedTokens);
      }
    }
  }

  callbacks.createSession(
      [this, blocks = std::move(blocks), opts = std::move(opts), onResolved,
       slotToCopyFrom, copyMatchedTokens,
       attachPrefixRegistrar](const domain::Session& session) mutable {
        // Unlock source slot now that allocation is complete
        if (slotToCopyFrom.has_value()) {
          callbacks.unlockSlot(*slotToCopyFrom);
        }

        auto slotId =
            callbacks.acquireInFlight(session.getSessionId(), opts.cancelFn);

        if (opts.responseId.has_value()) {
          registerResponseId(session.getSessionId(), *opts.responseId);
        }

        registerPrefixHash(session.getSessionId(), blocks);

        SlotAcquireResult result;
        result.sessionId = session.getSessionId();
        result.slotId = slotId;
        result.matchedTokens = copyMatchedTokens;
        result.accumulatedThinkTokens = 0;
        result.isNewSession = (copyMatchedTokens == 0);
        result.blocks = std::move(blocks);
        attachPrefixRegistrar(result);

        onResolved(std::move(result));
      },
      [onError, slotToCopyFrom, this](std::string_view errorMessage) {
        if (slotToCopyFrom.has_value()) {
          callbacks.unlockSlot(*slotToCopyFrom);
        }
        onError(std::string(errorMessage));
      },
      eventLoop, {}, slotToCopyFrom);
}

}  // namespace tt::services
