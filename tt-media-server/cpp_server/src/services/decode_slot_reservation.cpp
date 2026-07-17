// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/decode_slot_reservation.hpp"

#include "metrics/metrics.hpp"
#include "services/session_resolution.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/logger.hpp"

namespace tt::services::decode_slot_reservation {

namespace {

constexpr std::string_view kLogPrefix = "[DecodeSlotReservation]";

void setDecodePositionFields(DecodeDestinationSlot& slot, uint32_t kvPositionId,
                             int accumulatedThinkTokens, bool continuation) {
  slot.continuation = continuation;
  slot.accumulatedThinkTokens = accumulatedThinkTokens;
  if (continuation) {
    slot.decodePositionId = static_cast<int>(kvPositionId + 1);
    slot.decodeSkipTokens = slot.decodePositionId - accumulatedThinkTokens;
  } else {
    slot.decodePositionId = 0;
    slot.decodeSkipTokens = 0;
  }
}

DecodeDestinationSlot fromAcquiredPrefix(
    const SessionManager::AcquiredSession& acquired) {
  DecodeDestinationSlot slot;
  slot.slotId = acquired.slotId;
  slot.sessionId = acquired.sessionId;
  const uint32_t deltaMatchedTokens = acquired.numberOfMatchedTokens > 0
                                          ? acquired.numberOfMatchedTokens - 1
                                          : 0;
  const uint32_t kvPositionId =
      deltaMatchedTokens + acquired.accumulatedThinkTokens;
  setDecodePositionFields(slot, kvPositionId,
                          static_cast<int>(acquired.accumulatedThinkTokens),
                          true);
  return slot;
}

}  // namespace

void resolveDecodeDestinationSlot(
    SessionManager& sessionManager, const ResolveInput& input,
    trantor::EventLoop* eventLoop,
    std::function<void(DecodeDestinationSlot)> onResolved,
    std::function<void(std::string_view)> onError,
    std::function<void()> cancelFn) {
  auto blockInfos = utils::hashesToBlockInfos(input.registrationHashes);

  const bool useResponseId = input.previousResponseId.has_value() &&
                             !input.previousResponseId->empty();
  if (useResponseId) {
    try {
      auto acquired = sessionManager.tryAcquireByResponseId(
          *input.previousResponseId, cancelFn);
      if (acquired.has_value()) {
        tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(true);
        auto [matchedTokens, thinkTokens] = sessionManager.computeMatchedTokens(
            acquired->sessionId, blockInfos);
        sessionManager.registerPrefixHash(acquired->sessionId, blockInfos);

        DecodeDestinationSlot slot;
        slot.slotId = acquired->slotId;
        slot.sessionId = acquired->sessionId;
        const uint32_t kvPositionId = matchedTokens - 1 + thinkTokens;
        setDecodePositionFields(slot, kvPositionId,
                                static_cast<int>(thinkTokens), true);

        TT_LOG_INFO(
            "{} taskId={} response-id HIT sessionId={} slotId={} "
            "decodePositionId={}",
            kLogPrefix, input.taskId, slot.sessionId, slot.slotId,
            slot.decodePositionId);
        onResolved(std::move(slot));
        return;
      }

      tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(false);
      TT_LOG_INFO(
          "{} taskId={} response-id MISS prevId={} → allocating new session",
          kLogPrefix, input.taskId, *input.previousResponseId);
    } catch (const SessionInFlightException& e) {
      TT_LOG_WARN("{} taskId={} response-id busy: {}", kLogPrefix, input.taskId,
                  e.what());
      onError(e.what());
      return;
    }
  }

  std::optional<SessionManager::AcquiredSession> acquired;
  if (!useResponseId && !blockInfos.empty()) {
    try {
      acquired = sessionManager.tryAcquireByPrefixHash(blockInfos, cancelFn);
      if (acquired.has_value() && acquired->sessionFound) {
        tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(true);
        sessionManager.registerPrefixHash(acquired->sessionId, blockInfos);

        auto slot = fromAcquiredPrefix(*acquired);
        TT_LOG_INFO(
            "{} taskId={} prefix HIT sessionId={} slotId={} "
            "decodePositionId={}",
            kLogPrefix, input.taskId, slot.sessionId, slot.slotId,
            slot.decodePositionId);
        onResolved(std::move(slot));
        return;
      }

      tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(false);
      TT_LOG_INFO("{} taskId={} prefix MISS blocks={} → allocating new session",
                  kLogPrefix, input.taskId, blockInfos.size());
    } catch (const SessionInFlightException& e) {
      TT_LOG_WARN("{} taskId={} prefix candidates busy: {}", kLogPrefix,
                  input.taskId, e.what());
      onError(e.what());
      return;
    }
  }

  if (blockInfos.empty()) {
    TT_LOG_INFO("{} taskId={} no blocks → allocating new session", kLogPrefix,
                input.taskId);
  }

  auto copyPlan = acquired.has_value()
                      ? session_resolution::prepareSlotCopy(
                            sessionManager, acquired->candidatesList,
                            input.taskId, kLogPrefix)
                      : std::nullopt;
  std::optional<uint32_t> slotToCopyFrom =
      copyPlan.has_value() ? std::make_optional(copyPlan->slotToCopyFrom)
                           : std::nullopt;
  uint32_t copyMatchedTokens =
      copyPlan.has_value() ? copyPlan->matchedTokens : 0;

  sessionManager.createSession(
      [&sessionManager, blockInfos, slotToCopyFrom, copyMatchedTokens,
       onResolved,
       taskId = input.taskId](const tt::domain::Session& session) mutable {
        if (slotToCopyFrom.has_value()) {
          sessionManager.unlockSlot(*slotToCopyFrom);
        }

        DecodeDestinationSlot slot;
        slot.sessionId = session.getSessionId();
        slot.slotId = sessionManager.acquireInFlight(session.getSessionId(),
                                                     /*cancelFn=*/nullptr);
        sessionManager.registerPrefixHash(session.getSessionId(), blockInfos);

        if (slotToCopyFrom.has_value() && copyMatchedTokens > 0) {
          const uint32_t kvPositionId = copyMatchedTokens - 1;
          setDecodePositionFields(slot, kvPositionId,
                                  /*accumulatedThinkTokens=*/0, true);
        }

        TT_LOG_INFO(
            "{} taskId={} new session sessionId={} slotId={} continuation={}",
            kLogPrefix, taskId, slot.sessionId, slot.slotId, slot.continuation);
        onResolved(std::move(slot));
      },
      [onError, &sessionManager, slotToCopyFrom](std::string_view err) {
        if (slotToCopyFrom.has_value()) {
          sessionManager.unlockSlot(*slotToCopyFrom);
        }
        onError(err);
      },
      eventLoop, blockInfos, /*slotId=*/std::nullopt, slotToCopyFrom);
}

}  // namespace tt::services::decode_slot_reservation
