// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/llm_pipeline.hpp"

#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "config/settings.hpp"
#include "metrics/metrics.hpp"
#include "services/disaggregation_service.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"
#include "services/session_resolution.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::services {

LLMPipeline::LLMPipeline(
    std::shared_ptr<LLMService> service,
    std::shared_ptr<SessionManager> sessionManager,
    std::shared_ptr<DisaggregationService> disaggregationService,
    std::shared_ptr<sockets::InterServerService> socketService)
    : service_(std::move(service)),
      sessionManager_(std::move(sessionManager)),
      disaggregationService_(std::move(disaggregationService)),
      socketService_(std::move(socketService)) {}

namespace {

/**
 * Compute prefix-cache routing for the request, picking the message-level or
 * token-level hasher based on what the caller filled in.
 */
tt::utils::PrefixCachingInfo computeRoutingInfo(
    const tt::domain::llm::LLMRequest& req) {
  if (auto* tokens = std::get_if<std::vector<int>>(&req.prompt)) {
    // Full token-id dump: capped so we don't blow up the log line on long
    // prompts but keeps the head/tail context that matters for diagnosing
    // boundary detection. Set DYNAMO_LOG_FULL_TOKENS=1 to disable the cap.
    const bool fullDump = []() {
      const char* v = std::getenv("DYNAMO_LOG_FULL_TOKENS");
      return v && (*v == '1' || *v == 't' || *v == 'T');
    }();
    constexpr size_t kHeadTail = 32;
    auto dump = [&]() {
      const size_t n = tokens->size();
      std::string s;
      s.reserve(n * 6);
      auto append = [&](size_t i) {
        if (!s.empty()) s += ",";
        s += std::to_string((*tokens)[i]);
      };
      if (fullDump || n <= 2 * kHeadTail) {
        for (size_t i = 0; i < n; ++i) append(i);
      } else {
        for (size_t i = 0; i < kHeadTail; ++i) append(i);
        s += ",...,";
        for (size_t i = n - kHeadTail; i < n; ++i) append(i);
      }
      return s;
    };

    const auto& header =
        tt::utils::tokenizers::staticInfo().assistantHeaderSequence;
    std::string headerStr;
    for (int t : header) {
      if (!headerStr.empty()) headerStr += ",";
      headerStr += std::to_string(t);
    }
    TT_LOG_INFO(
        "[LLMPipeline] Hashing path=tokens taskId={} tokens={} "
        "asstHeaderSequence=[{}]",
        req.task_id, tokens->size(), headerStr);
    TT_LOG_INFO("[LLMPipeline] Token dump taskId={} ids=[{}]", req.task_id,
                dump());

    auto info = tt::utils::computePrefixCachingInfoFromTokens(*tokens);
    TT_LOG_INFO(
        "[LLMPipeline] Routing taskId={} blocks={} thinkTokens={}", req.task_id,
        info.blocks.size(),
        info.blocks.empty() ? 0 : info.blocks.back().accumulatedThinkTokens);
    return info;
  }
  return {};
}

/**
 * Wrap a streaming callback so the first chunk reports `cachedPromptTokens` via
 * LLMStreamChunk::cached_prompt_tokens — the same field the prefill server uses
 * for offloaded requests — so a locally-served prefix-cache hit surfaces in
 * usage.prompt_tokens_details.cached_tokens. Returns `cb` unchanged when there
 * is nothing to report.
 */
std::function<void(const tt::domain::llm::LLMStreamChunk&, bool)>
stampCachedPromptTokens(
    std::function<void(const tt::domain::llm::LLMStreamChunk&, bool)> cb,
    int cachedPromptTokens) {
  if (cachedPromptTokens <= 0) {
    return cb;
  }
  // Per-request callbacks are serialized, so a plain `stamped` flag is enough.
  return
      [cb = std::move(cb), cachedPromptTokens, stamped = false](
          const tt::domain::llm::LLMStreamChunk& chunk, bool isFinal) mutable {
        if (stamped) {
          cb(chunk, isFinal);
          return;
        }
        stamped = true;
        tt::domain::llm::LLMStreamChunk first = chunk;
        first.cached_prompt_tokens = cachedPromptTokens;
        cb(first, isFinal);
      };
}

void populateRequestFromContinuationHit(
    tt::domain::llm::LLMRequest& req,
    const SessionManager::AcquiredSession& acquired,
    const tt::utils::PrefixCachingInfo& routingInfo,
    const std::shared_ptr<SessionManager>& sessionManager,
    bool responseIdPath) {
  req.slotId = acquired.slotId;
  req.session = sessionManager->getSession(acquired.sessionId);
  req.sessionId = acquired.sessionId;
  req.continuation = true;

  const uint32_t matchedTokens = acquired.numberOfMatchedTokens;
  const uint32_t thinkTokens = acquired.accumulatedThinkTokens;
  req.kv_position_id = matchedTokens + thinkTokens;
  req.accumulated_think_tokens = static_cast<int>(thinkTokens);

  if (auto* promptTokens = std::get_if<std::vector<int>>(&req.prompt)) {
    if (responseIdPath) {
      session_resolution::applyDeltaPrompt(req, matchedTokens,
                                           {.skipUnlessRegularMode = true,
                                            .setKvPositionId = false,
                                            .logPrefix = {}});
      req.session->initTokenAccumulator(
          *promptTokens, routingInfo.blocks,
          [mgr = sessionManager](
              const std::string& sessionId,
              const std::vector<tt::utils::BlockHashInfo>& blocks) {
            mgr->registerPrefixHash(sessionId, blocks);
          });
    } else {
      std::vector<int> fullPrompt = *promptTokens;
      session_resolution::applyDeltaPrompt(req, matchedTokens,
                                           {.skipUnlessRegularMode = true,
                                            .setKvPositionId = false,
                                            .logPrefix = {}});
      if (!fullPrompt.empty()) {
        req.session->initTokenAccumulator(
            std::move(fullPrompt), /*initialBlocks=*/{},
            [mgr = sessionManager](
                const std::string& sessionId,
                const std::vector<tt::utils::BlockHashInfo>& blocks) {
              mgr->registerPrefixHash(sessionId, blocks);
            },
            /*parentThinkCount=*/thinkTokens);
      }
    }
  }
}

}  // namespace

void LLMPipeline::resolveSession(
    std::shared_ptr<tt::domain::llm::LLMRequest> req, trantor::EventLoop* loop,
    std::function<void(SessionInfo)> onResolved,
    std::function<void(const SessionError&)> onError,
    std::function<void()> cancelFn) const {
  size_t promptTokens = 0;
  const char* promptKind = "string";
  if (auto* toks = std::get_if<std::vector<int>>(&req->prompt)) {
    promptTokens = toks->size();
    promptKind = "tokens";
  }
  TT_LOG_INFO(
      "[LLMPipeline] Request received taskId={} model={} stream={} "
      "messages={} promptKind={} promptTokens={}",
      req->task_id, req->model.value_or("default"), req->stream,
      req->messages.size(), promptKind, promptTokens);

  SessionInfo info;

  if (!sessionManager_) {
    TT_LOG_WARN("[LLMPipeline] SessionManager not available");
    onResolved(info);
    return;
  }

  auto routingInfo = computeRoutingInfo(*req);
  TT_LOG_INFO("[LLMPipeline] Routing taskId={} blocks={}", req->task_id,
              routingInfo.blocks.size());

  const bool useResponseId =
      req->previousResponseId.has_value() && !req->previousResponseId->empty();

  std::optional<SessionManager::AcquiredSession> prefixMiss;
  try {
    const auto tAcquireStart = std::chrono::steady_clock::now();
    const auto resolveResult = sessionManager_->tryResolve(
        req->previousResponseId, routingInfo.blocks, cancelFn);
    const auto acquireUs =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - tAcquireStart)
            .count();
    TT_LOG_INFO("[SessionTimer] taskId={} tryResolve_us={} outcome={}",
                req->task_id, acquireUs,
                resolveResult.outcome == PrefixCacheResolveOutcome::Hit
                    ? "hit"
                    : "miss");

    if (resolveResult.outcome == PrefixCacheResolveOutcome::Hit) {
      const auto& acquired = resolveResult.acquired;
      tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(true);
      if (useResponseId) {
        TT_LOG_INFO(
            "[LLMPipeline] Response-id HIT taskId={} prevId={} sessionId={} "
            "slotId={}",
            req->task_id, *req->previousResponseId, acquired.sessionId,
            acquired.slotId);
      } else {
        TT_LOG_INFO(
            "[LLMPipeline] Prefix cache HIT taskId={} sessionId={} "
            "slotId={} matchedTokens={} thinkTokens={}",
            req->task_id, acquired.sessionId, acquired.slotId,
            acquired.numberOfMatchedTokens, acquired.accumulatedThinkTokens);
      }

      populateRequestFromContinuationHit(*req, acquired, routingInfo,
                                         sessionManager_, useResponseId);

      TT_LOG_INFO(
          "[LLMPipeline] Continuation delta taskId={} matchedTokens={} "
          "thinkTokens={} deltaTokens={}",
          req->task_id, acquired.numberOfMatchedTokens,
          acquired.accumulatedThinkTokens, req->prompt_tokens_count);

      ContinuationCommit commit;
      commit.sessionId = acquired.sessionId;
      commit.blocks = routingInfo.blocks;
      commit.matchedTokens = acquired.numberOfMatchedTokens;
      if (useResponseId && req->responseId.has_value()) {
        commit.previousResponseId = req->previousResponseId;
        commit.responseId = req->responseId;
      }
      sessionManager_->commitContinuation(commit);

      info.validSessionFound = true;
      info.registrationHashes = routingInfo.hashes();
      onResolved(info);
      return;
    }

    tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(false);
    if (resolveResult.outcome == PrefixCacheResolveOutcome::Miss &&
        !resolveResult.acquired.candidatesList.empty()) {
      prefixMiss = resolveResult.acquired;
    }

    if (useResponseId) {
      TT_LOG_INFO(
          "[LLMPipeline] Response-id MISS taskId={} prevId={} -> allocating "
          "new session",
          req->task_id, *req->previousResponseId);
    } else if (!routingInfo.blocks.empty()) {
      TT_LOG_INFO(
          "[LLMPipeline] Prefix cache MISS taskId={} blocks={} -> allocating "
          "new session",
          req->task_id, routingInfo.blocks.size());
    }
  } catch (const services::SessionInFlightException& e) {
    if (useResponseId) {
      TT_LOG_WARN("[LLMPipeline] Session busy for prevId={}: {}",
                  *req->previousResponseId, e.what());
    } else {
      TT_LOG_WARN("[LLMPipeline] All sessions busy: {}", e.what());
    }
    onError({SessionErrorType::RATE_LIMIT, e.what()});
    return;
  }

  if (routingInfo.blocks.empty()) {
    TT_LOG_INFO(
        "[LLMPipeline] No blocks for taskId={} → allocating new session",
        req->task_id);
  }

  // Layer 2: Allocate a new session. Async — onCompletion runs on `loop`.
  // Before allocating, check if there's a candidate slot worth copying from.
  auto copyPlan = prefixMiss.has_value()
                      ? session_resolution::prepareSlotCopy(
                            *sessionManager_, prefixMiss->candidatesList,
                            req->task_id, "[LLMPipeline]")
                      : std::nullopt;

  // The decode-side copy only pays off when the delta is prefilled locally. If
  // the uncached delta is large enough to route to the prefill server, drop the
  // copy and let the prefill server reuse its own prefix cache instead.
  if (copyPlan.has_value() &&
      tt::config::llmMode() == tt::config::LLMMode::DECODE_ONLY) {
    const size_t deltaTokens = promptTokens > copyPlan->matchedTokens
                                   ? promptTokens - copyPlan->matchedTokens
                                   : 0;
    if (!willPrefillOnDecode(*req, deltaTokens)) {
      TT_LOG_INFO(
          "[LLMPipeline] taskId={} delta={} exceeds prefill-on-decode limit; "
          "routing to prefill server, skipping slot copy from slotId={}",
          req->task_id, deltaTokens, copyPlan->slotToCopyFrom);
      sessionManager_->unlockSlot(copyPlan->slotToCopyFrom);
      copyPlan.reset();
    }
  }

  std::optional<uint32_t> slotToCopyFrom =
      copyPlan.has_value() ? std::make_optional(copyPlan->slotToCopyFrom)
                           : std::nullopt;
  uint32_t copyMatchedTokens =
      copyPlan.has_value() ? copyPlan->matchedTokens : 0;

  // Capture `tCreateStart` so the onCompletion callback can report end-to-end
  // createSession latency (submit → completion). Under contention this gap
  // grows: it covers queueing for the SessionManager, slot allocation, any
  // memory-request RPC, and the trantor hop back onto `loop`.
  const auto tCreateStart = std::chrono::steady_clock::now();
  sessionManager_->createSession(
      [req, routingInfo, onResolved, cancelFn = std::move(cancelFn),
       mgr = sessionManager_, slotToCopyFrom, copyMatchedTokens,
       tCreateStart](const tt::domain::Session& session) mutable {
        const auto createUs =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - tCreateStart)
                .count();

        // Unlock the source slot now that allocation is complete.
        if (slotToCopyFrom.has_value()) {
          mgr->unlockSlot(*slotToCopyFrom);
        }

        const auto tAcqInFlightStart = std::chrono::steady_clock::now();
        req->sessionId = session.getSessionId();
        req->slotId =
            mgr->acquireInFlight(session.getSessionId(), std::move(cancelFn));
        const auto acqInFlightUs =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - tAcqInFlightStart)
                .count();

        req->session = mgr->getSession(session.getSessionId());

        // Register under this turn's response id (when present) so the
        // next request's previous_response_id resolves to this session/slot.
        if (req->responseId.has_value()) {
          mgr->registerResponseId(session.getSessionId(), *req->responseId);
        }

        std::vector<int> fullPrompt;
        if (auto* p = std::get_if<std::vector<int>>(&req->prompt)) {
          fullPrompt = *p;
        }

        // If we copied from a slot, mark as continuation with kv_position_id.
        if (slotToCopyFrom.has_value() && copyMatchedTokens > 0) {
          req->continuation = true;
          req->kv_position_id = copyMatchedTokens;
          session_resolution::applyDeltaPrompt(*req, copyMatchedTokens,
                                               {.skipUnlessRegularMode = true,
                                                .setKvPositionId = false,
                                                .logPrefix = {}});
        } else {
          req->continuation = false;
        }

        // slotToCopyFrom requests the KV copy; this registers the new session
        // under the full request prefix so future lookups can find it.
        mgr->registerPrefixHash(session.getSessionId(), routingInfo.blocks);
        if (!fullPrompt.empty()) {
          req->session->initTokenAccumulator(
              std::move(fullPrompt), /*initialBlocks=*/{},
              [mgr](const std::string& sessionId,
                    const std::vector<tt::utils::BlockHashInfo>& blocks) {
                mgr->registerPrefixHash(sessionId, blocks);
              });
        }

        TT_LOG_INFO(
            "[SessionTimer] taskId={} createSession_us={} "
            "acquireInFlight_us={}",
            req->task_id, createUs, acqInFlightUs);
        TT_LOG_INFO(
            "[LLMPipeline] New session: sessionId={}, slotId={}, hashes={}",
            session.getSessionId(),
            req->slotId.has_value() ? std::to_string(*req->slotId) : "none",
            routingInfo.hashes().size());

        SessionInfo info;
        info.registrationHashes = routingInfo.hashes();
        onResolved(info);
      },
      [onError, mgr = sessionManager_, slotToCopyFrom](std::string_view err) {
        if (slotToCopyFrom.has_value()) {
          mgr->unlockSlot(*slotToCopyFrom);
        }
        onError({SessionErrorType::ALLOCATION_FAIL, std::string(err)});
      },
      loop, routingInfo.blocks, /*slotId=*/std::nullopt, slotToCopyFrom);
}

void LLMPipeline::runStreamingRequest(
    std::shared_ptr<tt::domain::llm::LLMRequest> req, trantor::EventLoop* loop,
    StreamCallbackFactory makeStreamCallback, GenerationHandlers handlers,
    std::function<void()> cancelFn) const {
  if (!cancelFn) {
    cancelFn = [this, taskId = req->task_id]() { abortRequest(taskId); };
  }
  auto resolvedHandlers = handlers;
  auto errorHandlers = std::move(handlers);

  resolveSession(
      req, loop,
      [this, req, makeStreamCallback = std::move(makeStreamCallback),
       handlers =
           std::move(resolvedHandlers)](SessionInfo sessionInfo) mutable {
        if (handlers.onSessionResolved) {
          handlers.onSessionResolved(sessionInfo);
        }

        // dispatchGeneration moves req->session, so keep a stable copy.
        auto sessionPtr = req->session;
        try {
          service_->preProcess(*req);
        } catch (const std::exception& e) {
          if (handlers.onPreProcessError) {
            handlers.onPreProcessError(e, sessionPtr);
          }
          return;
        }

        if (handlers.onPreProcessed) {
          handlers.onPreProcessed();
        }

        auto streamCallback = makeStreamCallback(sessionInfo, sessionPtr);
        try {
          dispatchGeneration(*req, sessionInfo, streamCallback);
        } catch (const std::exception& e) {
          if (handlers.onDispatchError) {
            handlers.onDispatchError(e, sessionPtr);
          }
          return;
        }

        if (handlers.onDispatchSucceeded) {
          handlers.onDispatchSucceeded();
        }
      },
      [handlers = std::move(errorHandlers)](const SessionError& err) mutable {
        if (handlers.onSessionError) {
          handlers.onSessionError(err);
        }
      },
      std::move(cancelFn));
}

void LLMPipeline::dispatchGeneration(
    tt::domain::llm::LLMRequest& request, SessionInfo sessionInfo,
    const std::function<void(const tt::domain::llm::LLMStreamChunk&, bool)>& cb)
    const {
  const auto mode = tt::config::llmMode();
  if (mode == tt::config::LLMMode::REGULAR) {
    service_->submitStreamingRequest(request, cb, /*skipPreProcess=*/true);
    return;
  }

  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    if (shouldDoPrefillOnDecode(request)) {
      // If continuation, trim prompt to only the uncached delta before
      // submitting to the local decode device. The trimmed-off prefix is what
      // the local KV cache served, i.e.
      // usage.prompt_tokens_details.cached_tokens.
      int reusedPrefixTokens = 0;
      if (request.continuation && request.kv_position_id.has_value()) {
        // kv_position_id is the first free KV index, so the matched non-think
        // prompt length is kv_position_id minus the resident think tokens.
        uint32_t matchedTokens =
            *request.kv_position_id -
            static_cast<uint32_t>(request.accumulated_think_tokens);
        const auto fullPromptTokens =
            std::get<std::vector<int>>(request.prompt).size();
        session_resolution::applyDeltaPrompt(request, matchedTokens);
        reusedPrefixTokens =
            static_cast<int>(fullPromptTokens -
                             std::get<std::vector<int>>(request.prompt).size());
      }
      TT_LOG_DEBUG("[LLMPipeline] Using prefill on decode for sessionId: {}",
                   request.sessionId.value_or("none"));
      service_->submitStreamingRequest(
          request, stampCachedPromptTokens(cb, reusedPrefixTokens),
          /*skipPreProcess=*/true);
    } else {
      TT_LOG_DEBUG(
          "[LLMPipeline] Using disaggregated prefill for request with "
          "sessionId: {}",
          request.sessionId.value_or("none"));
      // WARNING - TEMP CHANGE - PREFILL WILL OVERRIDE THINKING TOKENS
      uint32_t matchedTokens =
          *request.kv_position_id -
          static_cast<uint32_t>(request.accumulated_think_tokens);
      *request.kv_position_id = matchedTokens;
      if (sessionManager_ && request.session) {
        sessionManager_->clearSessionBlockThinkTokens(
            request.session->getSessionId());
      }
      // WARNING - TEMP CHANGE
      disaggregationService_->handleStreamingRequest(
          request, sessionInfo.registrationHashes, cb);
    }
    return;
  }

  throw std::runtime_error(
      "LLM Mode must be regular or decode only for chat completions");
}

void LLMPipeline::abortRequest(uint32_t taskId) const {
  service_->abortRequest(taskId);
  if (disaggregationService_) {
    disaggregationService_->abortRequest(taskId);
  }
}

bool LLMPipeline::willPrefillOnDecode(
    const tt::domain::llm::LLMRequest& request, size_t deltaTokens) const {
  const bool socketReady = socketService_ && socketService_->isConnected();
  if (!socketReady) {
    TT_LOG_WARN(
        "[LLMPipeline] Prefill server not connected; falling back to "
        "prefill on decode for taskId={}",
        request.task_id);
    return true;
  }

  if (request.disaggregation_override.has_value()) {
    const bool forceDisagg = *request.disaggregation_override;
    TT_LOG_INFO(
        "[LLMPipeline] Honoring disaggregation override={} for taskId={}",
        forceDisagg, request.task_id);
    return !forceDisagg;
  }

  return deltaTokens < tt::config::maxTokensToPrefillOnDecode();
}

bool LLMPipeline::shouldDoPrefillOnDecode(
    const tt::domain::llm::LLMRequest& request) const {
  size_t promptTokens = static_cast<size_t>(request.prompt_tokens_count);

  // If we have a prefix-cache hit, the matched tokens are already in the KV
  // cache and won't need prefilling again — deduct them from the effective
  // prompt size used for the threshold comparison.
  if (request.kv_position_id.has_value()) {
    // kv_position_id is the first free KV index; the cached non-think prefix
    // length is therefore kv_position_id minus the resident think tokens.
    const size_t cached = static_cast<size_t>(*request.kv_position_id) -
                          static_cast<size_t>(request.accumulated_think_tokens);
    promptTokens = (promptTokens > cached) ? promptTokens - cached : 0;
  }

  return willPrefillOnDecode(request, promptTokens);
}

}  // namespace tt::services
