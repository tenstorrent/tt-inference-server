// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/llm_pipeline.hpp"

#include <trantor/net/EventLoop.h>

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
#include "utils/logger.hpp"

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

}  // namespace

void LLMPipeline::resolveSession(
    std::shared_ptr<tt::domain::llm::LLMRequest> req, trantor::EventLoop* loop,
    std::function<void(SessionInfo)> onResolved,
    std::function<void(const SessionError&)> onError,
    std::function<void()> cancelFn) const {
  const char* promptKind = "string";
  auto* tokens = std::get_if<std::vector<uint32_t>>(&req->prompt);
  if (tokens) {
    promptKind = "tokens";
  }
  TT_LOG_INFO(
      "[LLMPipeline] Request received taskId={} model={} stream={} "
      "messages={} promptKind={} promptTokens={}",
      req->task_id, req->model.value_or("default"), req->stream,
      req->messages.size(), promptKind, tokens ? tokens->size() : 0);

  if (tt::config::llmMode() == tt::config::LLMMode::PREFILL_ONLY &&
      tt::config::usePrefillFirstDisaggregation()) {
    auto routingInfo = computeRoutingInfo(*req);
    info.registrationHashes = routingInfo.hashes();
    onResolved(info);
    return;
  }

  if (!sessionManager_) {
    TT_LOG_WARN("[LLMPipeline] SessionManager not available");
    loop->runInLoop([onResolved]() { onResolved(SessionInfo{}); });
    return;
  }

  if (!tokens) {
    TT_LOG_WARN("[LLMPipeline] No tokens in prompt, cannot route");
    loop->runInLoop([onResolved]() { onResolved(SessionInfo{}); });
    return;
  }

  // Build options for getSlot
  GetSlotOptions opts;
  opts.previousResponseId = req->previousResponseId;
  opts.responseId = req->responseId;
  opts.cancelFn = std::move(cancelFn);

  sessionManager_->getSlot(
      *tokens, std::move(opts), loop,
      // onResolved callback
      [req, onResolved, mgr = sessionManager_](SlotAcquireResult result) {
        // Track metrics
        tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(
            !result.isNewSession);

        TT_LOG_INFO(
            "[LLMPipeline] Slot acquired taskId={} sessionId={} slotId={} "
            "matchedTokens={} thinkTokens={} isNew={}",
            req->task_id, result.sessionId, result.slotId, result.matchedTokens,
            result.accumulatedThinkTokens, result.isNewSession);

        // Set up the request
        req->sessionId = result.sessionId;
        req->slotId = result.slotId;
        req->session = mgr->getSession(result.sessionId);
        req->continuation = !result.isNewSession;
        if (!result.isNewSession) {
          req->kv_position_id =
              result.matchedTokens + result.accumulatedThinkTokens;
          req->accumulated_think_tokens =
              static_cast<int>(result.accumulatedThinkTokens);
        }

        // Capture full prompt before delta trim; finalizeAndRegisterHashes
        // re-hashes the whole conversation from scratch (initialBlocks empty).
        std::vector<uint32_t> fullPrompt;
        if (auto* promptTokens =
                std::get_if<std::vector<uint32_t>>(&req->prompt)) {
          fullPrompt = *promptTokens;
        }

        // Apply delta prompt for continuations
        if (!result.isNewSession && result.matchedTokens > 0) {
          session_resolution::applyDeltaPrompt(*req, result.matchedTokens,
                                               {.skipUnlessRegularMode = true,
                                                .setKvPositionId = false,
                                                .logPrefix = {}});
        }

        // Initialize token accumulator for incremental hash registration
        if (!fullPrompt.empty()) {
          req->session->initTokenAccumulator(
              std::move(fullPrompt), /*initialBlocks=*/{},
              [mgr](const std::string& sessionId,
                    const std::vector<tt::utils::BlockHashInfo>& blocks) {
                mgr->registerPrefixHash(sessionId, blocks);
              },
              [mgr](const std::string& sessionId) {
                mgr->closeSession(sessionId);
              },
              result.accumulatedThinkTokens);
        }

        SessionInfo info;
        info.validSessionFound = !result.isNewSession;
        info.registrationHashes.reserve(result.blocks.size());
        for (const auto& b : result.blocks) {
          info.registrationHashes.push_back(b.hash);
        }
        onResolved(info);
      },
      // onError callback
      [onError, loop](const std::string& msg) {
        loop->runInLoop(
            [onError, msg]() { onError({SessionErrorType::RATE_LIMIT, msg}); });
      });
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
            std::get<std::vector<uint32_t>>(request.prompt).size();
        session_resolution::applyDeltaPrompt(request, matchedTokens);
        reusedPrefixTokens = static_cast<int>(
            fullPromptTokens -
            std::get<std::vector<uint32_t>>(request.prompt).size());
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

  if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    if (!tt::config::usePrefillFirstDisaggregation()) {
      throw std::runtime_error(
          "LLM Mode must be regular or decode only for chat completions");
    }
    if (!disaggregationService_) {
      throw std::runtime_error(
          "[LLMPipeline] Prefill-first disaggregation requires "
          "DisaggregationService");
    }
    request.max_tokens = 1;
    disaggregationService_->handlePrefillFirstStreamingRequest(
        request, sessionInfo.registrationHashes, cb);
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
