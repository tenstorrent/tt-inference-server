// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/llm_pipeline.hpp"

#include <chrono>
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
  if (!req.messages.empty()) {
    TT_LOG_INFO("[LLMPipeline] Hashing path=messages messages={} taskId={}",
                req.messages.size(), req.task_id);
    return tt::utils::computePrefixCachingInfo(req.messages);
  }
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

    return tt::utils::computePrefixCachingInfoFromTokens(*tokens);
  }
  TT_LOG_INFO(
      "[LLMPipeline] Hashing path=none (string prompt, no messages) "
      "taskId={}",
      req.task_id);
  return {};
}

/**
 * Apply the resolved delta prompt to the request. The hasher returns either
 * a rendered string (message path) or a token id vector (Dynamo path); both
 * variants slot directly into `req.prompt`.
 */
void applyDeltaPrompt(tt::domain::llm::LLMRequest& req,
                      tt::utils::PrefixCachingInfo& info) {
  if (auto* s = std::get_if<std::string>(&info.deltaPrompt)) {
    req.prompt_tokens_count = static_cast<int>(
        tt::utils::tokenizers::activeTokenizer().encode(*s).size());
    req.prompt = std::move(*s);
  } else {
    auto& tok = std::get<std::vector<int>>(info.deltaPrompt);
    req.prompt_tokens_count = static_cast<int>(tok.size());
    req.prompt = std::move(tok);
  }
}

/**
 * Response-id continuation delta. Shrinks `req.prompt` (token-id vector) to the
 * suffix the slot has not cached yet: tokens[cachedLen:]. `cachedLen` is what
 * the matched session committed to its slot last turn (that turn's full prompt
 * + generated tokens), which equals the slot's KV prefix — so only the new
 * turn is prefilled, matching the content-hash path's delta without any header
 * detection.
 *
 * Returns the full token count (pre-shrink) so the caller can re-record it as
 * the next turn's cached length. The prompt is left untouched (full reprefill)
 * when there's no usable cached prefix (`cachedLen` 0, out of range, or a
 * string prompt), so we never dispatch an empty or misaligned delta.
 */
size_t applyResponseIdDelta(tt::domain::llm::LLMRequest& req,
                            size_t cachedLen) {
  auto* tokens = std::get_if<std::vector<int>>(&req.prompt);
  if (tokens == nullptr) {
    return 0;
  }
  const size_t fullLen = tokens->size();
  if (cachedLen == 0 || cachedLen >= fullLen) {
    return fullLen;
  }
  std::vector<int> delta(
      tokens->begin() + static_cast<std::ptrdiff_t>(cachedLen), tokens->end());
  req.prompt_tokens_count = static_cast<int>(delta.size());
  req.prompt = std::move(delta);
  return fullLen;
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
  TT_LOG_INFO(
      "[LLMPipeline] Routing taskId={} hasPriorTurn={} lookupHash={} "
      "registrationHash={}",
      req->task_id, routingInfo.hasPriorTurn,
      routingInfo.lookupHash.has_value()
          ? std::to_string(*routingInfo.lookupHash)
          : "none",
      routingInfo.registrationHash);

  // Layer 1a: Responses API continuation. When the previous_response_id is
  // supplied, we route by that id instead of the content-prefix
  // hash (parallel to the prefixIndex path below). The token delta is still
  // computed by the hasher so we only prefill the new turn.
  const bool useResponseId =
      req->previousResponseId.has_value() && !req->previousResponseId->empty();
  if (useResponseId) {
    try {
      const auto tAcquireStart = std::chrono::steady_clock::now();
      auto acquired = sessionManager_->tryAcquireByResponseId(
          *req->previousResponseId, cancelFn);
      const auto acquireUs =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - tAcquireStart)
              .count();
      TT_LOG_INFO(
          "[SessionTimer] taskId={} tryAcquireByResponseId_us={} hit={}",
          req->task_id, acquireUs, acquired.has_value());

      if (acquired.has_value()) {
        tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(true);
        TT_LOG_INFO(
            "[LLMPipeline] Response-id HIT taskId={} prevId={} sessionId={} "
            "slotId={}",
            req->task_id, *req->previousResponseId, acquired->sessionId,
            acquired->slotId);
        req->slotId = acquired->slotId;
        req->session = sessionManager_->getSession(acquired->sessionId);
        req->continuation = true;
        // Prefill only the delta: the slot already holds the first
        // `cachedPromptLen` tokens (prior turn's prompt + generated, recorded
        // at that turn's completion), so dispatch tokens[cachedPromptLen:].
        const size_t fullLen =
            applyResponseIdDelta(*req, acquired->cachedPromptLen);
        TT_LOG_INFO(
            "[LLMPipeline] Response-id delta taskId={} cachedPromptLen={} "
            "fullLen={} deltaTokens={}",
            req->task_id, acquired->cachedPromptLen, fullLen,
            req->prompt_tokens_count);
        // Re-key the session under this turn's id with the full prompt length
        // as a lower-bound cached prefix; the completion hook bumps it to
        // prompt + generated tokens once this turn finishes.
        if (req->responseId.has_value()) {
          sessionManager_->registerResponseId(acquired->sessionId,
                                              *req->responseId, fullLen);
        }
        sessionManager_->registerPrefixHash(acquired->sessionId,
                                            routingInfo.registrationHash);
        info.validSessionFound = true;
        info.registrationHash = routingInfo.registrationHash;
        onResolved(info);
        return;
      }

      tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(false);
      TT_LOG_INFO(
          "[LLMPipeline] Response-id MISS taskId={} prevId={} → allocating "
          "new session",
          req->task_id, *req->previousResponseId);
    } catch (const services::SessionInFlightException& e) {
      TT_LOG_WARN("[LLMPipeline] Session busy for prevId={}: {}",
                  *req->previousResponseId, e.what());
      onError({SessionErrorType::RATE_LIMIT, e.what()});
      return;
    }
  }

  // Layer 1b: Prefix-cache routing. Requires a prior turn and a lookup hash.
  // Skipped when we routed by response id above (lookup by id, not hash).
  if (!useResponseId && routingInfo.hasPriorTurn &&
      routingInfo.lookupHash.has_value()) {
    try {
      // Time the synchronous SessionManager acquire. Under burst load this
      // is where prefixIndex / sessions mutex contention shows up; per-req
      // µs lets you spot tail latency events without a profiler.
      const auto tAcquireStart = std::chrono::steady_clock::now();
      auto acquired = sessionManager_->tryAcquireByPrefixHash(
          *routingInfo.lookupHash, cancelFn);
      const auto acquireUs =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - tAcquireStart)
              .count();
      TT_LOG_INFO(
          "[SessionTimer] taskId={} tryAcquireByPrefixHash_us={} hit={}",
          req->task_id, acquireUs, acquired.has_value());

      if (acquired.has_value()) {
        tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(true);
        TT_LOG_INFO(
            "[LLMPipeline] Prefix cache HIT taskId={} hash={} sessionId={} "
            "slotId={}",
            req->task_id, *routingInfo.lookupHash, acquired->sessionId,
            acquired->slotId);
        req->slotId = acquired->slotId;
        req->session = sessionManager_->getSession(acquired->sessionId);
        req->continuation = true;
        applyDeltaPrompt(*req, routingInfo);
        sessionManager_->registerPrefixHash(acquired->sessionId,
                                            routingInfo.registrationHash);
        info.validSessionFound = true;
        info.registrationHash = routingInfo.registrationHash;
        onResolved(info);
        return;
      }

      tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(false);
      TT_LOG_INFO(
          "[LLMPipeline] Prefix cache MISS taskId={} hash={} → allocating "
          "new session",
          req->task_id, *routingInfo.lookupHash);
    } catch (const services::SessionInFlightException& e) {
      TT_LOG_WARN("[LLMPipeline] All sessions busy for hash={}: {}",
                  *routingInfo.lookupHash, e.what());
      onError({SessionErrorType::RATE_LIMIT, e.what()});
      return;
    }
  }

  if (!routingInfo.hasPriorTurn || !routingInfo.lookupHash.has_value()) {
    TT_LOG_INFO(
        "[LLMPipeline] No prior turn detected taskId={} → allocating new "
        "session (registrationHash={})",
        req->task_id, routingInfo.registrationHash);
  }

  // Layer 2: Allocate a new session. Async — onCompletion runs on `loop`.
  // Capture `tCreateStart` so the onCompletion callback can report end-to-end
  // createSession latency (submit → completion). Under contention this gap
  // grows: it covers queueing for the SessionManager, slot allocation, any
  // memory-request RPC, and the trantor hop back onto `loop`.
  const auto tCreateStart = std::chrono::steady_clock::now();
  sessionManager_->createSession(
      [req, routingInfo, onResolved, cancelFn = std::move(cancelFn),
       mgr = sessionManager_,
       tCreateStart](const tt::domain::Session& session) mutable {
        const auto createUs =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - tCreateStart)
                .count();

        const auto tAcqInFlightStart = std::chrono::steady_clock::now();
        req->sessionId = session.getSessionId();
        req->slotId =
            mgr->acquireInFlight(session.getSessionId(), std::move(cancelFn));
        const auto acqInFlightUs =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - tAcqInFlightStart)
                .count();

        req->session = mgr->getSession(session.getSessionId());
        req->continuation = false;
        mgr->registerPrefixHash(session.getSessionId(),
                                routingInfo.registrationHash);
        // Also register under this turn's response id (when present) so the
        // next request's previous_response_id resolves to this session/slot.
        // The full prompt length is a lower-bound cached prefix; the
        // completion hook bumps it to prompt + generated tokens.
        if (req->responseId.has_value()) {
          mgr->registerResponseId(
              session.getSessionId(), *req->responseId,
              static_cast<size_t>(req->full_prompt_tokens_count));
        }

        TT_LOG_INFO(
            "[SessionTimer] taskId={} createSession_us={} "
            "acquireInFlight_us={}",
            req->task_id, createUs, acqInFlightUs);
        TT_LOG_INFO(
            "[LLMPipeline] New session: sessionId={}, slotId={}, "
            "registered under hash={}",
            session.getSessionId(),
            req->slotId.has_value() ? std::to_string(*req->slotId) : "none",
            routingInfo.registrationHash);

        SessionInfo info;
        info.registrationHash = routingInfo.registrationHash;
        onResolved(info);
      },
      [onError](std::string_view err) {
        onError({SessionErrorType::ALLOCATION_FAIL, std::string(err)});
      },
      loop, routingInfo.registrationHash);
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
    if (shouldDoPrefillOnDecode(request, sessionInfo.validSessionFound)) {
      TT_LOG_DEBUG("[LLMPipeline] Using prefill on decode for sessionId: {}",
                   request.sessionId.value_or("none"));
      service_->submitStreamingRequest(request, cb, /*skipPreProcess=*/true);
    } else {
      TT_LOG_DEBUG(
          "[LLMPipeline] Using disaggregated prefill for request with "
          "sessionId: {}",
          request.sessionId.value_or("none"));
      disaggregationService_->handleStreamingRequest(
          request, sessionInfo.registrationHash.value_or(0), cb);
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

bool LLMPipeline::shouldDoPrefillOnDecode(
    const tt::domain::llm::LLMRequest& request, bool validSessionFound) const {
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

  if (validSessionFound) {
    return true;
  }

  const size_t maxTokens = tt::config::maxTokensToPrefillOnDecode();
  const size_t promptTokens = static_cast<size_t>(request.prompt_tokens_count);

  return promptTokens < maxTokens;
}

}  // namespace tt::services
