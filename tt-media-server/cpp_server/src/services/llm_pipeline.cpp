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
#include "services/migration_service.hpp"
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
    std::shared_ptr<sockets::InterServerService> socketService,
    std::shared_ptr<MigrationService> migrationService)
    : service_(std::move(service)),
      sessionManager_(std::move(sessionManager)),
      disaggregationService_(std::move(disaggregationService)),
      socketService_(std::move(socketService)),
      migrationService_(std::move(migrationService)) {}

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
 * Trim the first `matchedTokens` from the prompt token vector so the worker
 * only prefills the uncached suffix. Expects prompt to already be a
 * vector<int> at this point. No-op if matchedTokens >= prompt size.
 */
void applyDeltaPrompt(tt::domain::llm::LLMRequest& req,
                      uint32_t matchedTokens) {
  auto& tokens = std::get<std::vector<int>>(req.prompt);
  const size_t skip = static_cast<size_t>(matchedTokens);
  if (skip >= tokens.size()) {
    return;
  }
  tokens.erase(tokens.begin(), tokens.begin() + static_cast<ptrdiff_t>(skip));
  req.prompt_tokens_count = static_cast<int>(tokens.size());
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

  // Layer 1: Prefix-cache routing. Always attempt lookup.
  if (!routingInfo.blocks.empty()) {
    try {
      const auto tAcquireStart = std::chrono::steady_clock::now();

      auto acquired =
          sessionManager_->tryAcquireByPrefixHash(routingInfo.blocks, cancelFn);

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
            "[LLMPipeline] Prefix cache HIT taskId={} sessionId={} "
            "slotId={} matchedTokens={} thinkTokens={}",
            req->task_id, acquired->sessionId, acquired->slotId,
            acquired->numberOfMatchedTokens, acquired->accumulatedThinkTokens);
        req->slotId = acquired->slotId;
        req->session = sessionManager_->getSession(acquired->sessionId);
        req->continuation = true;
        // kv_position_id accounts for both non-thinking tokens (matched) and
        // thinking tokens (accumulated in cache but not in hash)
        req->kv_position_id = --acquired->numberOfMatchedTokens +
                              acquired->accumulatedThinkTokens;
        applyDeltaPrompt(*req, acquired->numberOfMatchedTokens);

        // Initialize token accumulator with DELTA tokens (after trim).
        // At stream end, finalizeAndRegisterHashes() continues hashing from
        // initialBlocks using delta + generated tokens.
        if (auto* deltaTokens = std::get_if<std::vector<int>>(&req->prompt)) {
          req->session->initTokenAccumulator(
              *deltaTokens, routingInfo.blocks,
              [mgr = sessionManager_](
                  const std::string& sessionId,
                  const std::vector<tt::utils::BlockHashInfo>& blocks) {
                mgr->registerPrefixHash(sessionId, blocks);
              });
        }
        sessionManager_->registerPrefixHash(acquired->sessionId,
                                            routingInfo.blocks);
        info.validSessionFound = true;
        info.registrationHashes = routingInfo.hashes();
        onResolved(info);
        return;
      }

      tt::metrics::ServerMetrics::instance().onPrefixCacheLookup(false);
      TT_LOG_INFO(
          "[LLMPipeline] Prefix cache MISS taskId={} blocks={} → allocating "
          "new session",
          req->task_id, routingInfo.blocks.size());
    } catch (const services::SessionInFlightException& e) {
      TT_LOG_WARN("[LLMPipeline] All sessions busy: {}", e.what());
      onError({SessionErrorType::RATE_LIMIT, e.what()});
      return;
    }
  }

  if (routingInfo.blocks.empty()) {
    TT_LOG_INFO(
        "[LLMPipeline] No blocks for taskId={} → allocating new session",
        req->task_id);
  }

  // Layer 2: Allocate a new session. Async — onCompletion runs on `loop`.
  // Capture `tCreateStart` so the onCompletion callback can report end-to-end
  // createSession latency (submit → completion). Under contention this gap
  // grows: it covers queueing for the SessionManager, slot allocation, any
  // memory-request RPC, and the trantor hop back onto `loop`.
  const auto tCreateStart = std::chrono::steady_clock::now();
  sessionManager_->createSession(
      [req, routingInfo, onResolved, cancelFn = std::move(cancelFn),
       mgr = sessionManager_, migSvc = migrationService_,
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

        // Dummy migration call (placeholder for real slot copy logic)
        // if (migSvc && req->slotId.has_value()) {
        //   migSvc->copyFromSlot(/*sourceSlotId=*/0, *req->slotId,
        //                        /*numberOfTokens=*/0);
        // }

        mgr->registerPrefixHash(session.getSessionId(), routingInfo.blocks);

        // Initialize token accumulator for end-of-stream hash computation
        if (auto* promptTokens = std::get_if<std::vector<int>>(&req->prompt)) {
          req->session->initTokenAccumulator(
              *promptTokens, routingInfo.blocks,
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
      [onError](std::string_view err) {
        onError({SessionErrorType::ALLOCATION_FAIL, std::string(err)});
      },
      loop, routingInfo.blocks);
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
      TT_LOG_DEBUG("[LLMPipeline] Using prefill on decode for sessionId: {}",
                   request.sessionId.value_or("none"));
      service_->submitStreamingRequest(request, cb, /*skipPreProcess=*/true);
    } else {
      TT_LOG_DEBUG(
          "[LLMPipeline] Using disaggregated prefill for request with "
          "sessionId: {}",
          request.sessionId.value_or("none"));
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

bool LLMPipeline::shouldDoPrefillOnDecode(
    const tt::domain::llm::LLMRequest& request) const {
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

  const size_t maxTokens = tt::config::maxTokensToPrefillOnDecode();
  const size_t promptTokens = static_cast<size_t>(request.prompt_tokens_count);

  // delta is already applied so no matter if session is found
  // compare prompt (new or remaining) with max tokens
  return promptTokens < maxTokens;
}

}  // namespace tt::services
