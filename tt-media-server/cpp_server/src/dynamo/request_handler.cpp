// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/request_handler.hpp"

#include <trantor/net/EventLoop.h>
#include <trantor/net/EventLoopThreadPool.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "config/settings.hpp"
#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "domain/session.hpp"
#include "dynamo/llm_mapping.hpp"
#include "dynamo/prefill_result_mapping.hpp"
#include "dynamo/transport_server.hpp"
#include "services/disaggregation_contract_mapping.hpp"
#include "services/disaggregation_service.hpp"
#include "services/llm_pipeline.hpp"
#include "services/session_manager.hpp"
#include "sockets/socket_messages.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::dynamo {

DynamoRequestHandler::DynamoRequestHandler(
    std::shared_ptr<services::LLMPipeline> pipeline,
    std::shared_ptr<services::DisaggregationService> disaggregation,
    trantor::EventLoopThreadPool* loopPool)
    : pipeline_(std::move(pipeline)),
      disaggregation_(std::move(disaggregation)),
      loop_pool_(loopPool) {}

void DynamoRequestHandler::handle(const GenerateRequest& dynReq,
                                  const TcpStreamConnectionInfo& connInfo) {
  auto pipeline = pipeline_;
  auto disaggregation = disaggregation_;
  trantor::EventLoopThreadPool* pool = loop_pool_;

  using SteadyClock = std::chrono::steady_clock;
  const auto recvT = SteadyClock::now();
  const std::string probeId = dynReq.raw.get("request_id", "").asString();
  auto firstChunkSeen = std::make_shared<std::atomic<bool>>(false);

  trantor::EventLoop* loop = pool->getNextLoop();
  auto req = buildLLMRequestFromGenerateRequest(dynReq);

  // Reasoning/usage accounting for the Dynamo path. The worker doesn't decode
  // or run the response writer here (the frontend detokenizes), so we count
  // token ids directly. A model whose chat template opens <think> in the
  // prompt (Kimi) begins generation already inside the reasoning span; other
  // reasoning models emit <think> themselves. Reasoning ends at </think>.
  struct UsageAccum {
    uint32_t thinkStart;
    uint32_t thinkEnd;
    bool inReasoning;
    int completion = 0;
    int reasoning = 0;
    // Prefix-cache reuse reported by the prefill server in disaggregation
    // (carried on the first stream chunk). 0 in non-disaggregated runs.
    int cachedTokens = 0;
  };
  auto usage = std::make_shared<UsageAccum>();
  {
    const auto think = tt::utils::tokenizers::thinkTokenIds();
    usage->thinkStart = think.first;
    usage->thinkEnd = think.second;
    const auto kNo = tt::utils::tokenizers::kNoTokenId;
    usage->inReasoning = usage->thinkStart != kNo &&
                         !dynReq.token_ids.empty() &&
                         dynReq.token_ids.back() == usage->thinkStart;
  }

  // Capture which loop thread is serving this request — combined with the
  // pre-warm log this lets us spot any unexpected cold thread that bypassed
  // the warm-up (e.g. consumer thread spawned later in LLMService).
  const auto loopTid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  TT_LOG_INFO("[DynamoLatency] id={} stage=dispatched loop_tid={}",
              probeId.empty() ? "?" : probeId, loopTid);

  // The call-home writer owns the outbound connection and streams chunks
  // asynchronously on the loop — no blocking, no per-request thread. The
  // pipeline callbacks below are unchanged: sendChunk/signalDone forward to
  // it.
  auto cancelFn = [pipeline, taskId = req->task_id]() {
    pipeline->abortRequest(taskId);
  };
  auto writer = DynamoStreamWriter::create(loop, connInfo, probeId, cancelFn);
  writer->connect();

  auto sendChunk = [writer](const TokenChunk& chunk) {
    return writer->sendChunk(chunk);
  };
  auto signalDone = [writer]() { writer->finalize(); };

  auto sendErrorAndDone = [sendChunk, signalDone](
                              const std::string& message = "backend error",
                              uint16_t statusCode = 500) {
    TokenChunk err;
    err.error = message;
    err.error_code = statusCode;
    sendChunk(err);
    signalDone();
  };

  if (tt::config::dynamoRoutingEnabled() &&
      tt::config::llmMode() == tt::config::LLMMode::PREFILL_ONLY) {
    if (!disaggregation) {
      sendErrorAndDone(
          "DYN_ROUTING=1: prefill worker has no disaggregation "
          "contract service",
          500);
      return;
    }
    auto prefillMessage = buildPrefillRequestMessage(dynReq);
    auto prefillDone = std::make_shared<std::atomic<bool>>(false);
    try {
      disaggregation->handlePrefillFirstRequest(
          *req, prefillMessage.registrationHashes,
          [sendChunk, signalDone,
           prefillDone](const tt::sockets::PrefillResultMessage& result) {
            bool expected = false;
            if (!prefillDone->compare_exchange_strong(expected, true)) {
              return;
            }
            TokenChunk out;
            if (result.error) {
              out.error = result.generatedText.empty() ? "prefill error"
                                                       : result.generatedText;
              out.error_code = 500;
              sendChunk(out);
              signalDone();
              return;
            }
            auto resultForDynamo = result;
            if (!resultForDynamo.slotId.has_value()) {
              resultForDynamo.slotId = tt::domain::INVALID_SLOT_ID;
            }
            Json::Value params(Json::objectValue);
            params["tt_prefill_result"] = prefillResultToJson(resultForDynamo);
            out.disaggregated_params = std::move(params);
            DynamoUsage du;
            du.prompt_tokens =
                static_cast<int>(resultForDynamo.tokenIds.size());
            du.completion_tokens = 0;
            du.total_tokens = du.prompt_tokens;
            du.cached_tokens = resultForDynamo.cachedTokens;
            out.completion_usage = du;
            sendChunk(out);
            signalDone();
          });
    } catch (const std::exception& e) {
      sendErrorAndDone(e.what(), 503);
    }
    return;
  }

  if (tt::config::dynamoRoutingEnabled() &&
      tt::config::llmMode() == tt::config::LLMMode::DECODE_ONLY) {
    if (auto prefillResult = prefillResultFromJson(dynReq.raw)) {
      if (prefillResult->error) {
        sendErrorAndDone(prefillResult->generatedText.empty()
                             ? "prefill error"
                             : prefillResult->generatedText);
        return;
      }
      auto submitDecode =
          [pipeline, sendChunk, signalDone, usage](
              std::shared_ptr<tt::domain::llm::LLMRequest> decodeReq,
              std::shared_ptr<services::SessionManager> sessionManager,
              std::string sessionIdToRelease = "") {
            pipeline->submitResolvedStreamingRequest(
                *decodeReq, [pipeline, decodeReq, sendChunk, signalDone, usage,
                             sessionManager, sessionIdToRelease](
                                const tt::domain::llm::LLMStreamChunk& chunk,
                                bool isFinal) {
                  if (chunk.cached_prompt_tokens.has_value()) {
                    usage->cachedTokens = *chunk.cached_prompt_tokens;
                  }
                  if (!chunk.choices.empty() && chunk.choices[0].token_id) {
                    usage->completion += 1;
                  }
                  TokenChunk out = tokenChunkFromStreamChunk(chunk, isFinal);
                  if (isFinal) {
                    DynamoUsage du;
                    du.prompt_tokens = decodeReq->full_prompt_tokens_count;
                    du.completion_tokens = usage->completion;
                    du.total_tokens = du.prompt_tokens + du.completion_tokens;
                    du.cached_tokens = usage->cachedTokens;
                    out.completion_usage = du;
                  }
                  const bool sent = sendChunk(out);
                  if (isFinal) {
                    if (sessionManager && !sessionIdToRelease.empty()) {
                      sessionManager->releaseInFlight(sessionIdToRelease);
                    }
                    signalDone();
                  } else if (!sent) {
                    if (sessionManager && !sessionIdToRelease.empty()) {
                      sessionManager->releaseInFlight(sessionIdToRelease);
                    }
                    pipeline->abortRequest(decodeReq->task_id);
                  }
                });
          };

      auto decodeReq = std::make_shared<tt::domain::llm::LLMRequest>(
          tt::services::buildDecodeRequestFromPrefillResult(
              *prefillResult, {.skip_apply_chat_template = true,
                               .skip_text_decode = true,
                               .populate_token_counts = true}));
      // Dynamo sends the client completion budget on the decode hop
      // (e.g. max_tokens=32). Prefill is invoked with max_tokens=1, so any
      // remaining_tokens echoed in tt_prefill_result is not the decode
      // budget — prefer the decode GenerateRequest's max_tokens.
      if (dynReq.max_tokens.has_value()) {
        decodeReq->max_tokens = *dynReq.max_tokens;
      }
      const bool hasReservedDecodeSlot =
          prefillResult->slotId.has_value() &&
          *prefillResult->slotId != tt::domain::INVALID_SLOT_ID;
      if (!hasReservedDecodeSlot) {
        sendErrorAndDone(
            "DYN_ROUTING=1: prefill result did not include a "
            "reserved decode slot_id; slot reservation must complete "
            "before decode can continue",
            500);
        return;
      }
      submitDecode(decodeReq, pipeline->sessionManager(),
                   prefillResult->sessionId);
      return;
    }
  }

  // Reject requests whose prompt exceeds the maximum input sequence length.
  const size_t maxInputSeqLen = tt::config::maxISL();
  const size_t promptTokens =
      static_cast<size_t>(req->full_prompt_tokens_count);
  if (promptTokens > maxInputSeqLen) {
    TT_LOG_WARN(
        "[DynamoRequestHandler] Prompt exceeds max input sequence length ({} > "
        "{})",
        promptTokens, maxInputSeqLen);
    TokenChunk err;
    err.error = "Prompt exceeds maximum input sequence length (" +
                std::to_string(maxInputSeqLen) +
                " tokens): prompt_tokens=" + std::to_string(promptTokens);
    err.error_code = 400;
    sendChunk(err);
    return;
  }

  auto preProcessStart = std::make_shared<SteadyClock::time_point>();
  auto dispatchStart = std::make_shared<SteadyClock::time_point>();

  services::LLMPipeline::GenerationHandlers handlers;
  handlers.onSessionResolved = [recvT, probeId, preProcessStart](
                                   services::LLMPipeline::SessionInfo) {
    const auto tSession = SteadyClock::now();
    const auto sessionMs =
        std::chrono::duration_cast<std::chrono::microseconds>(tSession - recvT)
            .count() /
        1000.0;
    TT_LOG_INFO(
        "[DynamoLatency] id={} stage=session_ready ms_since_recv={:.3f}",
        probeId.empty() ? "?" : probeId, sessionMs);
    *preProcessStart = SteadyClock::now();
  };
  handlers.onPreProcessed = [preProcessStart, dispatchStart, probeId]() {
    const auto preProcessMs =
        std::chrono::duration_cast<std::chrono::microseconds>(
            SteadyClock::now() - *preProcessStart)
            .count() /
        1000.0;
    TT_LOG_INFO("[DynamoLatency] id={} stage=preprocessed preprocess_ms={:.3f}",
                probeId.empty() ? "?" : probeId, preProcessMs);
    *dispatchStart = SteadyClock::now();
  };
  handlers.onPreProcessError =
      [sendErrorAndDone](const std::exception& e,
                         std::shared_ptr<tt::domain::Session> sessionPtr) {
        TT_LOG_WARN("[DynamoRequestHandler] preProcess failed: {}", e.what());
        if (sessionPtr) sessionPtr->release();
        sendErrorAndDone(std::string{"preProcess failed: "} + e.what());
      };
  handlers.onDispatchError =
      [sendErrorAndDone](const std::exception& e,
                         std::shared_ptr<tt::domain::Session> sessionPtr) {
        TT_LOG_ERROR("[DynamoRequestHandler] dispatchGeneration failed: {}",
                     e.what());
        if (sessionPtr) sessionPtr->release();
        sendErrorAndDone(std::string{"dispatchGeneration failed: "} + e.what());
      };
  handlers.onSessionError =
      [sendErrorAndDone](const services::LLMPipeline::SessionError& err) {
        TT_LOG_WARN("[DynamoRequestHandler] Session resolution failed: {}",
                    err.message);
        sendErrorAndDone(std::string{"session resolution failed: "} +
                         err.message);
      };

  pipeline->runStreamingRequest(
      req, loop,
      [pipeline, req, sendChunk, signalDone, recvT, firstChunkSeen, probeId,
       usage, dispatchStart](services::LLMPipeline::SessionInfo,
                             std::shared_ptr<tt::domain::Session> sessionPtr) {
        services::LLMPipeline::StreamCallback cb = [pipeline, req, sessionPtr,
                                                    sendChunk, signalDone,
                                                    recvT, firstChunkSeen,
                                                    probeId,
                                                    tDispatch = *dispatchStart,
                                                    usage](
                                                       const tt::domain::llm::
                                                           LLMStreamChunk&
                                                               chunk,
                                                       bool isFinal) {
          bool expected = false;
          if (firstChunkSeen->compare_exchange_strong(expected, true)) {
            using SteadyClock = std::chrono::steady_clock;
            const auto firstChunkT = SteadyClock::now();
            const auto sinceRecvMs =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    firstChunkT - recvT)
                    .count() /
                1000.0;
            const auto sinceDispatchMs =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    firstChunkT - tDispatch)
                    .count() /
                1000.0;
            TT_LOG_INFO(
                "[DynamoLatency] id={} stage=first_chunk "
                "worker_recv_to_first_chunk_ms={:.3f} "
                "dispatch_to_first_chunk_ms={:.3f}",
                probeId.empty() ? "?" : probeId, sinceRecvMs, sinceDispatchMs);
          }

          if (chunk.cached_prompt_tokens.has_value()) {
            usage->cachedTokens = *chunk.cached_prompt_tokens;
          }

          // dispatchGeneration moves req->session, so use the stable
          // copy.
          if (sessionPtr && !chunk.choices.empty() &&
              chunk.choices[0].token_id) {
            sessionPtr->addGeneratedToken(
                static_cast<uint32_t>(*chunk.choices[0].token_id));
          }

          if (!chunk.choices.empty() && chunk.choices[0].token_id) {
            const uint32_t tid =
                static_cast<uint32_t>(*chunk.choices[0].token_id);
            const auto kNo = tt::utils::tokenizers::kNoTokenId;
            usage->completion += 1;
            if (usage->thinkStart != kNo && tid == usage->thinkStart) {
              usage->inReasoning = true;
            }
            if (usage->inReasoning) usage->reasoning += 1;
            if (usage->thinkEnd != kNo && tid == usage->thinkEnd) {
              usage->inReasoning = false;
            }
          }

          if (isFinal && sessionPtr) {
            sessionPtr->finalizeAndRegisterHashes();
            sessionPtr->release();
          }

          TokenChunk out = tokenChunkFromStreamChunk(chunk, isFinal);
          if (isFinal) {
            DynamoUsage du;
            du.prompt_tokens = req->full_prompt_tokens_count;
            du.completion_tokens = usage->completion;
            du.total_tokens = du.prompt_tokens + du.completion_tokens;
            const int cached =
                usage->cachedTokens > 0
                    ? usage->cachedTokens
                    : (req->continuation ? req->full_prompt_tokens_count -
                                               req->prompt_tokens_count
                                         : 0);
            du.cached_tokens = cached < 0 ? 0 : cached;
            const auto kNo = tt::utils::tokenizers::kNoTokenId;
            if (usage->thinkStart != kNo || usage->thinkEnd != kNo) {
              du.reasoning_tokens = usage->reasoning;
            }
            out.completion_usage = du;
          }

          const bool sent = sendChunk(out);
          if (isFinal) {
            signalDone();
            return;
          }
          if (!sent) {
            TT_LOG_WARN(
                "[DynamoRequestHandler] downstream send failed for task {}; "
                "aborting generation",
                req->task_id);
            pipeline->abortRequest(req->task_id);
          }
        };
        return cb;
      },
      std::move(handlers), std::move(cancelFn));
}

}  // namespace tt::dynamo
