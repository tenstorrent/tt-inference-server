// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/disaggregation_service.hpp"

#include <cassert>

#include "config/settings.hpp"
#include "domain/llm/llm_request.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/logger.hpp"
#include "utils/mapper.hpp"

namespace tt::services {

using namespace tt::domain::llm;

DisaggregationService::DisaggregationService(
    tt::config::LLMMode mode, std::shared_ptr<LLMService> llmService,
    std::shared_ptr<sockets::InterServerService> socketService,
    std::shared_ptr<SessionManager> sessionMgr)
    : mode(mode),
      llmService(std::move(llmService)),
      socketService(std::move(socketService)),
      sessionManager(std::move(sessionMgr)) {
  eventLoopThread.run();
  setupSocketHandlers();
}

void DisaggregationService::setupSocketHandlers() {
  socketService->setHealthCheckCallback([](const std::string& serverId,
                                           double /*cpu*/, double /*memory*/,
                                           int tasks) {
    TT_LOG_INFO(
        "[DisaggregationService] Health check from {} (active_tasks={})",
        serverId, tasks);
  });

  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    socketService->onPrefillComplete(
        [this](const tt::sockets::PrefillResultMessage& message) {
          auto callback = streamCallbacks.get(message.task_id);
          if (!callback.has_value()) {
            TT_LOG_WARN("[DisaggregationService] No callback for task_id: {}",
                        message.task_id);
            return;
          }
          streamCallbacks.erase(message.task_id);

          if (message.error) {
            TT_LOG_ERROR(
                "[DisaggregationService] Prefill error received for task {}, "
                "propagating error to client",
                message.task_id);
            callback.value()(makeErrorChunk(message.task_id, "prefill error"),
                             /*isFinal=*/true);
            return;
          }

          auto response = LLMStreamChunk(message.task_id);
          LLMChoice choice;
          choice.text = message.generated_text;
          response.choices.push_back(std::move(choice));
          // Surface the prefill server's prefix-cache reuse count to the
          // transport's usage accounting (prompt_tokens_details.cached_tokens).
          response.cached_prompt_tokens = message.cached_tokens;

          callback.value()(response, false);

          bool continueDecode = !message.token_ids.empty() &&
                                (!message.remaining_tokens.has_value() ||
                                 message.remaining_tokens.value() > 0);
          if (continueDecode) {
            if (auto* reasoningParser = llmService->getReasoningParser()) {
              reasoningParser->initializeTask(message.task_id);
              reasoningParser->processToken(message.task_id,
                                            message.token_ids.back(),
                                            /*decodedText=*/"");
            }
            auto request = LLMRequest(message.task_id);
            request.disaggregated = true;
            // -2 because last token doesnt count, and we need current pos in kv
            // cache.
            request.kv_position_id =
                static_cast<uint32_t>(message.token_ids.size() - 2);
            request.prompt.emplace<std::vector<int>>(
                message.token_ids.end() - 1, message.token_ids.end());
            request.max_tokens = message.remaining_tokens;
            request.slotId = message.slot_id;
            // Restore the sampling subset echoed back from the prefill server.
            request.temperature = message.temperature;
            request.top_p = message.top_p;
            request.top_k = message.top_k;
            request.fast_mode = message.fast_mode;
            llmService->submitStreamingRequest(request, callback.value());
          } else {
            auto finalResponse = LLMStreamChunk(message.task_id);
            LLMChoice finalChoice;
            finalChoice.text = "";
            finalChoice.index = 0;
            finalChoice.finish_reason = "stop";
            finalResponse.choices.push_back(std::move(finalChoice));
            callback.value()(finalResponse, true);
          }
        });

    socketService->setConnectionLostCallback([this]() {
      streamCallbacks.forEach(
          [](uint32_t taskId, const StreamCallback& callback) {
            callback(makeErrorChunk(taskId, "connection lost"),
                     /*isFinal=*/true);
          });
      streamCallbacks.clear();
    });
  }

  if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    // On prefill runner death, drop the inter-server socket so the decode
    // side's connection-lost handler can fail in-flight streams instead of
    // hanging on requests nobody can answer.  Assumes WorkerManager does not
    // auto-restart workers; if that changes, the callback will fire again on
    // the replacement worker's first crash and stop a possibly-rearmed socket.
    if (auto* workerManager = llmService->getWorkerManager()) {
      workerManager->setWorkerDeathCallback(
          [socket = socketService](size_t workerIdx, pid_t pid) {
            TT_LOG_ERROR(
                "[DisaggregationService] Prefill runner (worker {}, PID {}) "
                "is down; disconnecting inter-server socket",
                workerIdx, pid);
            socket->stop();
          });
    }

    socketService->onPrefillRequested(
        [this](const tt::sockets::PrefillRequestMessage& message) {
          auto request = std::make_shared<LLMRequest>(message.task_id);
          request->max_tokens = 1;
          request->temperature = message.temperature;
          request->top_p = message.top_p;
          request->top_k = message.top_k;
          request->fast_mode = message.fast_mode;

          TT_LOG_DEBUG(
              "[DisaggregationService] Prefill request taskId={} "
              "registration_hashes={}",
              message.task_id, message.registration_hashes.size());

          auto maxTokens = message.max_tokens;

          request->prompt.emplace<std::vector<int>>(message.token_ids.begin(),
                                                    message.token_ids.end());
          auto slotId = message.slot_id;
          request->slotId = slotId;
          request->number_of_decode_skip_tokens =
              message.number_of_decode_skip_tokens;

          // Resolve prefix cache asynchronously: on HIT sets prefillSlotId
          // and trims prompt, on MISS allocates a new session first.
          resolvePrefillSession(
              request, message.registration_hashes,
              [this, request, message, maxTokens, slotId]() {
                // Tokens the prefill server served from its KV cache
                // (prefix-cache reuse) = prompt tokens trimmed off by
                // resolvePrefillSession (full prompt - remaining delta).
                const size_t fullPromptTokens = message.token_ids.size();
                const size_t trimmedPromptTokens =
                    std::get<std::vector<int>>(request->prompt).size();
                // Cached (reused) prompt tokens = the leading prefix that did
                // NOT need a fresh prefill. Two sources describe that same
                // prefix; take the larger:
                //   - decode_skip_tokens: the prefix the decode node already
                //     holds in its KV from earlier turns. The decode node does
                //     the multi-turn prefix match and ships the full prompt
                //     plus this skip count (it does not trim itself); the
                //     prefill runner honors it. This is the dominant source in
                //     chat.
                //   - prefill-side trim (fullPrompt - delta): when the prefill
                //     server independently hits its own prefix cache.
                const int prefillTrim = static_cast<int>(
                    fullPromptTokens >= trimmedPromptTokens
                        ? fullPromptTokens - trimmedPromptTokens
                        : 0);
                const int cachedTokens =
                    std::max(prefillTrim, message.number_of_decode_skip_tokens);
                // Capture the resolved sessionId by value:
                // submitStreamingRequest hands the request to the pipeline, so
                // request->sessionId is no longer reliable by the time this
                // async callback fires.
                const std::string prefillSessionId =
                    request->sessionId.value_or("");
                llmService->submitStreamingRequest(
                    *request,
                    [this, prefillSessionId, message, maxTokens, slotId,
                     cachedTokens](const LLMStreamChunk& response,
                                   bool /*isFinal*/) {
                      auto prefillResult =
                          tt::sockets::PrefillResultMessage(message.task_id);
                      prefillResult.slot_id = slotId;
                      prefillResult.temperature = message.temperature;
                      prefillResult.top_p = message.top_p;
                      prefillResult.top_k = message.top_k;
                      prefillResult.fast_mode = message.fast_mode;
                      prefillResult.cached_tokens = cachedTokens;

                      bool isError =
                          !response.choices.empty() &&
                          response.choices.back().finish_reason == "error";
                      if (isError) {
                        TT_LOG_WARN(
                            "[DisaggregationService] Prefill error for task "
                            "{}, propagating to decode server",
                            message.task_id);
                        prefillResult.error = true;
                        prefillResult.finished = true;
                      } else {
                        prefillResult.remaining_tokens =
                            maxTokens.has_value()
                                ? std::optional<int>(
                                      std::max(0, maxTokens.value() - 1))
                                : std::nullopt;
                        prefillResult.token_ids.insert(
                            prefillResult.token_ids.end(),
                            message.token_ids.begin(), message.token_ids.end());
                        if (response.choices.back().token_id.has_value()) {
                          prefillResult.token_ids.push_back(
                              response.choices.back().token_id.value());
                        }
                        prefillResult.generated_text =
                            response.choices.back().text;
                      }

                      socketService->sendPrefillResult(prefillResult);

                      // Release the prefill session's in-flight hold now that
                      // this one-shot prefill (max_tokens=1) is done. Unlike
                      // the decode/HTTP transports, the prefill path has no
                      // stream-end release, so without this the session stays
                      // IN_FLIGHT forever — and evictOldSessions only reclaims
                      // IDLE sessions, so the prefill pool fills with
                      // un-evictable sessions and allocation eventually fails.
                      // Releasing to IDLE-but-cached also lets the next turn's
                      // prefix cache match it. clearInFlight() is idempotent.
                      if (!prefillSessionId.empty()) {
                        if (auto* s =
                                sessionManager->getSession(prefillSessionId)) {
                          s->clearInFlight();
                        }
                      }
                    });
              },
              [this, message, slotId](std::string_view error) {
                TT_LOG_WARN(
                    "[DisaggregationService] Session resolution failed for "
                    "taskId={}: {}",
                    message.task_id, error);
                auto prefillResult =
                    tt::sockets::PrefillResultMessage(message.task_id);
                prefillResult.slot_id = slotId;
                prefillResult.error = true;
                prefillResult.finished = true;
                socketService->sendPrefillResult(prefillResult);
              });
        });

    socketService->onPrefillCancelled(
        [this](const tt::sockets::CancelPrefillMessage& message) {
          llmService->abortRequest(message.task_id);
        });
  }
}

DisaggregationService::~DisaggregationService() { stop(); }

void DisaggregationService::start() {
  if (socketService->isEnabled()) {
    socketService->start();
  }
}

void DisaggregationService::stop() { socketService->stop(); }

void DisaggregationService::applyDeltaPrompt(LLMRequest& req,
                                             uint32_t matchedTokens) {
  auto& tokens = std::get<std::vector<int>>(req.prompt);
  if (matchedTokens == 0 || matchedTokens >= tokens.size()) {
    return;
  }

  // The remaining (unmatched) prompt tokens that will be prefilled must be
  // aligned to 32 so the prefill kernel can operate on full tiles.  If the
  // remainder isn't divisible by 32, we pull back some matched tokens into
  // the delta to pad the remainder up to the next multiple of 32.
  constexpr uint32_t kAlignment = 32;
  const uint32_t totalTokens = static_cast<uint32_t>(tokens.size());
  const uint32_t remainder = totalTokens - matchedTokens;
  const uint32_t alignedRemainder =
      ((remainder + kAlignment - 1) / kAlignment) * kAlignment;

  // How many extra tokens we need to pull back from the matched prefix.
  const uint32_t pullBack = alignedRemainder - remainder;
  const uint32_t effectiveSkip =
      (pullBack <= matchedTokens) ? (matchedTokens - pullBack) : 0;

  if (effectiveSkip == 0) {
    TT_LOG_DEBUG(
        "[DisaggregationService] applyDeltaPrompt: matchedTokens={} "
        "remainder={} — cannot align, full prefill will run",
        matchedTokens, remainder);
    return;
  }

  TT_LOG_DEBUG(
      "[DisaggregationService] applyDeltaPrompt: matchedTokens={} "
      "effectiveSkip={} pullBack={} alignedRemainder={}",
      matchedTokens, effectiveSkip, pullBack, alignedRemainder);

  // Remove the first `effectiveSkip` tokens — they are already in KV cache.
  tokens.erase(tokens.begin(),
               tokens.begin() + static_cast<ptrdiff_t>(effectiveSkip));
  req.prompt_tokens_count = static_cast<int>(tokens.size());

  // kv_position_id points to the last valid KV cache position (0-indexed),
  // which is one less than the number of tokens we're reusing.
  req.kv_position_id = effectiveSkip - 1;
}

void DisaggregationService::resolvePrefillSession(
    std::shared_ptr<LLMRequest> request,
    const std::vector<uint64_t>& routingHashes,
    std::function<void()> onResolved,
    std::function<void(std::string_view)> onError) {
  if (!sessionManager || routingHashes.empty()) {
    onResolved();
    return;
  }

  // Convert hashes to BlockHashInfo for session manager calls.
  // Think token counts are 0 since prefill server doesn't track them.
  auto blockInfos = utils::hashesToBlockInfos(routingHashes);

  auto acquired = sessionManager->tryAcquireByPrefixHash(blockInfos, nullptr);

  if (acquired.has_value() && acquired->sessionFound) {
    TT_LOG_INFO(
        "[DisaggregationService] Prefill prefix cache HIT taskId={} "
        "sessionId={} slotId={} matchedTokens={}",
        request->task_id, acquired->sessionId, acquired->slotId,
        acquired->numberOfMatchedTokens);
    request->prefillSlotId = acquired->slotId;
    // Record the acquired session so the prefill completion can release its
    // in-flight hold (see clearInFlight below).
    request->sessionId = acquired->sessionId;
    applyDeltaPrompt(*request, acquired->numberOfMatchedTokens);
    sessionManager->registerPrefixHash(acquired->sessionId, blockInfos);
    onResolved();
  } else {
    // Check if there's a candidate slot worth copying from.
    std::optional<uint32_t> slotToCopyFrom;
    uint32_t copyMatchedTokens = 0;
    if (acquired.has_value() && !acquired->candidatesList.empty()) {
      auto copyCandidate =
          sessionManager->findASlotToCopyFrom(acquired->candidatesList);
      if (copyCandidate.has_value()) {
        uint32_t sourceSlot =
            sessionManager->getSlotIdBySessionId(copyCandidate->sessionId);
        if (sourceSlot != tt::domain::INVALID_SLOT_ID) {
          sessionManager->lockSlot(sourceSlot);
          slotToCopyFrom = sourceSlot;
          const size_t firstBlockSize = tt::config::kvCacheFirstBlockSize();
          const size_t blockSize = tt::config::kvCacheBlockSize();
          copyMatchedTokens = static_cast<uint32_t>(
              firstBlockSize +
              (copyCandidate->matchedBlocks > 1
                   ? (copyCandidate->matchedBlocks - 1) * blockSize
                   : 0));
          TT_LOG_INFO(
              "[DisaggregationService] Found slot to copy from: slotId={} "
              "matchedTokens={} for taskId={}",
              sourceSlot, copyMatchedTokens, request->task_id);
        }
      }
    }

    TT_LOG_INFO(
        "[DisaggregationService] Prefill prefix cache MISS taskId={} "
        "hashes={}, creating new session",
        request->task_id, routingHashes.size());

    sessionManager->createSession(
        [this, request, infos = std::move(blockInfos), sm = sessionManager,
         slotToCopyFrom, copyMatchedTokens, onResolved = std::move(onResolved)](
            const tt::domain::Session& session) mutable {
          if (slotToCopyFrom.has_value()) {
            sm->unlockSlot(*slotToCopyFrom);
          }
          TT_LOG_INFO(
              "[DisaggregationService] New session allocated taskId={} "
              "sessionId={} slotId={}",
              request->task_id, session.getSessionId(), session.getSlotId());
          sm->registerPrefixHash(session.getSessionId(), infos);
          request->sessionId = session.getSessionId();
          request->prefillSlotId =
              sm->acquireInFlight(session.getSessionId(), nullptr);

          // If copying, set continuation and kv_position_id on the request.
          if (slotToCopyFrom.has_value() && copyMatchedTokens > 0) {
            request->continuation = true;
            request->kv_position_id = copyMatchedTokens - 1;
            applyDeltaPrompt(*request, copyMatchedTokens);
          }
          onResolved();
        },
        [request, sm = sessionManager, slotToCopyFrom,
         onError = std::move(onError)](std::string_view errorMessage) {
          if (slotToCopyFrom.has_value()) {
            sm->unlockSlot(*slotToCopyFrom);
          }
          TT_LOG_WARN(
              "[DisaggregationService] Failed to create session for "
              "taskId={}: {}",
              request->task_id, errorMessage);
          onError(errorMessage);
        },
        /*eventLoop=*/eventLoopThread.getLoop(), blockInfos,
        /*slotId=*/std::nullopt, slotToCopyFrom);
  }
}

void DisaggregationService::handleStreamingRequest(
    LLMRequest& request, const std::vector<uint64_t>& registrationHashes,
    const StreamCallback& callback) {
  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    streamCallbacks.insert(request.task_id, callback);

    auto maxTokens = request.max_tokens;
    auto slotId = request.slotId;
    auto tokenIds = std::get<std::vector<int>>(request.prompt);
    int decodeSkipTokens = request.kv_position_id.has_value()
                               ? static_cast<int>(*request.kv_position_id + 1)
                               : 0;

    auto sent = socketService->sendPrefillRequest(
        request.task_id, registrationHashes,
        std::vector<int64_t>(tokenIds.begin(), tokenIds.end()), maxTokens,
        slotId, tt::utils::mapper::mapSamplingParams(request),
        decodeSkipTokens);

    if (!sent) {
      streamCallbacks.erase(request.task_id);
      throw std::runtime_error(
          "[DisaggregationService] Failed to send prefill request for "
          "task_id: " +
          std::to_string(request.task_id));
    }
  } else {
    throw std::runtime_error(
        "[DisaggregationService] Server must be in decode only mode to handle "
        "streaming requests");
  }
}

void DisaggregationService::abortRequest(uint32_t taskId) {
  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    auto callback = streamCallbacks.take(taskId);
    if (!callback.has_value()) {
      return;
    }

    bool sent = socketService->sendPrefillCancel(taskId);
    if (!sent) {
      TT_LOG_WARN(
          "[DisaggregationService] Failed to send prefill cancel for task_id: "
          "{}",
          taskId);
    }

    LLMStreamChunk abortResponse{taskId};
    LLMChoice choice;
    choice.finish_reason = "abort";
    abortResponse.choices.push_back(std::move(choice));
    callback.value()(abortResponse, /*isFinal=*/true);
    return;
  }

  if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    llmService->abortRequest(taskId);
  }
}

}  // namespace tt::services
