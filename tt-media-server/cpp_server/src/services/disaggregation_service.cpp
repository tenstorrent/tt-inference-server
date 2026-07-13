// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/disaggregation_service.hpp"

#include "domain/llm/llm_request.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"
#include "services/session_resolution.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"
#include "utils/mapper.hpp"

namespace tt::services {

using namespace tt::domain::llm;

namespace {

std::vector<uint64_t> blockHashes(
    const std::vector<tt::utils::BlockHashInfo>& blockInfos) {
  std::vector<uint64_t> hashes;
  hashes.reserve(blockInfos.size());
  for (const auto& block : blockInfos) {
    hashes.push_back(block.hash);
  }
  return hashes;
}

}  // namespace

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
  if (!socketService) {
    return;
  }

  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    socketService->onPrefillComplete(
        [this](const tt::sockets::PrefillResultMessage& message) {
          auto callback = streamCallbacks.get(message.taskId);
          if (!callback.has_value()) {
            TT_LOG_WARN("[DisaggregationService] No callback for task_id: {}",
                        message.taskId);
            return;
          }
          streamCallbacks.erase(message.taskId);

          if (message.error) {
            TT_LOG_ERROR(
                "[DisaggregationService] Prefill error received for task {}, "
                "propagating error to client",
                message.taskId);
            const auto reason =
                tt::sockets::errorReasonFromPrefillResult(message);
            callback.value()(makeErrorChunk(message.taskId,
                                            reason == LLMErrorReason::TIMEOUT
                                                ? "prefill timeout"
                                                : "prefill error",
                                            reason),
                             /*isFinal=*/true);
            return;
          }

          auto response = LLMStreamChunk(message.taskId);
          LLMChoice choice;
          choice.text = message.generatedText;
          response.choices.push_back(std::move(choice));
          // Surface the prefill server's prefix-cache reuse count to the
          // transport's usage accounting (prompt_tokens_details.cached_tokens).
          response.cached_prompt_tokens = message.cachedTokens;

          callback.value()(response, false);

          bool continueDecode = !message.tokenIds.empty() &&
                                (!message.remainingTokens.has_value() ||
                                 message.remainingTokens.value() > 0);
          if (continueDecode) {
            auto request = LLMRequest(message.taskId);
            request.disaggregated = true;
            request.migrationId = message.migrationId;
            // Intentional exception to the first-free-index convention: the
            // prefill-generated token is NOT migrated, so decode's first step
            // reprocesses the last prompt token. That token already occupies
            // KV index size-1, so decode resumes there rather than at the next
            // free slot, and the prompt is just that single trailing token.
            request.kv_position_id =
                static_cast<uint32_t>(message.tokenIds.size() - 1);
            request.prompt.emplace<std::vector<uint32_t>>(
                message.tokenIds.end() - 1, message.tokenIds.end());
            request.max_tokens = message.remainingTokens;
            request.slotId = message.slotId;
            // Restore the sampling subset echoed back from the prefill server.
            request.temperature = message.temperature;
            request.top_p = message.topP;
            request.top_k = message.topK;
            request.fast_mode = message.fastMode;
            llmService->submitStreamingRequest(request, callback.value());
          } else {
            auto finalResponse = LLMStreamChunk(message.taskId);
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
        [this, socket = socketService](
            const tt::sockets::PrefillRequestMessage& message) {
          handlePrefillRequest(
              message,
              [socket](const tt::sockets::PrefillResultMessage& result) {
                socket->sendPrefillResult(result);
              });
        });

    socketService->onPrefillCancelled(
        [this](const tt::sockets::CancelPrefillMessage& message) {
          llmService->abortRequest(message.taskId);
        });
  }
}

DisaggregationService::~DisaggregationService() { stop(); }

void DisaggregationService::start() {
  if (socketService && socketService->isEnabled()) {
    socketService->start();
  }
}

void DisaggregationService::stop() {
  if (socketService) {
    socketService->stop();
  }
}

void DisaggregationService::handlePrefillRequest(
    const tt::sockets::PrefillRequestMessage& message,
    std::function<void(const tt::sockets::PrefillResultMessage&)> callback) {
  if (mode != tt::config::LLMMode::PREFILL_ONLY) {
    auto prefillResult = tt::sockets::PrefillResultMessage(message.taskId);
    prefillResult.slotId = message.slotId;
    prefillResult.error = true;
    prefillResult.generatedText =
        "prefill request received by non-prefill worker";
    callback(prefillResult);
    return;
  }

  auto request = std::make_shared<LLMRequest>(message.taskId);
  request->max_tokens = 1;
  request->temperature = message.temperature;
  request->top_p = message.topP;
  request->top_k = message.topK;
  request->fast_mode = message.fastMode;

  TT_LOG_DEBUG(
      "[DisaggregationService] Prefill request taskId={} "
      "registration_hashes={}",
      message.taskId, message.registrationHashes.size());

  auto maxTokens = message.maxTokens;
  request->prompt.emplace<std::vector<uint32_t>>(message.tokenIds.begin(),
                                                 message.tokenIds.end());
  auto slotId = message.slotId;
  request->slotId = slotId;
  request->decode_position_id = message.decodePositionId;
  request->decode_skip_tokens = message.decodeSkipTokens;
  request->migrationId = tt::utils::MigrationIDGenerator::generate();

  auto resultCallback = std::make_shared<
      std::function<void(const tt::sockets::PrefillResultMessage&)>>(
      std::move(callback));

  resolvePrefillSession(
      request, message.registrationHashes,
      [this, request, message, maxTokens, slotId, resultCallback]() {
        const size_t fullPromptTokens = message.tokenIds.size();
        const size_t trimmedPromptTokens =
            std::get<std::vector<uint32_t>>(request->prompt).size();
        const int cachedTokens =
            static_cast<int>(fullPromptTokens >= trimmedPromptTokens
                                 ? fullPromptTokens - trimmedPromptTokens
                                 : 0);
        request->migrationStartPosition =
            request->decode_skip_tokens < cachedTokens
                ? 0u
                : static_cast<uint32_t>(cachedTokens);
        const std::string prefillSessionId = request->sessionId.value_or("");
        const uint64_t migrationId = request->migrationId.value_or(0);

        llmService->submitStreamingRequest(
            *request, [this, prefillSessionId, message, maxTokens, slotId,
                       cachedTokens, migrationId, resultCallback](
                          const LLMStreamChunk& response, bool /*isFinal*/) {
              auto prefillResult =
                  tt::sockets::PrefillResultMessage(message.taskId);
              prefillResult.slotId = slotId;
              prefillResult.temperature = message.temperature;
              prefillResult.topP = message.topP;
              prefillResult.topK = message.topK;
              prefillResult.fastMode = message.fastMode;
              prefillResult.cachedTokens = cachedTokens;
              prefillResult.migrationId = migrationId;

              const auto finishReason =
                  response.choices.empty()
                      ? std::optional<std::string>{}
                      : response.choices.back().finish_reason;
              const bool isError = finishReason.has_value() &&
                                   isErrorFinishReason(finishReason.value());
              if (isError) {
                prefillResult.error = true;
                const auto reason =
                    errorReasonFromFinishReason(finishReason.value());
                prefillResult.generatedText =
                    tt::sockets::prefillErrorTextForReason(
                        reason, response.error.value_or("error"));
              } else {
                prefillResult.remainingTokens =
                    maxTokens.has_value()
                        ? std::optional<int>(std::max(0, maxTokens.value()))
                        : std::nullopt;
                prefillResult.tokenIds.insert(prefillResult.tokenIds.end(),
                                              message.tokenIds.begin(),
                                              message.tokenIds.end());
                if (!response.choices.empty()) {
                  prefillResult.generatedText = response.choices.back().text;
                }
              }

              (*resultCallback)(prefillResult);

              if (!prefillSessionId.empty() && sessionManager) {
                if (!isError) {
                  sessionManager->setResidentPrefixBlocks(
                      prefillSessionId,
                      static_cast<uint32_t>(message.registrationHashes.size()));
                }
                sessionManager->releaseInFlight(prefillSessionId);
              }
            });
      },
      [message, slotId, resultCallback](std::string_view error) {
        TT_LOG_WARN(
            "[DisaggregationService] Session resolution failed for taskId={}: "
            "{}",
            message.taskId, error);
        auto prefillResult = tt::sockets::PrefillResultMessage(message.taskId);
        prefillResult.slotId = slotId;
        prefillResult.error = true;
        prefillResult.generatedText = tt::sockets::prefillErrorTextForReason(
            LLMErrorReason::GENERIC, std::string(error));
        (*resultCallback)(prefillResult);
      });
}

void DisaggregationService::resolvePrefillSession(
    std::shared_ptr<LLMRequest> request,
    const std::vector<uint64_t>& routingHashes,
    std::function<void()> onResolved,
    std::function<void(std::string_view)> onError) {
  if (!sessionManager) {
    TT_LOG_ERROR(
        "[DisaggregationService] No session manager configured; skipping "
        "prefix cache resolution for taskId={}",
        request->task_id);
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
    request->continuation = true;
    session_resolution::applyDeltaPrompt(
        *request, acquired->numberOfMatchedTokens,
        {.skipUnlessRegularMode = false,
         .setKvPositionId = true,
         .logPrefix = "[DisaggregationService]"});
    sessionManager->registerPrefixHash(acquired->sessionId, blockInfos);
    // Eagerly drop any resident tail past the common prefix: this turn's
    // new/diverged blocks are not computed yet. The full prefix is marked
    // resident again when this prefill completes (see prefill result callback).
    sessionManager->shrinkResidentPrefixToMatchedTokens(
        acquired->sessionId, acquired->numberOfMatchedTokens);
    if (socketService) {
      socketService->sendPrefillCacheBlocksAdded(blockHashes(blockInfos));
    }
    onResolved();
  } else {
    // Check if there's a candidate slot worth copying from.
    auto copyPlan = acquired.has_value()
                        ? session_resolution::prepareSlotCopy(
                              *sessionManager, acquired->candidatesList,
                              request->task_id, "[DisaggregationService]")
                        : std::nullopt;
    std::optional<uint32_t> slotToCopyFrom =
        copyPlan.has_value() ? std::make_optional(copyPlan->slotToCopyFrom)
                             : std::nullopt;
    uint32_t copyMatchedTokens =
        copyPlan.has_value() ? copyPlan->matchedTokens : 0;

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
          if (socketService) {
            socketService->sendPrefillCacheBlocksAdded(blockHashes(infos));
          }
          request->sessionId = session.getSessionId();
          request->prefillSlotId =
              sm->acquireInFlight(session.getSessionId(), nullptr);

          // If copying, set continuation and kv_position_id on the request.
          if (slotToCopyFrom.has_value() && copyMatchedTokens > 0) {
            request->continuation = true;
            request->kv_position_id = copyMatchedTokens;
            session_resolution::applyDeltaPrompt(
                *request, copyMatchedTokens,
                {.skipUnlessRegularMode = false,
                 .setKvPositionId = false,
                 .logPrefix = "[DisaggregationService]"});
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
    auto tokenIds = std::get<std::vector<uint32_t>>(request.prompt);
    // kv_position_id is the first free KV index (the matched prefix occupies
    // [0, kv_position_id)), which is exactly the position the prefill server
    // should resume writing from.
    int decodePositionId = request.kv_position_id.has_value()
                               ? static_cast<int>(*request.kv_position_id)
                               : 0;
    // Same reused prefix as decodePositionId but excluding the accumulated
    // think tokens that were folded into kv_position_id during session
    // resolution.
    int decodeSkipTokens = decodePositionId - request.accumulated_think_tokens;

    auto sent = socketService->sendPrefillRequest(
        request.task_id, registrationHashes, tokenIds, maxTokens, slotId,
        tt::utils::mapper::mapSamplingParams(request), decodePositionId,
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

    bool sent = socketService && socketService->sendPrefillCancel(taskId);
    if (!sent && socketService) {
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
