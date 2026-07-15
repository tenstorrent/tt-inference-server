// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/disaggregation_service.hpp"

#include "config/settings.hpp"
#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/decode_slot_reservation.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"
#include "services/session_resolution.hpp"
#include "sockets/inter_server_service.hpp"
#include "sockets/socket_messages.hpp"
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

    if (tt::config::usePrefillFirstDisaggregation()) {
      TT_LOG_INFO(
          "[DisaggregationService] Prefill-first disaggregation enabled on "
          "decode: handling SlotReservationRequest");
      socketService->onSlotReservationRequest(
          [this](const tt::sockets::SlotReservationRequestMessage& message) {
            handleSlotReservationRequest(message);
          });
    }
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

          PrefillWorkContext work;
          work.request = std::move(request);
          work.fullPromptTokenIds.assign(message.tokenIds.begin(),
                                         message.tokenIds.end());
          work.decodeSlotId =
              message.slotId.value_or(tt::domain::INVALID_SLOT_ID);
          work.maxTokens = message.maxTokens;
          work.registrationHashCount =
              static_cast<uint32_t>(message.registrationHashes.size());

          work.request->prompt.emplace<std::vector<uint32_t>>(
              message.tokenIds.begin(), message.tokenIds.end());
          work.request->slotId = message.slotId;
          work.request->decode_position_id = message.decodePositionId;
          work.request->decode_skip_tokens = message.decodeSkipTokens;

          resolvePrefillSession(
              work.request, message.registrationHashes,
              [this, work = std::move(work), message]() mutable {
                const auto requestPtr = work.request;
                launchPrefillWork(
                    std::move(work),
                    [this, message, requestPtr, slotId = message.slotId,
                     maxTokens = message.maxTokens](
                        const LLMStreamChunk& response, bool /*isFinal*/) {
                      auto prefillResult =
                          tt::sockets::PrefillResultMessage(message.taskId);
                      prefillResult.slotId = slotId;
                      prefillResult.temperature = message.temperature;
                      prefillResult.topP = message.topP;
                      prefillResult.topK = message.topK;
                      prefillResult.fastMode = message.fastMode;

                      const auto finishReason =
                          response.choices.empty()
                              ? std::optional<std::string>{}
                              : response.choices.back().finish_reason;
                      const bool isError =
                          finishReason.has_value() &&
                          isErrorFinishReason(finishReason.value());
                      if (isError) {
                        TT_LOG_WARN(
                            "[DisaggregationService] Prefill error for task "
                            "{}, propagating to decode server",
                            message.taskId);
                        prefillResult.error = true;
                        const auto reason =
                            errorReasonFromFinishReason(finishReason.value());
                        prefillResult.generatedText =
                            tt::sockets::prefillErrorTextForReason(
                                reason, response.error.value_or("error"));
                      } else {
                        prefillResult.remainingTokens =
                            maxTokens.has_value() ? std::optional<int>(std::max(
                                                        0, maxTokens.value()))
                                                  : std::nullopt;
                        prefillResult.tokenIds.insert(
                            prefillResult.tokenIds.end(),
                            message.tokenIds.begin(), message.tokenIds.end());
                        prefillResult.generatedText =
                            response.choices.back().text;
                        prefillResult.cachedTokens =
                            response.cached_prompt_tokens.value_or(0);
                        prefillResult.migrationId =
                            requestPtr->migrationId.value_or(0);
                      }

                      socketService->sendPrefillResult(prefillResult);
                    });
              },
              [this, message, slotId = message.slotId](std::string_view error) {
                TT_LOG_WARN(
                    "[DisaggregationService] Session resolution failed for "
                    "taskId={}: {}",
                    message.taskId, error);
                auto prefillResult =
                    tt::sockets::PrefillResultMessage(message.taskId);
                prefillResult.slotId = slotId;
                prefillResult.error = true;
                prefillResult.generatedText =
                    tt::sockets::prefillErrorTextForReason(
                        LLMErrorReason::GENERIC, std::string(error));
                socketService->sendPrefillResult(prefillResult);
              });
        });

    socketService->onPrefillCancelled(
        [this](const tt::sockets::CancelPrefillMessage& message) {
          llmService->abortRequest(message.taskId);
        });

    if (tt::config::usePrefillFirstDisaggregation()) {
      TT_LOG_INFO(
          "[DisaggregationService] Prefill-first disaggregation enabled on "
          "prefill: handling SlotReservationResponse");
      socketService->onSlotReservationResponse(
          [this](const tt::sockets::SlotReservationResponseMessage& message) {
            handleSlotReservationResponse(message);
          });

      socketService->setConnectionLostCallback([this]() {
        std::vector<uint32_t> pendingTaskIds;
        pendingSlotReservations.forEach(
            [&pendingTaskIds](uint32_t taskId, const PrefillFirstPending&) {
              pendingTaskIds.push_back(taskId);
            });
        for (uint32_t taskId : pendingTaskIds) {
          failPrefillFirstPending(taskId, "connection lost");
        }
      });
    }
  }
}

DisaggregationService::~DisaggregationService() { stop(); }

void DisaggregationService::start() {
  if (socketService->isEnabled()) {
    socketService->start();
  }
}

void DisaggregationService::stop() { socketService->stop(); }

void DisaggregationService::handleSlotReservationRequest(
    const tt::sockets::SlotReservationRequestMessage& message) {
  if (!sessionManager) {
    TT_LOG_ERROR(
        "[DisaggregationService] Slot reservation taskId={} rejected: no "
        "session manager",
        message.taskId);
    tt::sockets::SlotReservationResponseMessage response;
    response.taskId = message.taskId;
    response.error = true;
    response.errorText = "session manager unavailable";
    socketService->sendSlotReservationResponse(response);
    return;
  }

  TT_LOG_INFO(
      "[DisaggregationService] Slot reservation request taskId={} "
      "prefillServerId={} hashes={}",
      message.taskId, message.prefillServerId,
      message.registrationHashes.size());

  decode_slot_reservation::ResolveInput input;
  input.taskId = message.taskId;
  input.registrationHashes = message.registrationHashes;
  if (message.hasPreviousResponseId) {
    input.previousResponseId = message.previousResponseId;
  }

  resolveDecodeDestinationSlot(
      *sessionManager, input, eventLoopThread.getLoop(),
      [this, taskId = message.taskId](
          decode_slot_reservation::DecodeDestinationSlot slot) {
        tt::sockets::SlotReservationResponseMessage response;
        response.taskId = taskId;
        response.hasSlot = slot.slotId != tt::domain::INVALID_SLOT_ID;
        response.slotId = slot.slotId;
        response.decodePositionId = slot.decodePositionId;
        response.decodeSkipTokens = slot.decodeSkipTokens;
        response.continuation = slot.continuation;
        response.accumulatedThinkTokens = slot.accumulatedThinkTokens;

        if (!socketService->sendSlotReservationResponse(response)) {
          TT_LOG_WARN(
              "[DisaggregationService] Failed to send slot reservation "
              "response taskId={} sessionId={}",
              taskId, slot.sessionId);
          if (!slot.sessionId.empty()) {
            sessionManager->releaseInFlight(slot.sessionId);
          }
        }
      },
      [this, taskId = message.taskId](std::string_view errorText) {
        tt::sockets::SlotReservationResponseMessage response;
        response.taskId = taskId;
        response.error = true;
        response.errorText = std::string(errorText);
        socketService->sendSlotReservationResponse(response);
      });
}

void DisaggregationService::launchPrefillWork(
    PrefillWorkContext work,
    std::function<void(const LLMStreamChunk&, bool)> onChunk) {
  auto request = std::move(work.request);
  const size_t fullPromptTokens = work.fullPromptTokenIds.size();
  const size_t trimmedPromptTokens =
      std::get<std::vector<uint32_t>>(request->prompt).size();
  // Cached (reused) prompt tokens = the leading prefix this prefill did NOT
  // recompute = what resolvePrefillSession trimmed off its own prefix-cache
  // hit (fullPrompt - remaining delta).
  //
  // Only the prefill-side trim counts: the prefill runner trims and recomputes
  // purely by its own prefix match, so any prefix the decode node reports
  // (decode_skip_tokens) but that prefill does not have gets recomputed here
  // — it is not cached. Folding decode_skip_tokens in (e.g. via max) would
  // over-report cached_tokens by exactly the tokens prefill re-prefilled.
  const int cachedTokens = static_cast<int>(
      fullPromptTokens >= trimmedPromptTokens
          ? fullPromptTokens - trimmedPromptTokens
          : 0);
  request->migrationStartPosition =
      request->decode_skip_tokens < cachedTokens
          ? 0u
          : static_cast<uint32_t>(cachedTokens);
  TT_LOG_DEBUG(
      "[DisaggregationService] taskId={} migrationStartPosition={} "
      "prefillMatchedTokens={} decodeSkipTokens={}",
      request->task_id, *request->migrationStartPosition, cachedTokens,
      request->decode_skip_tokens);

  // Capture the resolved sessionId by value: submitStreamingRequest hands the
  // request to the pipeline, so request->sessionId is no longer reliable by
  // the time this async callback fires.
  const std::string prefillSessionId = request->sessionId.value_or("");
  const uint32_t registrationHashCount = work.registrationHashCount;

  request->migrationId = tt::utils::MigrationIDGenerator::generate();
  TT_LOG_DEBUG(
      "[DisaggregationService] Assigned migrationId={} for taskId={}",
      request->migrationId.value_or(0), request->task_id);

  if (!request->migrationId.has_value() || request->migrationId.value() == 0) {
    TT_LOG_ERROR(
        "[DisaggregationService] migrationId is unset for taskId={} — prefill "
        "result will not correlate with KV transfer",
        request->task_id);
    throw std::runtime_error(
        "[DisaggregationService] migrationId must be set before submitting "
        "prefill request for taskId=" +
        std::to_string(request->task_id));
  }

  llmService->submitStreamingRequest(
      *request,
      [this, onChunk = std::move(onChunk), prefillSessionId, cachedTokens,
       registrationHashCount](const LLMStreamChunk& response, bool isFinal) {
        LLMStreamChunk chunk = response;
        if (cachedTokens > 0 && !chunk.cached_prompt_tokens.has_value()) {
          chunk.cached_prompt_tokens = cachedTokens;
        }
        onChunk(chunk, isFinal);

        // Release the prefill session's in-flight hold now that this one-shot
        // prefill (max_tokens=1) is done. Unlike the decode/HTTP transports,
        // the prefill path has no stream-end release, so without this the
        // session stays IN_FLIGHT forever — and evictOldSessions only reclaims
        // IDLE sessions, so the prefill pool fills with un-evictable sessions
        // and allocation eventually fails. Releasing to IDLE-but-cached also
        // lets the next turn's prefix cache match it.
        if (isFinal && !prefillSessionId.empty() && sessionManager) {
          const auto finishReason =
              response.choices.empty() ? std::optional<std::string>{}
                                       : response.choices.back().finish_reason;
          const bool isError = finishReason.has_value() &&
                               isErrorFinishReason(finishReason.value());
          // The prefill computed the whole prompt prefix, so all of its blocks
          // are now resident and safe to copy from. (Prefill is one-shot, so
          // there is no stream-end finalize to mark residency as on the decode
          // path.)
          if (!isError) {
            sessionManager->setResidentPrefixBlocks(prefillSessionId,
                                                    registrationHashCount);
          }
          sessionManager->releaseInFlight(prefillSessionId);
        }
      });
}

void DisaggregationService::failPrefillFirstPending(uint32_t taskId,
                                                    std::string_view errorText) {
  auto pending = pendingSlotReservations.take(taskId);
  if (!pending.has_value()) {
    return;
  }
  pending->callback(makeErrorChunk(taskId, std::string(errorText)),
                    /*isFinal=*/true);
}

void DisaggregationService::handlePrefillFirstStreamingRequest(
    LLMRequest& request, const std::vector<uint64_t>& registrationHashes,
    const StreamCallback& callback) {
  if (mode != tt::config::LLMMode::PREFILL_ONLY) {
    throw std::runtime_error(
        "[DisaggregationService] Prefill-first streaming requires prefill "
        "mode");
  }

  if (!socketService || !socketService->isConnected()) {
    throw std::runtime_error(
        "[DisaggregationService] Decode socket unavailable for slot "
        "reservation");
  }

  auto tokenIds = std::get<std::vector<uint32_t>>(request.prompt);
  PrefillFirstPending pending;
  pending.work.request = std::make_shared<LLMRequest>(request);
  pending.work.request->max_tokens = 1;
  pending.work.request->prompt = tokenIds;
  pending.work.fullPromptTokenIds = std::move(tokenIds);
  pending.work.maxTokens = request.max_tokens;
  pending.work.registrationHashCount =
      static_cast<uint32_t>(registrationHashes.size());
  pending.callback = callback;
  pending.registrationHashes = registrationHashes;

  tt::sockets::SlotReservationRequestMessage reservation;
  reservation.taskId = request.task_id;
  reservation.prefillServerId = tt::config::prefillServerId();
  reservation.registrationHashes = registrationHashes;
  reservation.promptTokenCount = request.prompt_tokens_count;
  if (request.previousResponseId.has_value() &&
      !request.previousResponseId->empty()) {
    reservation.hasPreviousResponseId = true;
    reservation.previousResponseId = *request.previousResponseId;
  }

  pendingSlotReservations.insert(request.task_id, std::move(pending));

  TT_LOG_INFO(
      "[DisaggregationService] Prefill-first slot reservation taskId={} "
      "hashes={} promptTokens={}",
      request.task_id, registrationHashes.size(),
      reservation.promptTokenCount);

  if (!socketService->sendSlotReservationRequest(reservation)) {
    pendingSlotReservations.erase(request.task_id);
    throw std::runtime_error(
        "[DisaggregationService] Failed to send slot reservation for taskId=" +
        std::to_string(request.task_id));
  }
}

void DisaggregationService::handleSlotReservationResponse(
    const tt::sockets::SlotReservationResponseMessage& message) {
  auto pending = pendingSlotReservations.take(message.taskId);
  if (!pending.has_value()) {
    TT_LOG_WARN(
        "[DisaggregationService] Slot reservation response for unknown "
        "taskId={}",
        message.taskId);
    return;
  }

  if (message.error || !message.hasSlot) {
    const std::string errorText =
        message.error ? message.errorText : "decode slot reservation denied";
    TT_LOG_WARN(
        "[DisaggregationService] Slot reservation failed taskId={}: {}",
        message.taskId, errorText);
    pending->callback(makeErrorChunk(message.taskId, errorText),
                      /*isFinal=*/true);
    return;
  }

  TT_LOG_INFO(
      "[DisaggregationService] Slot reservation granted taskId={} slotId={} "
      "decodePositionId={} continuation={}",
      message.taskId, message.slotId, message.decodePositionId,
      message.continuation);

  PrefillWorkContext work = std::move(pending->work);
  work.decodeSlotId = message.slotId;
  work.request->slotId = message.slotId;
  work.request->decode_position_id = message.decodePositionId;
  work.request->decode_skip_tokens = message.decodeSkipTokens;
  work.request->continuation = message.continuation;
  work.request->accumulated_think_tokens = message.accumulatedThinkTokens;
  auto registrationHashes = std::move(pending->registrationHashes);
  StreamCallback callback = pending->callback;

  resolvePrefillSession(
      work.request, registrationHashes,
      [this, work = std::move(work), callback]() mutable {
        launchPrefillWork(
            std::move(work),
            [callback](const LLMStreamChunk& response, bool isFinal) {
              callback(response, isFinal);
            });
      },
      [taskId = message.taskId, callback](std::string_view error) {
        TT_LOG_WARN(
            "[DisaggregationService] Prefill session resolution failed after "
            "slot reservation taskId={}: {}",
            taskId, error);
        callback(makeErrorChunk(taskId, std::string(error)), /*isFinal=*/true);
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
    socketService->sendPrefillCacheBlocksAdded(blockHashes(blockInfos));
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
          socketService->sendPrefillCacheBlocksAdded(blockHashes(infos));
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
    if (auto pending = pendingSlotReservations.take(taskId)) {
      LLMStreamChunk abortResponse{taskId};
      LLMChoice choice;
      choice.finish_reason = "abort";
      abortResponse.choices.push_back(std::move(choice));
      pending->callback(abortResponse, /*isFinal=*/true);
    }
    llmService->abortRequest(taskId);
  }
}

}  // namespace tt::services
