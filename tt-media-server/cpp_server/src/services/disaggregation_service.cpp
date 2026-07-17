// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/disaggregation_service.hpp"

#include <chrono>
#include <condition_variable>
#include <json/json.h>
#include <mutex>
#include <sstream>
#include <thread>

#include "config/settings.hpp"
#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "dynamo/etcd_client.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/decode_slot_reservation.hpp"
#include "services/disaggregation_contract_mapping.hpp"
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

constexpr auto kEtcdSlotPollInterval = std::chrono::milliseconds(50);
constexpr auto kEtcdSlotReservationTimeout = std::chrono::seconds(30);

std::vector<uint64_t> blockHashes(
    const std::vector<tt::utils::BlockHashInfo>& blockInfos) {
  std::vector<uint64_t> hashes;
  hashes.reserve(blockInfos.size());
  for (const auto& block : blockInfos) {
    hashes.push_back(block.hash);
  }
  return hashes;
}

std::string dumpJson(const Json::Value& value) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  return Json::writeString(builder, value);
}

Json::Value parseJsonOrEmpty(const std::string& text) {
  Json::Value root;
  Json::CharReaderBuilder builder;
  std::string errs;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  if (!reader->parse(text.data(), text.data() + text.size(), &root, &errs)) {
    return Json::Value(Json::objectValue);
  }
  return root;
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
  if (useEtcdSlotReservation()) {
    try {
      etcdClient = std::make_unique<tt::dynamo::EtcdClient>(
          tt::config::dynamoEtcdEndpoints());
      TT_LOG_INFO(
          "[DisaggregationService] Etcd slot reservation enabled "
          "(endpoints={})",
          tt::config::dynamoEtcdEndpoints());
    } catch (const std::exception& e) {
      TT_LOG_ERROR(
          "[DisaggregationService] Failed to create etcd client for slot "
          "reservation: {}",
          e.what());
    }
  }
  setupSocketHandlers();
}

bool DisaggregationService::useEtcdSlotReservation() const {
  return tt::config::dynamoRoutingEnabled();
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

    if (!useEtcdSlotReservation()) {
      TT_LOG_INFO(
          "[DisaggregationService] Registering SlotReservationRequest handler "
          "on decode (socket path)");
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

    if (!useEtcdSlotReservation()) {
      TT_LOG_INFO(
          "[DisaggregationService] Registering SlotReservationResponse handler "
          "on prefill (socket path)");
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
  if (socketService && socketService->isEnabled()) {
    socketService->start();
  }
  if (useEtcdSlotReservation() && mode == tt::config::LLMMode::DECODE_ONLY) {
    startEtcdSlotReservationListener();
  }
}

void DisaggregationService::stop() {
  stopEtcdSlotReservationListener();
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

  PrefillWorkContext work;
  work.request = request;
  work.fullPromptTokenIds.assign(message.tokenIds.begin(),
                                 message.tokenIds.end());
  work.decodeSlotId = slotId.value_or(tt::domain::INVALID_SLOT_ID);
  work.maxTokens = maxTokens;
  work.registrationHashCount =
      static_cast<uint32_t>(message.registrationHashes.size());

  resolvePrefillSession(
      request, message.registrationHashes,
      [this, work = std::move(work), message, resultCallback]() mutable {
        const auto requestPtr = work.request;
        launchPrefillWork(
            std::move(work),
            [message, requestPtr, slotId = message.slotId,
             maxTokens = message.maxTokens, resultCallback](
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
                prefillResult.cachedTokens =
                    response.cached_prompt_tokens.value_or(0);
                prefillResult.migrationId = requestPtr->migrationId.value_or(0);
              }

              (*resultCallback)(prefillResult);
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

void DisaggregationService::handleSlotReservationRequest(
    const tt::sockets::SlotReservationRequestMessage& message) {
  auto sendResponse =
      [this](const tt::sockets::SlotReservationResponseMessage& response) {
        if (socketService) {
          socketService->sendSlotReservationResponse(response);
        }
      };

  if (!sessionManager) {
    TT_LOG_ERROR(
        "[DisaggregationService] Slot reservation taskId={} rejected: no "
        "session manager",
        message.taskId);
    tt::sockets::SlotReservationResponseMessage response;
    response.taskId = message.taskId;
    response.error = true;
    response.errorText = "session manager unavailable";
    sendResponse(response);
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
      [this, taskId = message.taskId, sendResponse](
          decode_slot_reservation::DecodeDestinationSlot slot) {
        tt::sockets::SlotReservationResponseMessage response;
        response.taskId = taskId;
        response.hasSlot = slot.slotId != tt::domain::INVALID_SLOT_ID;
        response.slotId = slot.slotId;
        response.decodePositionId = slot.decodePositionId;
        response.decodeSkipTokens = slot.decodeSkipTokens;
        response.continuation = slot.continuation;
        response.accumulatedThinkTokens = slot.accumulatedThinkTokens;

        if (socketService &&
            !socketService->sendSlotReservationResponse(response)) {
          TT_LOG_WARN(
              "[DisaggregationService] Failed to send slot reservation "
              "response taskId={} sessionId={}",
              taskId, slot.sessionId);
          if (!slot.sessionId.empty()) {
            sessionManager->releaseInFlight(slot.sessionId);
          }
        } else if (!socketService) {
          sendResponse(response);
        }
      },
      [taskId = message.taskId, sendResponse](std::string_view errorText) {
        tt::sockets::SlotReservationResponseMessage response;
        response.taskId = taskId;
        response.error = true;
        response.errorText = std::string(errorText);
        sendResponse(response);
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
  if (pending->resultCallback.has_value()) {
    auto result = tt::sockets::PrefillResultMessage(taskId);
    result.error = true;
    result.generatedText = std::string(errorText);
    (*pending->resultCallback)(result);
    return;
  }
  pending->callback(makeErrorChunk(taskId, std::string(errorText)),
                    /*isFinal=*/true);
}

void DisaggregationService::handlePrefillFirstRequest(
    LLMRequest& request, const std::vector<uint64_t>& registrationHashes,
    std::function<void(const tt::sockets::PrefillResultMessage&)> callback) {
  enqueuePrefillFirst(request, registrationHashes, StreamCallback{},
                      std::move(callback));
}

void DisaggregationService::handlePrefillFirstStreamingRequest(
    LLMRequest& request, const std::vector<uint64_t>& registrationHashes,
    const StreamCallback& callback) {
  enqueuePrefillFirst(request, registrationHashes, callback, std::nullopt);
}

void DisaggregationService::enqueuePrefillFirst(
    LLMRequest& request, const std::vector<uint64_t>& registrationHashes,
    StreamCallback streamCallback,
    std::optional<std::function<void(const tt::sockets::PrefillResultMessage&)>>
        resultCallback) {
  if (mode != tt::config::LLMMode::PREFILL_ONLY) {
    throw std::runtime_error(
        "[DisaggregationService] Prefill-first streaming requires prefill "
        "mode");
  }

  auto tokenIds = std::get<std::vector<uint32_t>>(request.prompt);
  PrefillFirstPending pending;
  pending.work.request = std::make_shared<LLMRequest>(request);
  pending.work.request->max_tokens = 1;
  pending.work.request->prompt = tokenIds;
  pending.work.fullPromptTokenIds = tokenIds;
  pending.work.maxTokens = request.max_tokens;
  pending.work.registrationHashCount =
      static_cast<uint32_t>(registrationHashes.size());
  pending.callback = std::move(streamCallback);
  pending.registrationHashes = registrationHashes;
  pending.resultCallback = std::move(resultCallback);

  pendingSlotReservations.insert(request.task_id, std::move(pending));

  TT_LOG_INFO(
      "[DisaggregationService] Prefill-first slot reservation taskId={} "
      "hashes={} promptTokens={} transport={}",
      request.task_id, registrationHashes.size(), request.prompt_tokens_count,
      useEtcdSlotReservation() ? "etcd" : "socket");

  try {
    if (useEtcdSlotReservation()) {
      reserveDecodeSlotViaEtcd(request.task_id, registrationHashes, request);
    } else {
      reserveDecodeSlotViaSocket(request.task_id, registrationHashes, request);
    }
  } catch (...) {
    pendingSlotReservations.erase(request.task_id);
    throw;
  }
}

void DisaggregationService::reserveDecodeSlotViaSocket(
    uint32_t taskId, const std::vector<uint64_t>& registrationHashes,
    const LLMRequest& request) {
  if (!socketService || !socketService->isConnected()) {
    throw std::runtime_error(
        "[DisaggregationService] Decode socket unavailable for slot "
        "reservation");
  }

  tt::sockets::SlotReservationRequestMessage reservation;
  reservation.taskId = taskId;
  reservation.prefillServerId = tt::config::prefillServerId();
  reservation.registrationHashes = registrationHashes;
  reservation.promptTokenCount = request.prompt_tokens_count;
  if (request.previousResponseId.has_value() &&
      !request.previousResponseId->empty()) {
    reservation.hasPreviousResponseId = true;
    reservation.previousResponseId = *request.previousResponseId;
  }

  if (!socketService->sendSlotReservationRequest(reservation)) {
    throw std::runtime_error(
        "[DisaggregationService] Failed to send slot reservation for taskId=" +
        std::to_string(taskId));
  }
}

void DisaggregationService::reserveDecodeSlotViaEtcd(
    uint32_t taskId, const std::vector<uint64_t>& registrationHashes,
    const LLMRequest& request) {
  if (!etcdClient) {
    throw std::runtime_error(
        "[DisaggregationService] Etcd client unavailable for slot reservation");
  }

  auto peers = discoverDecodePeers();
  auto peer = selectDecodePeer(peers);
  if (!peer.has_value()) {
    throw std::runtime_error(
        "[DisaggregationService] No decode peers discovered in etcd for slot "
        "reservation");
  }

  {
    auto pending = pendingSlotReservations.get(taskId);
    if (pending.has_value()) {
      PrefillFirstPending updated = *pending;
      updated.decodeInstanceId = peer->instanceIdHex;
      pendingSlotReservations.insert(taskId, std::move(updated));
    }
  }

  Json::Value body(Json::objectValue);
  body["task_id"] = taskId;
  body["prefill_server_id"] = tt::config::prefillServerId();
  body["prompt_token_count"] = request.prompt_tokens_count;
  Json::Value hashes(Json::arrayValue);
  for (uint64_t hash : registrationHashes) {
    hashes.append(Json::UInt64(hash));
  }
  body["registration_hashes"] = std::move(hashes);
  if (request.previousResponseId.has_value() &&
      !request.previousResponseId->empty()) {
    body["previous_response_id"] = *request.previousResponseId;
  }

  const std::string requestKey =
      etcdSlotRequestKey(peer->instanceIdHex, taskId);
  const std::string responseKey =
      etcdSlotResponseKey(peer->instanceIdHex, taskId);
  try {
    etcdClient->deleteRange(responseKey);
  } catch (...) {
  }
  etcdClient->put(requestKey, dumpJson(body));

  TT_LOG_INFO(
      "[DisaggregationService] Etcd slot reservation published taskId={} "
      "decodeInstance={} key={}",
      taskId, peer->instanceIdHex, requestKey);

  // Poll for the decode response on a background thread so we don't block
  // the Dynamo request loop.
  std::thread([this, taskId, requestKey, responseKey]() {
    const auto deadline =
        std::chrono::steady_clock::now() + kEtcdSlotReservationTimeout;
    while (std::chrono::steady_clock::now() < deadline) {
      try {
        auto raw = etcdClient->get(responseKey);
        if (raw.has_value()) {
          try {
            etcdClient->deleteRange(requestKey);
            etcdClient->deleteRange(responseKey);
          } catch (...) {
          }
          const Json::Value json = parseJsonOrEmpty(*raw);
          tt::sockets::SlotReservationResponseMessage response;
          response.taskId = taskId;
          response.error = json.get("error", false).asBool();
          response.errorText = json.get("error_text", "").asString();
          response.hasSlot = json.get("has_slot", false).asBool();
          response.slotId = json.get("slot_id", tt::domain::INVALID_SLOT_ID)
                                .asUInt();
          response.decodePositionId = json.get("decode_position_id", 0).asInt();
          response.decodeSkipTokens = json.get("decode_skip_tokens", 0).asInt();
          response.continuation = json.get("continuation", false).asBool();
          response.accumulatedThinkTokens =
              json.get("accumulated_think_tokens", 0).asInt();
          handleSlotReservationResponse(response);
          return;
        }
      } catch (const std::exception& e) {
        TT_LOG_WARN(
            "[DisaggregationService] Etcd slot reservation poll error "
            "taskId={}: {}",
            taskId, e.what());
      }
      std::this_thread::sleep_for(kEtcdSlotPollInterval);
    }
    try {
      etcdClient->deleteRange(requestKey);
      etcdClient->deleteRange(responseKey);
    } catch (...) {
    }
    failPrefillFirstPending(taskId, "etcd slot reservation timed out");
  }).detach();
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

  applySlotReservationAndLaunch(std::move(*pending), message);
}

void DisaggregationService::applySlotReservationAndLaunch(
    PrefillFirstPending pending,
    const tt::sockets::SlotReservationResponseMessage& message) {
  if (message.error || !message.hasSlot) {
    const std::string errorText =
        message.error ? message.errorText : "decode slot reservation denied";
    TT_LOG_WARN(
        "[DisaggregationService] Slot reservation failed taskId={}: {}",
        message.taskId, errorText);
    if (pending.resultCallback.has_value()) {
      auto result = tt::sockets::PrefillResultMessage(message.taskId);
      result.error = true;
      result.generatedText = errorText;
      (*pending.resultCallback)(result);
    } else if (pending.callback) {
      pending.callback(makeErrorChunk(message.taskId, errorText),
                       /*isFinal=*/true);
    }
    return;
  }

  TT_LOG_INFO(
      "[DisaggregationService] Slot reservation granted taskId={} slotId={} "
      "decodePositionId={} continuation={} decodeInstance={}",
      message.taskId, message.slotId, message.decodePositionId,
      message.continuation, pending.decodeInstanceId);

  PrefillWorkContext work = std::move(pending.work);
  work.decodeSlotId = message.slotId;
  work.request->slotId = message.slotId;
  work.request->decode_position_id = message.decodePositionId;
  work.request->decode_skip_tokens = message.decodeSkipTokens;
  work.request->continuation = message.continuation;
  work.request->accumulated_think_tokens = message.accumulatedThinkTokens;
  auto registrationHashes = std::move(pending.registrationHashes);
  StreamCallback streamCallback = std::move(pending.callback);
  auto resultCallback = std::move(pending.resultCallback);
  const uint32_t taskId = message.taskId;
  const auto fullPromptTokenIds = work.fullPromptTokenIds;
  const auto maxTokens = work.maxTokens;
  const auto temperature = work.request->temperature;
  const auto topP = work.request->top_p;
  const auto topK = work.request->top_k;
  const bool fastMode = work.request->fast_mode;
  const auto slotId = work.request->slotId;

  resolvePrefillSession(
      work.request, registrationHashes,
      [this, work = std::move(work), streamCallback,
       resultCallback = std::move(resultCallback), taskId, fullPromptTokenIds,
       maxTokens, temperature, topP, topK, fastMode,
       slotId]() mutable {
        const auto requestPtr = work.request;
        launchPrefillWork(
            std::move(work),
            [streamCallback, resultCallback, taskId, fullPromptTokenIds,
             maxTokens, temperature, topP, topK, fastMode, slotId,
             requestPtr](const LLMStreamChunk& response, bool isFinal) {
              if (resultCallback.has_value()) {
                if (!isFinal && !response.choices.empty() &&
                    response.choices.back().finish_reason.has_value() &&
                    isErrorFinishReason(
                        *response.choices.back().finish_reason)) {
                  // fall through to build error result below
                } else if (!isFinal) {
                  return;
                }
                auto prefillResult = tt::sockets::PrefillResultMessage(taskId);
                prefillResult.slotId = slotId;
                prefillResult.temperature = temperature;
                prefillResult.topP = topP;
                prefillResult.topK = topK;
                prefillResult.fastMode = fastMode;

                const auto finishReason =
                    response.choices.empty()
                        ? std::optional<std::string>{}
                        : response.choices.back().finish_reason;
                const bool isError =
                    finishReason.has_value() &&
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
                  prefillResult.tokenIds = fullPromptTokenIds;
                  if (!response.choices.empty()) {
                    prefillResult.generatedText = response.choices.back().text;
                  }
                  prefillResult.cachedTokens =
                      response.cached_prompt_tokens.value_or(0);
                  prefillResult.migrationId =
                      requestPtr->migrationId.value_or(0);
                }
                (*resultCallback)(prefillResult);
                return;
              }
              if (streamCallback) {
                streamCallback(response, isFinal);
              }
            });
      },
      [taskId, streamCallback, resultCallback = std::move(resultCallback)](
          std::string_view error) {
        TT_LOG_WARN(
            "[DisaggregationService] Prefill session resolution failed after "
            "slot reservation taskId={}: {}",
            taskId, error);
        if (resultCallback.has_value()) {
          auto result = tt::sockets::PrefillResultMessage(taskId);
          result.error = true;
          result.generatedText = std::string(error);
          (*resultCallback)(result);
        } else if (streamCallback) {
          streamCallback(makeErrorChunk(taskId, std::string(error)),
                         /*isFinal=*/true);
        }
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

    auto sent = socketService &&
                socketService->sendPrefillRequest(
                    request.task_id, registrationHashes, tokenIds, maxTokens,
                    slotId, tt::utils::mapper::mapSamplingParams(request),
                    decodePositionId, decodeSkipTokens);

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
    if (auto pending = pendingSlotReservations.take(taskId)) {
      if (pending->resultCallback.has_value()) {
        auto result = tt::sockets::PrefillResultMessage(taskId);
        result.error = true;
        result.generatedText = "aborted";
        (*pending->resultCallback)(result);
      } else if (pending->callback) {
        LLMStreamChunk abortResponse{taskId};
        LLMChoice choice;
        choice.finish_reason = "abort";
        abortResponse.choices.push_back(std::move(choice));
        pending->callback(abortResponse, /*isFinal=*/true);
      }
    }
    llmService->abortRequest(taskId);
  }
}

std::string DisaggregationService::etcdSlotRequestPrefix() const {
  return "v1/tt/slot_reservation/" + tt::config::dynamoNamespace() + "/";
}

std::string DisaggregationService::etcdSlotRequestKey(
    const std::string& decodeInstanceId, uint32_t taskId) const {
  return etcdSlotRequestPrefix() + decodeInstanceId + "/" +
         std::to_string(taskId) + "/req";
}

std::string DisaggregationService::etcdSlotResponseKey(
    const std::string& decodeInstanceId, uint32_t taskId) const {
  return etcdSlotRequestPrefix() + decodeInstanceId + "/" +
         std::to_string(taskId) + "/resp";
}

std::vector<DisaggregationService::DecodePeer>
DisaggregationService::discoverDecodePeers() const {
  std::vector<DecodePeer> peers;
  if (!etcdClient) {
    return peers;
  }
  const std::string prefix = "v1/instances/" + tt::config::dynamoNamespace() +
                             "/decode/" + tt::config::dynamoEndpointName() +
                             "/";
  try {
    for (const auto& [key, value] : etcdClient->getPrefix(prefix)) {
      const Json::Value json = parseJsonOrEmpty(value);
      DecodePeer peer;
      if (json.isMember("instance_id") && json["instance_id"].isIntegral()) {
        peer.instanceId = json["instance_id"].asUInt64();
        std::ostringstream oss;
        oss << std::hex << peer.instanceId;
        peer.instanceIdHex = oss.str();
      } else {
        const auto slash = key.find_last_of('/');
        if (slash != std::string::npos) {
          peer.instanceIdHex = key.substr(slash + 1);
        }
      }
      if (json.isMember("transport") && json["transport"].isObject()) {
        peer.tcpAddress = json["transport"].get("tcp", "").asString();
      }
      if (!peer.instanceIdHex.empty()) {
        peers.push_back(std::move(peer));
      }
    }
  } catch (const std::exception& e) {
    TT_LOG_WARN("[DisaggregationService] etcd decode peer discovery failed: {}",
                e.what());
  }
  return peers;
}

std::optional<DisaggregationService::DecodePeer>
DisaggregationService::selectDecodePeer(
    const std::vector<DecodePeer>& peers) const {
  if (peers.empty()) {
    return std::nullopt;
  }
  for (const auto& peer : peers) {
    if (!peer.tcpAddress.empty()) {
      return peer;
    }
  }
  return peers.front();
}

void DisaggregationService::startEtcdSlotReservationListener() {
  if (!etcdClient || etcdListenerRunning.exchange(true)) {
    return;
  }
  TT_LOG_INFO(
      "[DisaggregationService] Starting etcd slot-reservation listener "
      "prefix={}",
      etcdSlotRequestPrefix());
  etcdListenerThread =
      std::thread([this]() { etcdSlotReservationListenLoop(); });
}

void DisaggregationService::stopEtcdSlotReservationListener() {
  if (!etcdListenerRunning.exchange(false)) {
    return;
  }
  if (etcdListenerThread.joinable()) {
    etcdListenerThread.join();
  }
}

void DisaggregationService::etcdSlotReservationListenLoop() {
  while (etcdListenerRunning.load()) {
    if (localDecodeInstanceId.empty()) {
      std::this_thread::sleep_for(kEtcdSlotPollInterval);
      continue;
    }
    try {
      const std::string prefix =
          etcdSlotRequestPrefix() + localDecodeInstanceId + "/";
      for (const auto& [key, value] : etcdClient->getPrefix(prefix)) {
        if (key.size() < 4 || key.substr(key.size() - 4) != "/req") {
          continue;
        }
        processEtcdSlotReservationRequest(key, value);
      }
    } catch (const std::exception& e) {
      TT_LOG_WARN(
          "[DisaggregationService] etcd slot-reservation listener error: {}",
          e.what());
    }
    std::this_thread::sleep_for(kEtcdSlotPollInterval);
  }
}

void DisaggregationService::processEtcdSlotReservationRequest(
    const std::string& requestKey, const std::string& requestJson) {
  const Json::Value json = parseJsonOrEmpty(requestJson);
  const uint32_t taskId = json.get("task_id", 0).asUInt();
  if (taskId == 0 || !sessionManager) {
    return;
  }

  const std::string responseKey =
      etcdSlotResponseKey(localDecodeInstanceId, taskId);

  try {
    if (etcdClient->get(responseKey).has_value()) {
      return;
    }
  } catch (...) {
  }

  decode_slot_reservation::ResolveInput input;
  input.taskId = taskId;
  if (json.isMember("registration_hashes") &&
      json["registration_hashes"].isArray()) {
    for (const auto& hash : json["registration_hashes"]) {
      input.registrationHashes.push_back(hash.asUInt64());
    }
  }
  const std::string previousResponseId =
      json.get("previous_response_id", "").asString();
  if (!previousResponseId.empty()) {
    input.previousResponseId = previousResponseId;
  }

  TT_LOG_INFO(
      "[DisaggregationService] Etcd slot reservation request taskId={} "
      "hashes={} key={}",
      taskId, input.registrationHashes.size(), requestKey);

  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  tt::sockets::SlotReservationResponseMessage response;
  response.taskId = taskId;

  resolveDecodeDestinationSlot(
      *sessionManager, input, eventLoopThread.getLoop(),
      [&](decode_slot_reservation::DecodeDestinationSlot slot) {
        std::lock_guard<std::mutex> lock(mu);
        response.hasSlot = slot.slotId != tt::domain::INVALID_SLOT_ID;
        response.slotId = slot.slotId;
        response.decodePositionId = slot.decodePositionId;
        response.decodeSkipTokens = slot.decodeSkipTokens;
        response.continuation = slot.continuation;
        response.accumulatedThinkTokens = slot.accumulatedThinkTokens;
        done = true;
        cv.notify_one();
      },
      [&](std::string_view errorText) {
        std::lock_guard<std::mutex> lock(mu);
        response.error = true;
        response.errorText = std::string(errorText);
        done = true;
        cv.notify_one();
      });

  {
    std::unique_lock<std::mutex> lock(mu);
    if (!cv.wait_for(lock, kEtcdSlotReservationTimeout,
                     [&]() { return done; })) {
      response.error = true;
      response.errorText = "decode slot reservation timed out";
    }
  }

  Json::Value out(Json::objectValue);
  out["task_id"] = response.taskId;
  out["error"] = response.error;
  out["error_text"] = response.errorText;
  out["has_slot"] = response.hasSlot;
  out["slot_id"] = response.slotId;
  out["decode_position_id"] = response.decodePositionId;
  out["decode_skip_tokens"] = response.decodeSkipTokens;
  out["continuation"] = response.continuation;
  out["accumulated_think_tokens"] = response.accumulatedThinkTokens;

  try {
    etcdClient->put(responseKey, dumpJson(out));
    etcdClient->deleteRange(requestKey);
  } catch (const std::exception& e) {
    TT_LOG_WARN(
        "[DisaggregationService] Failed to publish etcd slot reservation "
        "response taskId={}: {}",
        taskId, e.what());
  }
}

}  // namespace tt::services
